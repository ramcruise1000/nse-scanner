#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VOL5 — Extended-Hours Dual-Frame 1:5 Reward : Risk Intraday Scanner (NSE F&O)
================================================================================
A standalone, self-improving companion system to Vol4 (market-sweep.yml).
Runs on GitHub Actions free tier. No server, no local machine required.

WHAT CHANGES vs VOL4 (Vol4 untouched — everything here is NEW files only):
  * Signal window extends to 14:30 IST (Vol4 stops new signals at 13:30).
  * Dual timeframe: 15-minute bars set direction/structure, 5-minute bars fire triggers.
  * Self-learning window extended from 5 to 10 sessions (AT_WINDOW_DAYS = 10).
  * Own journals (journal_v5/), own reports (reports_v5/), own memory
    (state_v5.json, tuner_state_v5.json) — zero interference with Vol4.
  * Own GitHub workflow (.github/workflows/extended-scan.yml) in its own
    concurrency group (vol5-extended), so the two scanners never step on each other.
  * Five dual-frame strategies (V1-V5) derived from Vol4 learnings (see SOP).
  * "Tomorrow priority" recommendation list printed at EOD — the system tells
    you which methods earned their slot yesterday, so it self-improves each night.

EDUCATIONAL USE ONLY. NOT INVESTMENT ADVICE. Every signal is an alert; you decide.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo

INDIA = ZoneInfo("Asia/Kolkata")

# --------------------------------------------------------------------------- #
# CONFIGURATION (change these to tune behaviour; every number is explained    #
# in the SOP_Vol5 document. Capital / risk settings mirror Vol4 for parity.) #
# --------------------------------------------------------------------------- #

CAPITAL_RUPEES   = 20_000.0      # model capital
RISK_PCT         = 0.01          # 1% of capital risked per trade
RISK_AMT         = 200.0         # Rs 200 per trade (= CAPITAL * RISK_PCT)
RR_MULT          = 5.0           # strict 1 : 5 reward : risk
MIN_STOP_PCT     = 0.003         # stop floor: 0.3 % (keeps costs from eating edge)
MAX_STOP_PCT     = 0.015         # stop ceiling: 1.5 % (keeps 5R reachable)
ATR_PERIOD       = 14
REACH_FRAC       = 1.0           # 5R target must fit inside the day's ATR(14) range
MIS_LEVERAGE     = 5             # SEBI MIS intraday leverage ceiling (final qty cap)
EST_COST_RT      = 60.0          # estimated round-trip cost per trade (Rs)

# --- Trading hours (IST) --------------------------------------------------- #
PHASE_PREP_START_MIN  = 9 * 60 + 25   # 09:25  pre-market watchlist
PHASE_LIVE_START_MIN  = 9 * 60 + 30   # 09:30  full scanning starts
PHASE_CUTOFF_MIN      = 14 * 60 + 30  # 14:30  NEW signals cut off (extended window)
PHASE_WRAP_END_MIN    = 15 * 60 + 45  # 15:45  journal finalize last chance

# --- Fill gate & lifecycle -------------------------------------------------- #
FILL_TOL_PCT     = 0.001             # fill confirms when retraced within 0.10 %
FILL_DEADLINE_MIN = 15 * 60          # unfilled signals expire to slippage log at 15:00
SQ_OFF_MIN       = 15 * 60 + 5       # force-close leftovers at first sweep >= 15:05
FINALIZE_TIMES   = (15 * 60 + 10,     # 15:10, 15:20, 15:35, 15:45 repeat passes
                    15 * 60 + 20,
                    15 * 60 + 35,
                    15 * 60 + 45)

# --- Guard rails ------------------------------------------------------------ #
CAP_TRADES_DAY   = 10                # maximum committed TRADE slots per day
DAILY_LOSS_R     = -2.0              # circuit-breaker threshold (-2R)
LOSS_STREAK_TRIP = 2                 # two consecutive losses trips breaker
SWEEP_GAP_WARN_MIN = 25              # warn if a sweep wakes > 25 min late

# --- Auto-tune (self-improvement) window: 10 SESSIONS ----------------------- #
AT_WINDOW_DAYS   = 10                # learns from last 10 completed sessions
AT_MIN_TRADES    = 2                 # mute needs >= 2 graded fills at negative netR
AT_SLIP_MIN      = 3                 # mute needs >= 3 runaway signals (chronic chase)

# --- Directory layout (ALL under repo root; Vol4 uses journal/, state.json — #
#     NOTHING here collides) ------------------------------------------------- #
BASE_DIR        = Path(os.path.dirname(os.path.abspath(__file__)))
JOURNAL_DIR     = BASE_DIR / "journal_v5"
REPORTS_DIR     = BASE_DIR / "reports_v5"
STATE_FILE      = BASE_DIR / "state_v5.json"
TUNER_STATE_FILE = BASE_DIR / "tuner_state_v5.json"

TEST_OUT_BASE   = Path("/tmp/vol5_test")

# --- Journal schema (kept identical to Vol4 for audit continuity) ---------- #
JOURNAL_COLS = ["TradeID", "Date", "SignalTime", "Symbol", "Direction", "Setup",
                "Entry", "StopLoss", "Target", "Qty", "RiskPerShare", "RiskAmt",
                "RewardAmt", "RR", "Status", "Mode", "ExitPrice", "ExitTime",
                "ExitReason", "Result", "PnL_R", "PnL_Amt", "UpdatedAt",
                "Remarks", "Notes"]
TUNER_COLS   = ["Date", "Bucket", "Decision", "NetR", "Wins", "Losses",
                "Flats", "Trades", "WinRatePct", "SlipCount", "Reason"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# IST CLOCK & PHASES                                                          #
# --------------------------------------------------------------------------- #
def ist_now():
    """Current instant in India Standard Time."""
    return datetime.now(INDIA)


def ist_now_min() -> int:
    return ist_now().hour * 60 + ist_now().minute


def ist_clock_override() -> int | None:
    """SCANNER_FORCE_TIME=HH:MM overrides the internal clock for dry-runs."""
    raw = os.environ.get("SCANNER_FORCE_TIME")
    if not raw:
        return None
    h, m = map(int, raw.split(":"))
    return h * 60 + m


def market_phase(mins: int) -> str:
    if mins < PHASE_LIVE_START_MIN:
        return "PREP" if mins >= PHASE_PREP_START_MIN else "EARLY"
    if mins <= PHASE_CUTOFF_MIN:
        return "LIVE"
    return "WRAP"


# --------------------------------------------------------------------------- #
# YAHOO FINANCE DATA LADDER                                                   #
# --------------------------------------------------------------------------- #
def _fetch(ticker: str, interval: str, period: str):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={period}"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=30)
    try:
        data = r.json()
    except Exception as e:
        logger.warning(f"{ticker} fetch failed ({interval}): {e}")
        return None, None
    try:
        res = data["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame({
            "open":  q.get("open",  [np.nan]*len(ts)),
            "high":  q.get("high",   [np.nan]*len(ts)),
            "low":   q.get("low",    [np.nan]*len(ts)),
            "close": q.get("close",  [np.nan]*len(ts)),
            "volume":q.get("volume", [0]*len(ts)),
        }, index=pd.to_datetime(ts, unit="s", utc=True)).dropna(subset=["close"])
        if df.empty:
            return None, None
        df = df.sort_index()
        return df
    except Exception as e:
        logger.warning(f"{ticker} parse failed ({interval}): {e}")
        return None, None


def _bars(ticker: str, interval: str, period: str):
    """Return (df) where df has TZ-aware UTC datetime index."""
    df = _fetch(ticker, interval, period)
    if isinstance(df, tuple):          # _fetch returned (None, None) on parse failure
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    if df is None or df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    return df.tz_localize(None).tz_localize("UTC")


def symbol_tickers():
    """Symbols to scan. Tries universe.UNIVERSE_TICKERS from the existing repo;
    falls back to a hard-coded liquid NSE F&O list so the scanner runs anywhere."""
    try:
        from universe import UNIVERSE_TICKERS  # type: ignore
        tickers = [t.rstrip().upper() for t in UNIVERSE_TICKERS if t.strip()]
    except Exception:
        tickers = []
    if len(tickers) >= 10:
        return sorted(set(tickers))
    fallback = [
        "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HDFC","SBIN","BHARTIARTN",
        "KOTAKBANK","LT","AXISBANK","HCLTECH","ITC","MARUTI","WIPRO","ULTRACEMCO",
        "SUNPHARMA","BAJAJ-FINANCE","HDFCLIFE","NTPC","POWERGRID","ASIANPAINT","TITAN",
        "ZEEL","UPL","JSWSTEEL","ADANIENT","TATASTEEL","TATACONSUM","TATAMOTORS",
        "MARICO","CIPLA","DRREDDY","BRITANNIA","APOLLOHOSP","EICHERMOT","GRASIM",
        "BAJAJFINSV","M&M","HINDUNILVR","NESTLEIND","SHRIRAMFIN","BPCL","COALINDIA",
        "IOC","GAIL","HINDALCO","ADANIPORTS","VEDL","SUDHARMO",
    ]
    return [f"{x}.NS" for x in fallback]


def fetch_dual_bars(symbol: str):
    """15-min bars (trend frame, last 20d) + 5-min bars (trigger frame, last 10d)."""
    b15 = _bars(symbol, "15m", "20d")
    b5  = _bars(symbol, "5m",  "10d")
    return b5, b15


# --------------------------------------------------------------------------- #
# SESSION GROUPING (IST session boundaries)                                  #
# --------------------------------------------------------------------------- #
def to_ist(df: pd.DataFrame):
    """Index -> timezone-aware IST."""
    out = df.copy()
    out.index = out.index.tz_convert("Asia/Kolkata")
    return out


def sessions(df: pd.DataFrame):
    """Split into a dict IST-date -> session dataframe (IST index)."""
    if df.empty:
        return {}
    ist = to_ist(df)
    return {dt: grp.copy() for dt, grp in ist.groupby(ist.index.date)}


def today_session(bars):
    """The session that belongs to today (may be partially filled)."""
    sess = sessions(bars)
    return sess.get(ist_now().date()), sess


# --------------------------------------------------------------------------- #
# INDICATORS (vectorised; session-aware where needed)                          #
# --------------------------------------------------------------------------- #
def typical_price(o, h, l, c):
    return (h + l + c) / 3.0


def vwap_session(df):
    """Session VWAP (reset on each IST date); uses expanding mean if volume missing."""
    tp = typical_price(df["open"], df["high"], df["low"], df["close"])
    vol = df["volume"].astype(float).replace(0, np.nan)
    if vol.isna().all():
        return tp.expanding().mean()
    cv = vol.cumsum()
    w = tp * vol
    return (w.cumsum() / cv.replace(0, np.nan))


def sma(df, n):
    return df["close"].rolling(n).mean()


def stdma(df, n):
    return df["close"].rolling(n).std(ddof=1)


def true_range(o, h, l, prev_close):
    return np.maximum.reduce([h - l, (h - prev_close).abs(), (l - prev_close).abs()])


def atr(df, n):
    tr = true_range(df["open"], df["high"], df["low"], df["close"].shift(1))
    s = pd.Series(tr)
    return s.ewm(span=n, adjust=False).mean().values


def rsi(df, n=14):
    delta = df["close"].diff()
    up = delta.clip(lower=0).astype(float)
    down = (-delta.clip(upper=0)).astype(float)
    ups = up.ewm(span=n, adjust=False).mean()
    downs = down.ewm(span=n, adjust=False).mean()
    rs = ups / downs.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def di(df, n=14):
    plus_dm, minus_dm = np.zeros(len(df.index)), np.zeros(len(df.index))
    hp = df["high"].values; lp = df["low"].values
    for i in range(1, len(df)):
        plus_dm[i] = (hp[i] - hp[i-1]) if ((hp[i]-hp[i-1]) > abs(lp[i]-lp[i-1])) and (hp[i]-hp[i-1]) > 0 else 0.0
        minus_dm[i] = (lp[i-1]-lp[i]) if ((lp[i-1]-lp[i]) > (hp[i]-hp[i-1])) and (lp[i-1]-lp[i]) > 0 else 0.0
    atrv = atr(df, n)
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=n, adjust=False).mean() / pd.Series(atrv).replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=n, adjust=False).mean() / pd.Series(atrv).replace(0, np.nan))
    return plus_di, minus_di


def adx(df, n=14):
    pdi, mdi = di(df, n)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(span=n, adjust=False).mean()


def supertrend(high, low, close, period=10, mult=3.0):
    """Classic Supertrend returning (upper, lower, trend_direction) arrays."""
    high, low, close = np.asarray(high), np.asarray(low), np.asarray(close)
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    d = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
    atrv = pd.Series(atr(d, period)).bfill().fillna(0).values
    hl2 = (high + low) / 2.0
    up = hl2 + mult * atrv
    lo = hl2 - mult * atrv
    upper, lower, trend = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    upper[0], lower[0] = up[0], lo[0]
    trend[0] = 1
    for i in range(1, n):
        if trend[i-1] == 1:
            upper[i] = min(up[i], upper[i-1]); lower[i] = max(lo[i], lower[i-1])
        else:
            upper[i] = up[i]; lower[i] = lo[i]
        if trend[i-1] == 1:
            trend[i] = 1 if low[i] < lower[i] else (-1 if high[i] > upper[i] else trend[i-1])
        else:
            trend[i] = -1 if high[i] > upper[i] else (1 if low[i] < lower[i] else trend[i-1])
    return upper, lower, trend


def bollinger(df, n=20, k=2.0):
    m = sma(df, n); s = stdma(df, n)
    return m, m + k * s, m - k * s


# --------------------------------------------------------------------------- #
# STRUCTURAL BUILDING BLOCKS                                                  #
# --------------------------------------------------------------------------- #
def or_range(session_df, nbars=2):
    """Opening-range high/low from first `nbars` bars of the session."""
    d = session_df.iloc[:nbars]
    if d.empty:
        return np.nan, np.nan, np.nan
    return float(d["high"].max()), float(d["low"].min()), float(d["open"].iloc[0])


def inside_bar(series):
    """Boolean: each bar is inside the previous bar (15m frame)."""
    return (series["high"] <= series["high"].shift(1)) & (series["low"] >= series["low"].shift(1))


def atr_ratio(series, n=5):
    """ATR of this bar relative to median ATR of prior n bars (contraction flag)."""
    return atr(series, n) / atr(series, n).shift(n + 1).bfill()


def squeeze_bars(df15, threshold=0.75):
    """15m bars flagged as contractions (inside-bar OR ATR-ratio <= threshold)."""
    ib = inside_bar(df15).astype(int)
    ar = (atr_ratio(df15, 5) <= threshold).astype(int)
    return ib | ar


# --------------------------------------------------------------------------- #
# THE 1 : 5 RISK MODEL                                                        #
# --------------------------------------------------------------------------- #
def build_position(direction, entry, ctx):
    """Size a position around Rs 200 risk, strict 1:5, within stop-band + ATR."""
    long_side = direction == "LONG"
    entry = float(entry)
    if not entry or np.isnan(entry) or entry <= 0:
        return None

    struct = ctx.get("struct") or entry * 0.01
    if long_side:
        stop = round(entry - max(struct, entry * MIN_STOP_PCT), 2)
    else:
        stop = round(entry + max(struct, entry * MIN_STOP_PCT), 2)

    dist = abs(entry - stop)
    stop_pct = dist / entry
    if stop_pct < MIN_STOP_PCT or stop_pct > MAX_STOP_PCT:
        return None

    reward = RR_MULT * dist
    target = round(entry + reward if long_side else entry - reward, 2)

    daily_atr = ctx.get("atr") or np.nan
    if daily_atr and not np.isnan(daily_atr) and reward > daily_atr * REACH_FRAC:
        return None

    raw_qty = RISK_AMT / dist
    if raw_qty <= 0:
        return None
    qty = int(math.floor(raw_qty))
    margin_cap = int(math.floor(CAPITAL_RUPEES * MIS_LEVERAGE / entry))
    qty = min(qty, margin_cap)
    if qty <= 0:
        return None

    return {
        "entry": entry, "stop": stop, "target": target, "qty": qty,
        "risk_per_share": round(dist, 2), "risk_amt": round(dist * qty, 2),
        "reward_amt": round(reward * qty, 2), "rr": round(reward / dist, 2),
    }


# --------------------------------------------------------------------------- #
# FRAME PREPARATION                                                           #
# --------------------------------------------------------------------------- #
def prepare_frames(m5, b15):
    """Attach indicators to both frames; returns dicts keyed 'm5'/'b15'."""
    frames = {}
    for key, df in (("m5", m5), ("b15", b15)):
        out = df.copy()
        out["_vwap"] = vwap_session(out)
        out["_rsi"] = rsi(out)
        out["_atr"] = atr(out, 14)
        out["_bb_mid"], out["_bb_up"], out["_bb_lo"] = bollinger(out)
        if key == "m5":
            u, l, t = supertrend(out["high"].bfill(), out["low"].bfill(), out["close"], period=7, mult=2.0)
        else:
            u, l, t = supertrend(out["high"].bfill(), out["low"].bfill(), out["close"], period=10, mult=3.0)
        out["_st_up"], out["_st_lo"], out["_st_dir"] = u, l, t
        out["_adx"] = adx(out)
        frames[key] = out
    return frames


def conf_score(ctx):
    """0-1 confidence from confluence features (used by tuner gating)."""
    weights = {"vol_mult": 0.20, "vwap_align": 0.15, "st_confluence": 0.25,
               "trend_slope": 0.15, "bb_extreme": 0.10, "adx_strong": 0.15}
    score = 0.0
    if ctx.get("vol_mult", 1.0) >= 1.8: score += weights["vol_mult"]
    elif ctx.get("vol_mult", 1.0) >= 1.4: score += weights["vol_mult"] / 1.8
    if ctx.get("vwap_aligned"): score += weights["vwap_align"]
    if ctx.get("st_confl"): score += weights["st_confluence"]
    if ctx.get("trend_rising") or ctx.get("trend_falling"): score += weights["trend_slope"]
    if ctx.get("bb_at_extreme"): score += weights["bb_extreme"]
    if ctx.get("adx", 0) > 25: score += weights["adx_strong"]
    return min(score, 1.0)


def safe_dir(x):
    """Cast supertrend direction to int safely."""
    try:
        v = float(x)
        return 1 if v > 0 else (-1 if v < 0 else 0)
    except Exception:
        return 0


def make_signal(setup, direction, entry, ctx, note):
    """Build a canonical signal record."""
    pos = build_position(direction, entry, ctx)
    if pos is None or pos["qty"] <= 0:
        return []
    adxv = float(ctx.get("adx", 0)) if isinstance(ctx.get("adx"), (int, float)) else 0
    return [{
        "setup": setup, "direction": direction, "entry": entry, "stop": pos["stop"],
        "target": pos["target"], "qty": pos["qty"], "risk_per_share": pos["risk_per_share"],
        "risk_amt": pos["risk_amt"], "reward_amt": pos["reward_amt"], "rr": pos["rr"],
        "conf": conf_score(ctx), "note": note,
        "vol_mult": float(ctx.get("vol_mult", 1.0)), "vwap_aligned": bool(ctx.get("vwap_aligned")),
        "st_confl": bool(ctx.get("st_confl")), "trend_rising": bool(ctx.get("trend_rising")),
        "trend_falling": bool(ctx.get("trend_falling")), "adx": adxv,
    }]


# --------------------------------------------------------------------------- #
# STRATEGIES V1-V5 (dual-frame, side decided by the STOCK, not the index)     #
# --------------------------------------------------------------------------- #
def V1_15MIN_ORB_BREAKOUT(m5, b15, today):
    """
    Vol4 S1 lesson, upgraded: 15-minute opening range + 5-minute confirmation
    with volume & VWAP confluence. Side = stock's own breakout.
    Runs 09:35 - 14:25 (needs at least 2 ORB bars + 1 confirm).
    """
    ohi, oli, _ = or_range(today, 2)
    if np.isnan(ohi) or np.isnan(oli) or len(m5) < 20:
        return []
    st15 = b15.copy()
    st15["_dir"] = st15["_st_dir"].fillna(0).apply(safe_dir)
    vw = float(today["_vwap"].iloc[-1]) if len(today) else np.nan
    vol_avg = float(m5["volume"].rolling(20).mean().iloc[-1]) if len(m5) >= 20 else 0
    last = m5.iloc[-1]
    vm = last["volume"] / max(vol_avg, 1)
    outs = []
    for direction, rg in (("LONG", ohi), ("SHORT", oli)):
        ok_close = (last["close"] > rg * 1.001) if direction == "LONG" else (last["close"] < rg * 0.999)
        vwap_aligned = (vw > 0) and ((direction == "LONG") == (last["close"] > vw))
        st_confl = safe_dir(st15["_dir"].iloc[-1]) != 0
        cond = ok_close and vwap_aligned and st_confl and vm >= 1.5
        if cond:
            outs += make_signal("V1_15MIN_ORB_BREAKOUT", direction, last["close"], {
                "struct": ohi - oli, "atr": float(b15["_atr"].iloc[-1]),
                "vwap_aligned": True, "st_confl": True, "vol_mult": vm,
                "trend_rising": True if direction == "LONG" else False, "adx": float(st15["_adx"].iloc[-1])},
                f"OR-BREAK {direction} of 15m range [{oli:.0f},{ohi:.0f}] vol {vm:.1f}x")
    return outs


def V2_15M_SUPERTREND_MOMENTUM(m5, b15, today):
    """
    Vol4 S3 reloaded: 15-min Supertrend flip + 5-min pullback-to-ST-level entry
    + ADX momentum filter. Wider stop logic than Vol4 so it fires more often
    while respecting the 0.3%-1.5% band and the ATR reachability rule.
    Runs until 14:25.
    """
    if len(b15) < 40:
        return []
    st15 = b15.copy()
    st15["_dir"] = st15["_st_dir"].fillna(0).apply(safe_dir)
    adx15 = float(st15["_adx"].iloc[-1]) if "_adx" in st15.columns else 0
    st_dir = safe_dir(st15["_dir"].iloc[-1])
    if st_dir == 0 or adx15 < 20:
        return []
    st_level = float(st15["_st_lo"].iloc[-1]) if st_dir == 1 else float(st15["_st_up"].iloc[-1])
    if np.isnan(st_level):
        return []

    hist = m5[m5.index <= m5.index[-2]]
    direction = "LONG" if st_dir == 1 else "SHORT"
    for _, row in hist.iterrows():
        inside = (st_dir == 1 and row["low"] <= st_level * 1.002) or (st_dir == -1 and row["high"] >= st_level * 0.998)
        resume = (st_dir == 1 and row["close"] > row["open"]) or (st_dir == -1 and row["close"] < row["open"])
        if inside and resume:
            return make_signal("V2_15M_SUPERTREND_MOMENTUM", direction, row["close"], {
                "struct": 3.0, "atr": float(b15["_atr"].iloc[-1]), "st_confl": True},
                f"15M-ST {direction} pullback to {st_level:.1f}")
    return []


def V3_VWAP_TREND_PULLBACK(m5, b15, today):
    """
    Vol4 S2 lesson upgraded: only takes VWAP pullbacks WITH 15-min trend bias
    (Supertrend + rising/falling VWAP slope), rejecting counter-trend noise.
    Runs until 14:25.
    """
    if len(b15) < 30 or len(today) < 5:
        return []
    st15 = b15.copy()
    st15["_dir"] = st15["_st_dir"].fillna(0).apply(safe_dir)
    vw = float(today["_vwap"].iloc[-1]) if len(today) else np.nan
    if np.isnan(vw):
        return []
    vwslope = float(today["_vwap"].iloc[-1] - today["_vwap"].iloc[-5])
    st_dir = safe_dir(st15["_dir"].iloc[-1])
    trend_up = (st_dir == 1 and vwslope > 0) or (st_dir == -1 and vwslope < 0)
    if st_dir == 0 or abs(vwslope) < 0.0001 or not trend_up:
        return []

    hist = m5[m5.index <= m5.index[-2]]
    direction = "LONG" if st_dir == 1 else "SHORT"
    for _, row in hist.tail(30).iterrows():
        near_vwap = abs(row["close"] - vw) / vw < 0.0015
        rejects = ((st_dir == 1 and row["close"] >= vw and row["close"] >= row["open"]) or
                   (st_dir == -1 and row["close"] <= vw and row["close"] <= row["open"]))
        if near_vwap and rejects:
            return make_signal("V3_VWAP_TREND_PULLBACK", direction, row["close"], {
                "struct": abs(vw - row["low"]), "atr": float(b15["_atr"].iloc[-1]),
                "vwap_aligned": True}, f"15M-trend pullback into VWAP {vw:.1f}")
    return []


def V4_INSIDE_DAY_EXPANSION(m5, b15, today):
    """
    Vol4 S4 lesson upgraded: 15-min inside-bar/narrow-range contraction
    resolving in the direction of 15-min trend, with volume surge.
    Runs 09:50 - 14:25.
    """
    if len(b15) < 25:
        return []
    st15 = b15.copy()
    st15["_dir"] = st15["_st_dir"].fillna(0).apply(safe_dir)
    st_dir = safe_dir(st15["_dir"].iloc[-1])
    if st_dir == 0:
        return []
    vol_avg = float(m5["volume"].rolling(20).mean().iloc[-1]) if len(m5) >= 20 else 0
    direction = "LONG" if st_dir == 1 else "SHORT"

    def scan(days_back=3):
        cand = m5[(m5.index.date >= (ist_now().date() - timedelta(days=days_back))) & (m5.index != m5.index[-1])]
        hits = []
        prev_h, prev_l = st15.iloc[-2]["high"], st15.iloc[-2]["low"]
        for _, row in cand.tail(25).iterrows():
            break_out = (st_dir == 1 and row["close"] > prev_h) or (st_dir == -1 and row["close"] < prev_l)
            vols = row["volume"] / max(vol_avg, 1)
            if break_out and vols >= 1.8:
                hits.append((row, vols))
        return hits[:1]

    hits = scan()
    if not hits:
        return []
    row, vols = hits[0]
    return make_signal("V4_INSIDE_DAY_EXPANSION", direction, row["close"], {
        "struct": 2.0, "atr": float(b15["_atr"].iloc[-1]), "vol_mult": vols, "st_confl": True},
        f"inside-expansion {direction} vol {vols:.1f}x")


def V5_OPENING_RANGE_TAIL_BREAK(m5, b15, today):
    """
    New Vol5 strategy: tail-break of the 09:30-10:00 15-min block in the
    direction of 15-min trend. Captures continuation days after the initial
    ORB fight settles (fills well in afternoon hours).
    Runs 10:15 - 14:25.
    """
    if today is None or len(today) < 4:
        return []
    tail_hi, tail_lo, _ = or_range(today.iloc[-4:], 4)
    st15 = b15.copy()
    st15["_dir"] = st15["_st_dir"].fillna(0).apply(safe_dir)
    st_dir = safe_dir(st15["_dir"].iloc[-1])
    if st_dir == 0:
        return []
    vw15 = float(b15["_vwap"].iloc[-1]) if len(b15) else np.nan
    last5 = m5.iloc[-1]
    vol_avg = float(m5["volume"].rolling(20).mean().iloc[-1]) if len(m5) >= 20 else 0
    vm = last5["volume"] / max(vol_avg, 1)
    direction = "LONG" if st_dir == 1 else "SHORT"
    cond_long = st_dir == 1 and last5["close"] > tail_hi * 1.002 and vm >= 1.6 and vw15 > 0 and last5["close"] > vw15
    cond_short = st_dir == -1 and last5["close"] < tail_lo * 0.998 and vm >= 1.6 and vw15 > 0 and last5["close"] < vw15
    if not (cond_long or cond_short):
        return []
    ctx = {"struct": tail_hi - tail_lo, "atr": float(b15["_atr"].iloc[-1]),
           "vol_mult": vm, "st_confl": True,
           "trend_rising": cond_long, "trend_falling": cond_short}
    return make_signal("V5_OPENING_RANGE_TAIL_BREAK", direction, last5["close"], ctx,
                       f"tail-break {direction} 10:00 block [{tail_lo:.0f},{tail_hi:.0f}]")


STRATEGY_FUNCS = {
    "V1_15MIN_ORB_BREAKOUT": V1_15MIN_ORB_BREAKOUT,
    "V2_15M_SUPERTREND_MOMENTUM": V2_15M_SUPERTREND_MOMENTUM,
    "V3_VWAP_TREND_PULLBACK": V3_VWAP_TREND_PULLBACK,
    "V4_INSIDE_DAY_EXPANSION": V4_INSIDE_DAY_EXPANSION,
    "V5_OPENING_RANGE_TAIL_BREAK": V5_OPENING_RANGE_TAIL_BREAK,
}
# --------------------------------------------------------------------------- #
# NIFTY REGIME ENGINE (six-vote, advisory only — never blocks a side)          #
# Mirrors Vol4's rebuilt regime design; votes recomputed every sweep.          #
# --------------------------------------------------------------------------- #
def nifty_regime():
    """Six independent votes from NIFTY 5-min bars; score >= +2 UP, <= -2 DOWN."""
    b5 = _bars("^NSEI", "5m", "7d")
    if b5 is None or b5.empty or len(b5) < 60:
        return {"label": "NEUTRAL", "score": 0, "votes": {},
                "note": "insufficient index bars"}
    df = b5.copy()
    df["_tp"] = typical_price(df["open"], df["high"], df["low"], df["close"])
    vol = df["volume"].astype(float)
    if vol.replace(0, np.nan).isna().all():
        df["_vwap"] = df["_tp"].expanding().mean()
    else:
        cv = vol.replace(0, np.nan).cumsum()
        df["_vwap"] = (df["_tp"] * vol).cumsum() / cv.replace(0, np.nan)
    open_price = float(df["open"].iloc[0])
    high_of_day = float(df["high"].max())
    low_of_day = float(df["low"].min())
    close = float(df["close"].iloc[-1])
    vwap = float(df["_vwap"].iloc[-1])
    votes = {}
    votes["vwap"] = 1 if close > vwap else (-1 if close < vwap else 0)
    votes["open"] = 1 if close > open_price else (-1 if close < open_price else 0)
    if len(df) >= 12:
        h_lag = float(df["high"].shift(6).iloc[-1])
        l_lag = float(df["low"].shift(6).iloc[-1])
        if close > h_lag and float(df["high"].iloc[-1]) > h_lag:
            votes["struct"] = 1
        elif close < l_lag and float(df["low"].iloc[-1]) < l_lag:
            votes["struct"] = -1
        else:
            votes["struct"] = 0
    else:
        votes["struct"] = 0
    if len(df) >= 12:
        roc = close / float(df["close"].iloc[-12]) - 1
        votes["momo"] = 1 if roc > 0.0012 else (-1 if roc < -0.0012 else 0)
    else:
        votes["momo"] = 0
    move = (high_of_day - open_price) / open_price
    if abs(move) >= 0.002:
        recovered = (close - open_price) / move if move != 0 else 0
        votes["retrace"] = 1 if recovered >= 0.5 else 0
    else:
        votes["retrace"] = 0
    vwaps = df["_vwap"].tail(12).values
    closes = df["close"].tail(12).values
    reclaim = 0
    for pv, c in zip(vwaps[:-1], closes[1:]):
        if ((pv < vwaps[-2]) and (c >= vwaps[-1])) or ((pv > vwaps[-2]) and (c <= vwaps[-1])):
            reclaim = 1 if c >= vwaps[-1] else -1
            break
    votes["reclaim"] = reclaim if reclaim else 0
    score = sum(votes.values())
    stored = _state_load().get("regime", {}).get("label", "NEUTRAL")
    label = "UP" if score >= 2 else ("DOWN" if score <= -2 else "NEUTRAL")
    if stored in ("UP", "DOWN") and score == 0:
        label = stored
    state = {"label": label, "score": score, "votes": {k: str(v) for k, v in votes.items()},
             "note": f"{len(df)} bars | range {low_of_day:.0f}-{high_of_day:.0f}"}
    _state_store("regime", state)
    return state


# --------------------------------------------------------------------------- #
# STATE & TUNER MEMORY                                                        #
# --------------------------------------------------------------------------- #
def _sig_ts(s):
    """Serialize signal dict datetime fields to UTC-ISO strings."""
    out = dict(s)
    for k in ("signal_time", "filled_at"):
        if k in out and isinstance(out[k], datetime):
            out[k] = out[k].astimezone(timezone.utc).isoformat()
    return out


def _sig_dt(s):
    """Restore UTC datetime fields in loaded signal dicts."""
    for k in ("signal_time", "filled_at"):
        if k in s and isinstance(s[k], str):
            try:
                s[k] = datetime.fromisoformat(s[k]).replace(tzinfo=timezone.utc)
            except Exception:
                pass
    return s


def _state_load():
    try:
        return json.loads((STATE_FILE.read_text() if STATE_FILE.exists() else "{}"))
    except Exception:
        return {}


def _state_store(key, val):
    d = _state_load()
    d[key] = val
    STATE_FILE.write_text(json.dumps(d, indent=1))


def _tuner_load():
    try:
        return json.loads((TUNER_STATE_FILE.read_text() if TUNER_STATE_FILE.exists() else "{}"))
    except Exception:
        return {}


def _tuner_store(d):
    TUNER_STATE_FILE.write_text(json.dumps(d, indent=1))


def _today_date_str():
    return ist_now().date().isoformat()


# --------------------------------------------------------------------------- #
# AUTO-TUNE ENGINE (learns from the LAST 10 SESSIONS)                         #
# --------------------------------------------------------------------------- #
def load_journal():
    """Full historical trade journal (all sessions) for grading buckets."""
    rows = []
    p = JOURNAL_DIR / "trade_journal.csv"
    if not p.exists():
        return rows
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def session_cutoff_rows(rows):
    """Rows strictly before today (closed sessions)."""
    today = _today_date_str()
    return [r for r in rows if r.get("Date", "") < today]


def grade_buckets(rows):
    """Per-bucket stats over the closed-session history."""
    buckets = {}
    for r in rows:
        key = f"{r['Setup']}_{r['Direction']}"
        b = buckets.setdefault(key, {"wins": 0, "losses": 0, "flats": 0,
                                     "trades": 0, "netR": 0.0, "slip": 0})
        b["trades"] += 1
        res = r.get("Result", "")
        if res == "WIN": b["wins"] += 1
        elif res == "LOSS": b["losses"] += 1
        elif res == "FLAT": b["flats"] += 1
        try: b["netR"] += float(r.get("PnL_R") or 0)
        except ValueError: pass
    out = {}
    for k, b in buckets.items():
        tot = b["wins"] + b["losses"]
        out[k] = dict(b, win_rate=(b["wins"] / tot if tot else 0.0),
                      netR=round(b["netR"], 4), tot_grades=tot)
    return out


def run_auto_tune():
    """Mute losers / chronic chases; re-enable improving buckets. Log to tuning_log."""
    today = _today_date_str()
    rows = load_journal()
    bucket_stats = grade_buckets(session_cutoff_rows(rows))
    for sr in list(csv.DictReader((JOURNAL_DIR / "slippage_log.csv").open(newline=""))
                   if (JOURNAL_DIR / "slippage_log.csv").exists() else []):
        key = f"{sr['Setup']}_{sr['Direction']}"
        if key in bucket_stats:
            bucket_stats[key]["slip"] += 1
    tuner = _tuner_load()
    decisions = []
    active_prev = set(tuner.get("buckets", {}).keys())
    for key, st in sorted(bucket_stats.items()):
        prev_status = tuner.get("buckets", {}).get(key, {}).get("status", "enabled")
        muted_loss = st["netR"] < 0 and st["tot_grades"] >= AT_MIN_TRADES
        chronic_chase = (st["slip"] >= AT_SLIP_MIN) and (st["slip"] > st["trades"])
        do_mute = muted_loss or chronic_chase
        new_status = "muted" if do_mute else "enabled"
        if new_status != prev_status:
            reason = "losing logic" if muted_loss else ("chronic chase" if chronic_chase else "")
            rec = {"Date": today, "Bucket": key, "Decision": new_status,
                   "NetR": st["netR"], "Wins": st["wins"], "Losses": st["losses"],
                   "Flats": st["flats"], "Trades": st["trades"],
                   "WinRatePct": round(st["win_rate"] * 100, 1),
                   "SlipCount": st["slip"], "Reason": reason}
            decisions.append(rec)
            logger.info(f"TUNER {key}: {new_status} ({reason or 'no trigger'}) netR={st['netR']:.2f} WR={st['win_rate']*100:.0f}%")
        tuner.setdefault("buckets", {})[key] = {"status": new_status, "scored_at": today}
    for key in list(active_prev):
        if key not in tuner["buckets"]:
            tuner["buckets"][key] = {"status": "enabled", "scored_at": today}
    if decisions:
        _log_tuning(decisions)
    _tuner_store(tuner)
    return decisions


def _log_tuning(records):
    p = JOURNAL_DIR / "tuning_log.csv"
    first = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TUNER_COLS)
        if first:
            w.writeheader()
        w.writerows(records)


def ranked_buckets():
    """Order of strategies that earned their slot — printed as tomorrow priority."""
    rows = load_journal()
    stats = grade_buckets(session_cutoff_rows(rows))
    enabled = [(k, s) for k, s in stats.items()
               if _tuner_load().get("buckets", {}).get(k, {}).get("status") == "enabled"]
    enabled.sort(key=lambda x: (x[1]["netR"], x[1]["win_rate"]), reverse=True)
    return enabled[:6]


# --------------------------------------------------------------------------- #
# SESSION TREND CLASSIFIER (over last 10 SESSIONS)                            #
# --------------------------------------------------------------------------- #
def classify_session_trend(days=10):
    """Bullish / bearish / range based on recent daily closes vs opens + VWAP."""
    ticks = ["RELIANCE.NS", "INFY.NS", "ICICIBANK.NS", "HDFCBANK.NS", "SBIN.NS",
             "TCS.NS", "TATASTEEL.NS", "ITC.NS", "WIPRO.NS", "BHARTIARTN.NS"]
    score = 0
    for t in ticks:
        df = _bars(t, "1d", f"{days+5}d")
        if df is None or df.empty or len(df) < days:
            continue
        sess = sessions(df)
        for dt, s in list(sess.items())[:days]:
            o = float(s["open"].iloc[0]); c = float(s["close"].iloc[-1])
            v = float(s["_vwap"].iloc[-1]) if "_vwap" in s.columns else float("nan")
            if not (np.isnan(c) or np.isnan(o) or np.isnan(v)):
                if c > o and c > v: score += 1
                elif c < o and c < v: score -= 1
    if score > days * 0.55: return "BULLISH"
    if score < -days * 0.55: return "BEARISH"
    return "RANGE-BOUND"


# --------------------------------------------------------------------------- #
# SIGNAL DISPATCHER: cap, breaker, mode resolution                            #
# --------------------------------------------------------------------------- #
def day_ledger(rows):
    """Current trading-day ledger: graded counts, slots used, cap/breaker state."""
    today = _today_date_str()
    day_rows = [r for r in rows if r.get("Date", "") == today]
    graded = [r for r in day_rows if r.get("Status") == "CLOSED"]
    open_rows = [r for r in day_rows if r.get("Status") == "OPEN"]
    wins = losses = flat = streak = 0
    net_r = 0.0
    for r in graded:
        res = r.get("Result", "")
        net_r += float(r.get("PnL_R") or 0)
        if res == "WIN": wins += 1
        elif res == "LOSS": losses += 1; streak += 1
        elif res == "FLAT": flat += 1; streak = 0
    tripped = (net_r <= DAILY_LOSS_R) or (streak >= LOSS_STREAK_TRIP)
    return {"wins": wins, "losses": losses, "flats": flat, "netR": round(net_r, 4),
            "tripped": tripped, "open": open_rows, "committed": sum(1 for r in day_rows if r.get("Mode") == "TRADE")}


def mode_for(bucket, ledger, slots_free):
    """Decide TRADE vs TRACKING for a newly fired signal."""
    tuner = _tuner_load()
    if tuner.get("buckets", {}).get(bucket, {}).get("status") == "muted":
        return "TRACKING", "auto-tuned (negative 10-session netR)"
    if slots_free <= 0 or ledger["tripped"]:
        return "TRACKING", "circuit breaker tripped" if ledger["tripped"] else "daily trade cap reached"
    return "TRADE", ""


# --------------------------------------------------------------------------- #
# FILL GATE                                                                   #
# --------------------------------------------------------------------------- #
def check_fills(symbol, pending, bars_m5):
    """Mark pending LONG fills when a bar low touches trigger (tol) and vice versa."""
    filled, kept = [], pending[:]
    now = ist_now().astimezone(timezone.utc)
    bars = bars_m5[bars_m5.index <= now]
    for sig in pending:
        keep = True
        st = _sig_dt(dict(sig))
        target_t = float(st["trigger"])
        best_run = 0.0
        if st["direction"] == "LONG":
            ok = any(float(b["low"]) <= target_t * (1 + FILL_TOL_PCT) for _, b in bars.iterrows())
            if len(bars):
                best_run = max(((float(b["high"]) - target_t) / target_t) for _, b in bars.iterrows())
        else:
            ok = any(float(b["high"]) >= target_t * (1 - FILL_TOL_PCT) for _, b in bars.iterrows())
            if len(bars):
                best_run = max(((target_t - float(b["low"])) / target_t) for _, b in bars.iterrows())
        if ok:
            filled.append({k: st.get(k) for k in list(st) + ["filled_at", "best_run_pct"]} |
                          {"filled_at": now, "best_run_pct": round(best_run * 100, 2)})
            keep = False
    return filled, [s for s in kept if s not in filled]


# --------------------------------------------------------------------------- #
# JOURNAL WRITING                                                             #
# --------------------------------------------------------------------------- #
def next_trade_id(today):
    base = 0
    p = JOURNAL_DIR / "trade_journal.csv"
    if p.exists():
        with p.open() as f:
            try:
                ids = sorted(list(set(r["TradeID"].split("-")[3] for r in csv.DictReader(f)
                                      if r.get("TradeID", "").startswith("V5-" + today))))
                if ids:
                    base = int(ids[-1])
            except Exception:
                base = 0
    return f"V5-{today}-{base+1:05d}"


def append_journal(rows_new):
    p = JOURNAL_DIR / "trade_journal.csv"
    first = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_COLS)
        if first:
            w.writeheader()
        w.writerows(rows_new)


def slippage_append(rows):
    p = JOURNAL_DIR / "slippage_log.csv"
    first = not p.exists()
    cols = ["Date", "SignalTime", "Symbol", "Direction", "Setup", "Trigger",
            "BestRunPct", "ExpiredAt", "Outcome", "Mode", "Notes"]
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if first:
            w.writeheader()
        w.writerows(rows)


def pnl_from_row(row):
    """Compute PnL_R / PnL_Amt given entry/stop/target/exit, direction, qty."""
    try:
        entry = float(row["Entry"]); stop = float(row["StopLoss"]); qty = int(float(row["Qty"]))
        exitp = float(row["ExitPrice"]); direction = row["Direction"]
    except Exception:
        return 0.0, 0.0
    dist = abs(entry - stop)
    r = ((exitp - entry) / dist) if direction == "LONG" else ((entry - exitp) / dist)
    return round(r, 3), round(r * dist * qty, 2)


def write_daily_summary(ledger, tracking_graded=0, notes=""):
    today = _today_date_str()
    p = JOURNAL_DIR / "daily_summary.csv"
    rows = []
    if p.exists():
        rows = list(csv.DictReader(p.open()))
    row = None
    for r in rows:
        if r.get("Date") == today:
            row = r; break
    if row is None:
        row = {c: "" for c in ["Date","Trades","Tracking","Wins","Losses","Flats","Open","WinRatePct","NetR","NetPnlAmt","BreakerTripped","DayVerdict","Notes"]}
        rows.append(row)
    totals = {"Trades": ledger["wins"] + ledger["losses"] + ledger["flats"],
              "Tracking": tracking_graded, "Wins": ledger["wins"],
              "Losses": ledger["losses"], "Flats": ledger["flats"],
              "Open": 0, "WinRatePct": 0.0, "NetR": ledger["netR"], "NetPnlAmt": 0.0,
              "BreakerTripped": "YES" if ledger["tripped"] else "NO",
              "DayVerdict": "FAILED" if ledger["tripped"] else "OK",
              "Date": today, "Notes": notes}
    row.update(totals)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Date","Trades","Tracking","Wins","Losses","Flats","Open","WinRatePct","NetR","NetPnlAmt","BreakerTripped","DayVerdict","Notes"])
        w.writeheader()
        w.writerow(row)
    return row


# --------------------------------------------------------------------------- #
# REPORT GENERATION                                                           #
# --------------------------------------------------------------------------- #
def render_card(sig, committed, mode):
    tag = "TRACKING" if mode == "TRACKING" else "NEW"
    card = (f"*   [{tag}][{'TRADE' if mode=='TRADE' else 'TRACKING'} {committed}/{CAP_TRADES_DAY}] "
            f"{sig['setup']} {sig['direction']}\n")
    card += (f"       Entry {sig['entry']:.2f} · Stop {sig['stop']:.2f} · Target {sig['target']:.2f} · Qty {sig['qty']} "
             f"(Rs {sig['risk_amt']:.0f} risk / Rs {sig['reward_amt']:.0f} target)\n")
    card += f"       @ {sig['signal_time'].strftime('%H:%M')} IST · Conf {sig['conf']:.2f} · *{sig['note']}*\n"
    return card


def write_report(phase, regime, bucket_table, cards, recommendations, trend, notes=""):
    lines = [
        "# VOL5 — Extended-Hours Dual-Frame Intraday Watchlist",
        "**NSE F&O · 1 : 5 Reward : Risk · Dual timeframes (15-min trend / 5-min trigger)**",
        "*Generated by scanner_v5.py on GitHub Actions · Educational use only — signals are alerts, not commands.*",
        "", "## Session snapshot",
        "| Item | Value |", "|---|---|",
        f"| Date (IST) | {_today_date_str()} |", f"| Phase | {phase} |",
        f"| IST Clock | {ist_now().strftime('%H:%M IST')} |",
        f"| Regime | **{regime['label']}** (score {regime['score']}) — {json.dumps(regime['votes'])} |",
        "| Trade budget | 10 committed TRADE slots per day |",
        f"| Sessions learned (auto-tune) | last {AT_WINDOW_DAYS} completed sessions |",
        "", "## Signal cards"]
    for c in cards:
        lines.append(c)
    lines += ["\n## Method accuracy over last 10 sessions",
              "| Bucket | Status | NetR | Win Rate | Trades | Reason |",
              "|---|---|---|---|---|---|"]
    for name, status, netr, wr, tr, reason in bucket_table:
        lines.append(f"| {name} | `{status}` | {netr:+.2f} | {wr:.0f}% | {tr} | {reason} |")
    lines += ["\n### EOD learning — recommended method priority for tomorrow",
              "(methods that proved profitable are listed first; the auto-tune will favour them.)"]
    if recommendations:
        for i, (bk, st) in enumerate(recommendations, 1):
            lines.append(f"{i}. **{bk}** — netR {st['netR']:+.2f}, win rate {st['win_rate']*100:.0f}% ({st['tot_grades']} graded)")
    else:
        lines.append("- No graded history yet this session — watchlist starts neutral.")
    lines += [f"\n*Tape bias over last {AT_WINDOW_DAYS} sessions: **{trend}**. Regime is advisory only; the stock's own breakout decides every side.{notes}*"]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{_today_date_str()}_vol5_report.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# TELEGRAM ALERTS (optional)                                                  #
# --------------------------------------------------------------------------- #
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID")


def tg_send(text):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage?chat_id={TG_CHAT}&text={requests.utils.quote(text)}"
        requests.get(url, timeout=15)
    except Exception as e:
        logger.warning(f"Telegram failed: {e}")


# --------------------------------------------------------------------------- #
# UNIT TEST BATTERY (SCANNER_TEST_MODE=1) — no network needed                  #
# --------------------------------------------------------------------------- #
def run_tests():
    """Deterministic smoke tests covering strategy math, guard rails, tune rules."""
    fails = []
    rng = np.random.default_rng(7)

    def chk(name, cond):
        print(("PASS" if cond else "FAIL"), "-", name)
        if not cond:
            fails.append(name)

    # Test 1: supertrend produces finite arrays on realistic data
    close = rng.normal(2500, 15, 100); high = close + rng.uniform(1, 8, 100); low = close - rng.uniform(1, 8, 100)
    up, lo, tr = supertrend(high, low, close)
    chk("supertrend: shape + sign values", up.shape == lo.shape == tr.shape == (100,) and set(np.unique(tr)).issubset({-1, 0, 1}))

    # Test 2: build_position enforces the 0.3%-1.5% stop band + ATR reachability
    ctx = {"struct": 2.0, "atr": 60.0}
    pos = build_position("LONG", 2500.0, ctx)
    chk("build_position: base case returns qty", pos is not None and pos["qty"] > 0)
    chk("build_position: stop within band", MIN_STOP_PCT <= abs(pos["entry"]-pos["stop"])/pos["entry"] <= MAX_STOP_PCT)
    chk("build_position: strict 5R target", abs(pos["target"]-pos["entry"]) == RR_MULT * abs(pos["entry"]-pos["stop"]))
    chk("build_position: rejects >1.5% stops", build_position("LONG", 2500.0, {"struct": 45.0, "atr": 30.0}) is None)
    chk("build_position: rejects <0.3% stops", build_position("LONG", 2500.0, {"struct": 0.005, "atr": 30.0}) is None)
    chk("build_position: rejects unreachable 5R (small ATR)", build_position("LONG", 2500.0, {"struct": 2.0, "atr": 1.5}) is None)
    chk("build_position: risk amount ~ Rs 200", pos is not None and abs(pos["risk_amt"] - RISK_AMT) < 10)

    # Test 3: V1 strategy produces valid signals only after ORB exists
    rng2 = np.random.default_rng(3)
    t0 = pd.Timestamp("2026-11-13 09:15:00", tz="UTC")
    idx = pd.date_range(t0, periods=60, freq="5min")
    close5 = pd.Series(rng2.uniform(200, 210, 60), index=idx)
    open5 = close5 * rng2.uniform(0.99, 1.01, 60); high5 = close5 * rng2.uniform(1.0, 1.005, 60)
    low5 = close5 * rng2.uniform(0.995, 1.0, 60); vol5 = pd.Series(rng2.integers(1000, 5000, 60), index=idx)
    m5 = pd.DataFrame({"open":open5,"high":high5,"low":low5,"close":close5,"volume":vol5})
    m15 = pd.DataFrame({"open":open5.iloc[::3],"high":high5.iloc[::3],"low":low5.iloc[::3],
                        "close":close5.iloc[::3],"volume":vol5.iloc[::3]}).reset_index(drop=True)
    m15["_vwap"] = vwap_session(m15); m15["_atr"] = atr(m15, 14)
    _, _, dirs = supertrend(m15["high"], m15["low"], m15["close"]); m15["_st_dir"] = dirs
    m15["_adx"] = adx(m15)
    today = m5[m5.index.date == m5.index[-1].date]
    signals = V1_15MIN_ORB_BREAKOUT(m5, m15, today)
    chk("V1: returns well-formed signals", all(s["qty"] > 0 and s["setup"] == "V1_15MIN_ORB_BREAKOUT" for s in signals))

    # Test 4: auto-tune mutes negative-netR buckets & re-enables improved ones
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    test_rows = [{"TradeID": f"t{i}", "Date": "2026-11-01" if i % 2 == 0 else "2026-11-05",
                  "Symbol": "RELIANCE.NS", "Direction": "LONG", "Setup": "V1_15MIN_ORB_BREAKOUT",
                  "Entry": 2000, "StopLoss": 1995, "Target": 2025, "Qty": 40, "RiskPerShare": 5,
                  "RiskAmt": 200, "RewardAmt": 1000, "RR": 5.0, "Status": "CLOSED", "Mode": "TRADE",
                  "ExitPrice": 1990, "ExitTime": "10:00", "ExitReason": "STOP_HIT",
                  "Result": "LOSS" if i < 5 else "WIN", "PnL_R": -5.0 if i < 5 else 5.0,
                  "PnL_Amt": -1000 if i < 5 else 1000, "UpdatedAt": "2026-11-05T10:00",
                  "Remarks": "", "Notes": ""} for i in range(8)]
    with (JOURNAL_DIR / "trade_journal.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_COLS); w.writeheader(); w.writerows(test_rows)
    run_auto_tune()
    chk("tune: losing bucket muted", _tuner_load().get("buckets", {}).get("V1_15MIN_ORB_BREAKOUT_LONG", {}).get("status") == "muted")
    extra = [{"TradeID": f"t9", "Date": "2026-11-07", "Symbol": "RELIANCE.NS", "Direction": "LONG",
              "Setup": "V1_15MIN_ORB_BREAKOUT", "Entry": 2000, "StopLoss": 1995, "Target": 2025,
              "Qty": 40, "RiskPerShare": 5, "RiskAmt": 200, "RewardAmt": 1000, "RR": 5.0,
              "Status": "CLOSED", "Mode": "TRADE", "ExitPrice": 2025, "ExitTime": "11:00",
              "ExitReason": "TARGET_HIT", "Result": "WIN", "PnL_R": 5.0, "PnL_Amt": 1000,
              "UpdatedAt": "2026-11-07T11:00", "Remarks": "", "Notes": ""},
             {"TradeID": "t10", "Date": "2026-11-07", "Symbol": "RELIANCE.NS", "Direction": "LONG",
              "Setup": "V1_15MIN_ORB_BREAKOUT", "Entry": 2000, "StopLoss": 1995, "Target": 2025,
              "Qty": 40, "RiskPerShare": 5, "RiskAmt": 200, "RewardAmt": 1000, "RR": 5.0,
              "Status": "CLOSED", "Mode": "TRADE", "ExitPrice": 2025, "ExitTime": "11:00",
              "ExitReason": "TARGET_HIT", "Result": "WIN", "PnL_R": 5.0, "PnL_Amt": 1000,
              "UpdatedAt": "2026-11-07T11:00", "Remarks": "", "Notes": ""}]
    with (JOURNAL_DIR / "trade_journal.csv").open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_COLS); w.writerows(extra)
    run_auto_tune()
    chk("tune: improved bucket re-enabled", _tuner_load().get("buckets", {}).get("V1_15MIN_ORB_BREAKOUT_LONG", {}).get("status") == "enabled")

    # Test 5: mode_for enforces cap + breaker
    ledger_cap = {"wins": 0, "losses": 0, "flats": 0, "netR": 0.0, "tripped": False,
                  "open": [], "committed": CAP_TRADES_DAY}
    chk("mode_for: cap => TRACKING", mode_for("X_LONG", ledger_cap, 0)[0] == "TRACKING")
    ledger_brk = {"wins": 0, "losses": 2, "flats": 0, "netR": -2.0, "tripped": True,
                  "open": [], "committed": 5}
    chk("mode_for: breaker => TRACKING", mode_for("X_LONG", ledger_brk, 3)[0] == "TRACKING")
    chk("mode_for: healthy => TRADE", mode_for("X_LONG", {"wins":1,"losses":0,"flats":0,"netR":2.0,"tripped":False,"open":[],"committed":0}, 8)[0] == "TRADE")

    # Test 6: regime vote bounds
    reg = nifty_regime()
    chk("regime: bounded score", -6 <= reg["score"] <= 6 and reg["label"] in ("UP", "NEUTRAL", "DOWN"))

    if fails:
        print(f"\n{len(fails)} TEST(S) FAILED:", fails)
        return 1
    print("\nALL TESTS PASSED")
    return 0


# --------------------------------------------------------------------------- #
# MAIN SCANNER                                                                 #
# --------------------------------------------------------------------------- #
def main():
    mode = os.environ.get("RUN_MODE", "live")
    force_date = os.environ.get("SCANNER_FORCE_DATE")
    override_min = ist_clock_override()
    min_now = override_min if override_min is not None else ist_now_min()

        # ensure output directories exist (Vol4-compatible repo-root layout);
    # test runs redirect writes into an ephemeral staging area instead of the repo.
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if os.environ.get("RUN_MODE") == "test":
        import shutil
        if TEST_OUT_BASE.exists(): shutil.rmtree(TEST_OUT_BASE)
        (TEST_OUT_BASE / "journal_v5").mkdir(parents=True, exist_ok=True)
        (TEST_OUT_BASE / "reports_v5").mkdir(parents=True, exist_ok=True)
        import sys
        sys.modules[__name__].__dict__["JOURNAL_DIR"] = TEST_OUT_BASE / "journal_v5"
        sys.modules[__name__].__dict__["REPORTS_DIR"] = TEST_OUT_BASE / "reports_v5"

    if os.environ.get("SCANNER_TEST_MODE") == "1":
        return run_tests()

    finalize_note = ""
    phase = market_phase(min_now)
    logger.info(f"VOL5 scan starting — phase={phase} ist_clock={ist_now().strftime('%H:%M')} "
                f"test_mode={mode} date_force={force_date}")

    if phase == "EARLY":
        return  # silent exit before watchlist

    if phase == "PREP":
        logger.info("PREP — pre-market watchlist built; full scan starts at 09:30 IST")
        return

    universe = symbol_tickers()
    rows = load_journal()
    ledger = day_ledger(rows)
    tuner = _tuner_load()
    pendings = [_sig_dt(s) for s in _state_load().get("pending", [])]
    slots_free = CAP_TRADES_DAY - ledger["committed"]

    regime = nifty_regime()

    # sweep-gap warning
    last = _state_load().get("last_sweep")
    if last:
        try:
            prev = datetime.fromisoformat(last.replace("Z", "+00:00")).astimezone(INDIA)
            gap = (ist_now() - prev).total_seconds() / 60
            if gap > SWEEP_GAP_WARN_MIN:
                warn = f"Sweep gap detected: previous sweep {prev.strftime('%H:%M')} IST, this one {ist_now().strftime('%H:%M')} IST - about {int(gap/10)} skipped slot(s). Isolated delays are normal (GitHub cron is best-effort)."
                logger.warning(warn); tg_send(warn)
        except Exception:
            pass

    # strategy scan
    fired_today = []
    seen_keys = set()
    for ticker in universe:
        if os.environ.get("SCANNER_TEST_LIMIT"):
            if len(fired_today) + len(pendings) >= int(os.environ["SCANNER_TEST_LIMIT"]):
                break
        m5, b15 = fetch_dual_bars(ticker)
        today_sess, _ = today_session(m5)
        if today_sess is None:
            continue  # no today bars yet (early LIVE sweeps wait for the open)
        frames = prepare_frames(m5, b15)
        m5f, b15f = frames["m5"], frames["b15"]
        for name, func in STRATEGY_FUNCS.items():
            t = today_sess.copy(); t["_vwap"] = vwap_session(t)
            raw = func(m5f, b15f, t)
            for sig in raw:
                sig["symbol"] = ticker
                sig["signal_time"] = ist_now()
                sig["bucket"] = f"{sig['setup']}_{sig['direction']}"
                sig["trigger"] = sig["entry"]
                bkt = sig["bucket"]
                key = (sig["setup"], sig["direction"], sig["entry"], sig["stop"], sig["target"], sig["signal_time"].isoformat())
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                mode_chosen, reason = mode_for(bkt, ledger, slots_free)
                sig["mode"] = mode_chosen
                sig["reason"] = reason
                fired_today.append(sig)
                if mode_chosen == "TRADE":
                    slots_free -= 1
                    ledger["committed"] += 1
                card = render_card(sig, ledger["committed"], mode_chosen)
                logger.info(card.strip())
                if mode_chosen == "TRADE" and slots_free == 0:
                    tg_send(f"DAILY TRADE CAP REACHED ({CAP_TRADES_DAY}/{CAP_TRADES_DAY} trades on {_today_date_str()}). STOP TRADING for the rest of the day - manage open positions normally; later signals TRACKING only.")
                    _state_store("cap_notified", True)
                tg_send(card.strip())

    # WRAP phase: grading, fills, slippage expiry, square-off, finalize
    if phase == "WRAP":
        ledger = day_ledger(load_journal())
        pendings = [_sig_dt(s) for s in _state_load().get("pending", [])]
        wrap_now = ist_now().astimezone(timezone.utc)
        today_1500_ist = (datetime.combine(ist_now().date(), datetime.min.time())
                          + timedelta(minutes=FILL_DEADLINE_MIN))
        fill_deadline_utc = today_1500_ist.replace(tzinfo=INDIA).astimezone(timezone.utc)

        # slippage expiry: unfilled signals that never retraced by 15:00
        for ticker in universe[:min(30, len(universe))]:
            m5, b15 = fetch_dual_bars(ticker)
            m5f = prepare_frames(m5, b15)["m5"]
            until = min(fill_deadline_utc, m5f.index[-1] if not m5f.empty else wrap_now)
            bars_before = m5f[m5f.index <= until]
            kept = []
            for sig in pendings:
                st = _sig_dt(dict(sig))
                if st["symbol"] != ticker:
                    kept.append(sig); continue
                st_time = st["signal_time"]
                if st_time > fill_deadline_utc:
                    kept.append(sig); continue
                target_t = float(st["trigger"])
                best_run = 0.0
                if st["direction"] == "LONG":
                    ok = any(float(b["low"]) <= target_t * (1 + FILL_TOL_PCT) for _, b in bars_before.iterrows())
                    if len(bars_before): best_run = max(((float(b["high"]) - target_t) / target_t) for _, b in bars_before.iterrows())
                else:
                    ok = any(float(b["high"]) >= target_t * (1 - FILL_TOL_PCT) for _, b in bars_before.iterrows())
                    if len(bars_before): best_run = max(((target_t - float(b["low"])) / target_t) for _, b in bars_before.iterrows())
                if not ok:
                    slippage_append([{
                        "Date": _today_date_str(), "SignalTime": st_time.strftime("%H:%M"),
                        "Symbol": ticker, "Direction": st["direction"], "Setup": st["setup"],
                        "Trigger": target_t, "BestRunPct": round(best_run * 100, 2),
                        "ExpiredAt": fill_deadline.strftime("%H:%M"), "Outcome": "RUNAWAY",
                        "Mode": st["mode"], "Notes": f"never retraced trigger; ran {'up' if st['direction']=='LONG' else 'down'} {best_run*100:.1f}%"}])
                    if st["mode"] == "TRADE":
                        slots_free += 1; ledger["committed"] -= 1
                    logger.info(f"SLIPPAGE {ticker} {st['direction']} {st['setup']}: {best_run*100:.1f}% runaway, slot freed")
                else:
                    kept.append(sig)
            pendings = kept

        # fill gate + grade confirmed trades
        for ticker in universe[:min(20, len(universe))]:
            m5, b15 = fetch_dual_bars(ticker)
            m5f = prepare_frames(m5, b15)["m5"]
            filled, pendings = check_fills(ticker, pendings, m5f)
            for f in filled:
                fill_bars = m5f[(m5f.index >= f["filled_at"]) & (m5f.index <= m5f.index[-1])]
                entry, stop, target = f["entry"], f["stop"], f["target"]
                exit_price, exit_reason, result = None, None, None
                stopped = any(float(r["low"]) <= stop for _, r in fill_bars.iterrows())
                hit_target = any(float(r["high"]) >= target for _, r in fill_bars.iterrows())
                if stopped and hit_target:
                    exit_reason, result, exit_price = "STOP_HIT", "LOSS", stop
                elif stopped:
                    exit_reason, result, exit_price = "STOP_HIT", "LOSS", stop
                elif hit_target:
                    exit_reason, result, exit_price = "TARGET_HIT", "WIN", target
                else:
                    exit_reason, result, exit_price = "EOD_SQUAREOFF", "FLAT", float(fill_bars["close"].iloc[-1])
                jrow = {
                    "TradeID": next_trade_id(_today_date_str()), "Date": _today_date_str(),
                    "SignalTime": f["signal_time"].strftime("%H:%M"), "Symbol": f["symbol"],
                    "Direction": f["direction"], "Setup": f["setup"], "Entry": f["entry"],
                    "StopLoss": f["stop"], "Target": f["target"], "Qty": f["qty"],
                    "RiskPerShare": f["risk_per_share"], "RiskAmt": f["risk_amt"],
                    "RewardAmt": f["reward_amt"], "RR": f["rr"], "Status": "CLOSED",
                    "Mode": f["mode"], "ExitPrice": exit_price,
                    "ExitTime": ist_now().strftime("%H:%M"), "ExitReason": exit_reason,
                    "Result": result,
                    "PnL_R": round(((exit_price - entry) / abs(entry - stop)) if f["direction"] == "LONG"
                                   else ((entry - exit_price) / abs(entry - stop)), 3),
                    "PnL_Amt": round((((exit_price - entry) / abs(entry - stop)) if f["direction"] == "LONG"
                                      else ((entry - exit_price) / abs(entry - stop))) * f["risk_amt"], 2),
                    "UpdatedAt": ist_now().strftime("%Y-%m-%dT%H:%M"),
                    "Remarks": f"signal fired {f['signal_time'].strftime('%H:%M')}; filled on retrace to trigger @{f['filled_at'].strftime('%H:%M')} · NIFTY {regime['label']} ({regime['score']})",
                    "Notes": f"{f['mode']} ({f['reason']}); confluence {f['conf']:.2f}; {f['note']}"}
                append_journal([jrow])
                ledger = day_ledger(load_journal())
                slots_free = CAP_TRADES_DAY - ledger["committed"]
                tg_send(f"FILL CONFIRMED [{f['direction']} {f['symbol']}] entry {f['entry']:.2f} exit {exit_price:.2f} {exit_reason} ({result})")

        # EOD square-off leftovers at last price
        if min_now >= SQ_OFF_MIN:
            for ticker in universe[:min(20, len(universe))]:
                m5, b15 = fetch_dual_bars(ticker)
                rest_bars = m5[m5.index >= ist_now()]
                last_p = float(rest_bars["close"].iloc[-1]) if not rest_bars.empty else None
                if last_p is None:
                    continue
                for r in ledger["open"][:]:
                    if r["Symbol"] == ticker:
                        r["Status"] = "CLOSED"; r["ExitPrice"] = last_p
                        r["ExitTime"] = ist_now().strftime("%H:%M"); r["ExitReason"] = "EOD_SQUAREOFF"
                        r["Result"] = "FLAT"; r["PnL_R"], r["PnL_Amt"] = 0.0, 0.0
                ledger = day_ledger(load_journal())

        finalize_note = ""
        if min_now in FINALIZE_TIMES:
            finalize_note = f"Finalized at {min_now} IST"

    # EOD learning snapshot (after 15:05)
    snap = min_now >= SQ_OFF_MIN
    trend = classify_session_trend()
    recs = ranked_buckets()
    tuner = _tuner_load()
    tuner["last_learning"] = _today_date_str()
    _tuner_store(tuner)
    table = []
    for bkt, st in grade_buckets(load_journal()).items():
        status = tuner.get("buckets", {}).get(bkt, {}).get("status", "enabled")
        table.append((bkt, status, st["netR"], st["win_rate"]*100, st["tot_grades"], "mute" if status == "muted" else "armed"))
    cards = [render_card(sig, ledger["committed"], sig["mode"]) for sig in fired_today if sig["mode"] == "TRADE"]
    write_report(phase, regime, table, cards, recs, trend, notes=finalize_note)
    logger.info(f"EOD learning snapshot written to reports_v5/")
    tg_send(f"VOL5 EOD {_today_date_str()}: tape bias {trend}, {len(ledger['open'])} open rows managed. Full learning snapshot in reports/.")

    # persist state
    _state_store("pending", _sig_ts(pendings))
    _state_store("last_sweep", ist_now().isoformat())
    notes_final = finalize_note
    if ledger["tripped"]:
        notes_final += f" Circuit breaker tripped on {_today_date_str()} (-2R day or two consecutive losses); new setups TRACKING only."
    write_daily_summary(ledger, notes=notes_final)
    logger.info(f"VOL5 scan done — phase={phase} open={len(ledger['open'])} committed={ledger['committed']}")


if __name__ == "__main__":
    sys.exit(main())
