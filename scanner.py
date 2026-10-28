#!/usr/bin/env python3
"""
NSE F&O Intraday Scanner - SOP Volume IV (strict 1:5 reward:risk)
=================================================================
Scans the NSE F&O cash universe for four intraday setups. Writes a
Markdown report, sends Telegram alerts, and maintains a self-updating
trade journal (journal/trade_journal.csv + journal/daily_summary.csv)
with automatic exit grading, a Remarks column that narrates the exact
entry trigger price and the exit trigger time (stop, target or the
15:05 auto square-off), and a -2R circuit breaker that stops new
signal journaling for the rest of the day.

Strategies
  S1  Opening Range Breakout (09:15-09:30 range) to a strict 5R target
  S2  VWAP Pullback Continuation (trend-day pullback to VWAP)
  S3  Supertrend (10,3) 15-min flip continuation
  S4  NR7 / Inside-Day range expansion (yesterday's contraction)

A multi-factor NIFTY intraday regime engine (VWAP position, day-open
position, 30-min market structure, 1-hour momentum, reversal
retracement, fresh VWAP reclaim/loss) re-grades the index on EVERY
sweep: UP unlocks longs only, DOWN unlocks shorts only, NEUTRAL lets
both sides through to the stock-level filters. This replaces the old
single "close vs VWAP" flag that stayed DOWN for hours after a morning
selloff even after the index had visibly reversed (V-bottom days left
no buy setups and kept shorting into strength).

This script runs inside GitHub Actions every 10 minutes between 08:30
and 15:45 IST, Monday-Friday. It reads the IST clock itself:

  before 09:20        -> exits silently (nothing to do yet)
  09:20 - 09:29 PREP  -> pre-market watchlist (gaps, NR7/inside-day names)
  09:30 - 14:29 LIVE  -> full strategy scan
  14:30 - 15:45 WRAP  -> keeps grading open rows; force-closes whatever
                         is still open at the 15:05+ sweep (auto square-off).
                         Extended past 15:20 so GitHub cron delays/skips
                         cannot strand a row overnight.
  weekends / holidays -> exits (no fresh market data)

Data source: Yahoo Finance via the free `yfinance` library (no API key).
Optional test hooks (environment variables):
  SCANNER_FORCE_DATE=YYYY-MM-DD   dry-run a past session
  SCANNER_FORCE_TIME=HH:MM        IST clock override for dry-runs (default 12:30)
  SCANNER_TEST_LIMIT=N            scan only the first N symbols (debug)
"""

import os
import csv
import json
import math
import time
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from universe import FNO_STOCKS

# ------------------------------------------------------------------
# Trading parameters - the numbers the SOP is built around.
# Edit these constants to change behaviour; everything else follows.
# ------------------------------------------------------------------
CAPITAL_RS    = 20000   # account size (rupees)
RISK_RS       = 200     # money risked per trade (1% of capital)
RR            = 5.0     # strict reward : risk ratio
MIN_STOP_PCT  = 0.003   # stop must be >= 0.3% away (cost floor)
MAX_STOP_PCT  = 0.015   # stop must be <= 1.5% away (5R must stay reachable)
REACH_FRAC    = 1.00    # 5R target must fit inside ATR(14) daily range
MAX_SIGNALS   = 3       # max new signals reported per run
COST_RS       = 60      # estimated round-trip cost per trade (reporting only)
MARGIN_MULT   = 5       # SEBI intraday (MIS) leverage ceiling for sizing cap

# --- NIFTY intraday regime engine (recomputed on every sweep) ---
REG_MOMO_TH  = 0.0012   # 1-hour NIFTY rate-of-change threshold (0.12%)
REG_MOVE_TH  = 0.002    # min open->extreme move (0.2%) for the retrace vote
REG_RETRACE  = 0.5      # retrace fraction of that move to flag a reversal
REG_SCORE_TH = 2        # |net votes| needed for a directional regime
REG_NAME     = {1: "UP", -1: "DOWN", 0: "NEUTRAL"}

IST = ZoneInfo("Asia/Kolkata")   # NSE runs on India Standard Time (UTC+5:30)

STATE_FILE   = "state.json"
REPORTS_DIR  = "reports"

# ------------------------------------------------------------------
# Small clock helpers
# ------------------------------------------------------------------

def now_ist():
    """Current time in IST (or a forced past session for dry-runs)."""
    force = os.environ.get("SCANNER_FORCE_DATE", "")
    if force:
        d = dt.datetime.strptime(force, "%Y-%m-%d").date()
        hh, mm = map(int, os.environ.get("SCANNER_FORCE_TIME", "12:30").split(":"))
        return dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=IST)
    return dt.datetime.now(IST)


def phase_of(now):
    """Map the IST clock to a scanner phase (None = exit immediately)."""
    if now.weekday() >= 5:                      # Saturday / Sunday
        return None
    t = now.time()
    if dt.time(9, 20) <= t < dt.time(9, 30):
        return "PREP"
    if dt.time(9, 30) <= t < dt.time(14, 30):
        return "LIVE"
    if dt.time(14, 30) <= t <= dt.time(15, 45):   # 15:20 -> 15:45: cron-delay
        return "WRAP"                             # safety for the EOD close
    return None


# ------------------------------------------------------------------
# Data download (yfinance, batched, with light retry)
# ------------------------------------------------------------------

def fetch(symbols, period, interval):
    """Download OHLCV for a list of Yahoo symbols. Returns {sym: DataFrame}."""
    out = {}
    batch = 25
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        df = pd.DataFrame()
        for attempt in range(1, 4):
            try:
                df = yf.download(chunk, period=period, interval=interval,
                                 group_by="ticker", auto_adjust=False,
                                 progress=False, threads=True)
                break
            except Exception:
                if attempt < 3:
                    time.sleep(3 * attempt)
        if df is None or df.empty:
            continue
        for s in chunk:
            try:
                sub = df[s] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub.dropna(how="all")
                if len(sub):
                    out[s] = sub
            except Exception:
                pass
        time.sleep(1.5)          # be polite to the free API
    return out


def completed_bars(df, day, now, minutes=5):
    """Bars of one session that have FINISHED by `now` (a 09:15 5-min bar
    covers 09:15-09:20, so it counts as finished from 09:20 onwards)."""
    b = df.copy()
    try:
        b.index = b.index.tz_convert(IST)
    except (TypeError, AttributeError):
        pass
    b = b[[t.date() == day for t in b.index]]
    if minutes:
        b = b[b.index + pd.Timedelta(minutes=minutes) <= now]
    return b.sort_index()


# ------------------------------------------------------------------
# Indicators
# ------------------------------------------------------------------

def vwap_of(bars):
    """Session VWAP series (typical price weighted by volume). Yahoo's
    intraday INDEX data (^NSEI) carries zero volume, which makes a naive
    VWAP NaN for the whole session - and NaN comparisons silently read as
    'index bearish' everywhere. Fall back to the unweighted expanding
    mean of typical prices so the NIFTY regime engine always gets a live
    reference line. Stock bars keep true volume weighting."""
    tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
    if len(bars) and float(bars["Volume"].sum()) <= 0:
        return tp.expanding().mean()
    pv = (tp * bars["Volume"]).cumsum()
    v = bars["Volume"].cumsum()
    v = v.where(v > 0)
    return pv / v


def resample15(bars):
    """Aggregate 5-minute bars into 15-minute bars."""
    return bars.resample("15min", label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna()


def supertrend(df, period=10, mult=3.0):
    """Classic Supertrend. Adds 'st' (line) and 'dir' (1 long, -1 short)."""
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    hl2 = (h + l) / 2.0
    bu, bl = hl2 + mult * atr, hl2 - mult * atr
    fbu, fbl = bu.copy(), bl.copy()
    st = pd.Series(index=df.index, dtype=float)
    dr = pd.Series(index=df.index, dtype=int)
    for i in range(len(df)):
        if i == 0:
            st.iloc[i], dr.iloc[i] = fbl.iloc[i], 1
            continue
        fbu.iloc[i] = min(bu.iloc[i], fbu.iloc[i - 1]) if c.iloc[i - 1] <= fbu.iloc[i - 1] else bu.iloc[i]
        fbl.iloc[i] = max(bl.iloc[i], fbl.iloc[i - 1]) if c.iloc[i - 1] >= fbl.iloc[i - 1] else bl.iloc[i]
        if st.iloc[i - 1] == fbu.iloc[i - 1]:
            st.iloc[i] = fbl.iloc[i] if c.iloc[i] > fbu.iloc[i] else fbu.iloc[i]
        else:
            st.iloc[i] = fbu.iloc[i] if c.iloc[i] < fbl.iloc[i] else fbl.iloc[i]
        dr.iloc[i] = 1 if st.iloc[i] == fbl.iloc[i] else -1
    out = df.copy()
    out["st"], out["dir"] = st, dr
    return out


# ------------------------------------------------------------------
# Intraday regime engine - NIFTY strength/weakness, recomputed on
# EVERY sweep from completed 5-minute bars. Six votes, each
# -1 / 0 / +1:
#   vwap    close vs session VWAP
#   open    close vs the day's opening print
#   struct  last 6 bars vs prior 6 (HH+HL vs LH+LL)
#   momo    1-hour rate of change beyond +/-0.12%
#   retrace >=50% pullback off a >=0.2% open->extreme move (reversal)
#   reclaim fresh VWAP reclaim / loss inside the last hour
# Net score >= +2 -> UP (longs only), <= -2 -> DOWN (shorts only),
# in between -> NEUTRAL (both sides allowed; the stock-level filters
# in S1-S4 still decide). A one-tick hysteresis band (a held regime
# releases only when the score crosses 0) stops UP<->NEUTRAL flapping
# on marginal sweeps.
# ------------------------------------------------------------------

def regime_votes(nb, nv):
    """The six -1/0/+1 votes. nb = completed 5-min NIFTY bars,
    nv = their session VWAP series."""
    closes = nb["Close"].astype(float)
    n, c = len(nb), float(nb["Close"].iloc[-1])
    v = {}

    vw = float(nv.iloc[-1])
    v["vwap"] = 0 if math.isnan(vw) else (1 if c > vw else -1 if c < vw else 0)

    op = float(nb["Open"].iloc[0])
    v["open"] = 1 if c > op else -1 if c < op else 0

    v["struct"] = 0
    if n >= 12:
        w1, w0 = nb.iloc[-6:], nb.iloc[-12:-6]
        hh = float(w1["High"].max()) > float(w0["High"].max())
        hl = float(w1["Low"].min()) > float(w0["Low"].min())
        lh = float(w1["High"].max()) < float(w0["High"].max())
        ll = float(w1["Low"].min()) < float(w0["Low"].min())
        if hh and hl:
            v["struct"] = 1
        elif lh and ll:
            v["struct"] = -1

    v["momo"] = 0
    if n >= 13:
        roc = c / float(closes.iloc[-13]) - 1.0
        if roc > REG_MOMO_TH:
            v["momo"] = 1
        elif roc < -REG_MOMO_TH:
            v["momo"] = -1

    v["retrace"] = 0
    lo, hi = float(nb["Low"].min()), float(nb["High"].max())
    if op > 0:
        drop = (op - lo) / op
        if drop >= REG_MOVE_TH and (c - lo) / (op - lo) >= REG_RETRACE:
            v["retrace"] += 1          # fell hard, then recovered the move
        rise = (hi - op) / op
        if rise >= REG_MOVE_TH and (hi - c) / (hi - op) >= REG_RETRACE:
            v["retrace"] -= 1          # rallied hard, then gave it back

    v["reclaim"] = 0
    if n >= 13:
        d = (closes - nv.astype(float)).dropna()
        w = d.iloc[-13:].values
        if len(w) >= 2:                      # all-NaN VWAP guard
            up = [i for i in range(1, len(w)) if w[i - 1] <= 0 < w[i]]
            dn = [i for i in range(1, len(w)) if w[i - 1] >= 0 > w[i]]
            if w[-1] > 0 and up and (not dn or up[-1] > dn[-1]):
                v["reclaim"] = 1           # reclaimed VWAP inside the last hour
            elif w[-1] < 0 and dn and (not up or dn[-1] > up[-1]):
                v["reclaim"] = -1          # lost VWAP inside the last hour
    return v


def nifty_regime(nb, nv, day):
    """Composite intraday regime. Returns (side, score, report line).
    side: +1 UP (longs only), -1 DOWN (shorts only), 0 NEUTRAL (both
    allowed). The resolved regime is persisted in state.json with a
    one-tick hysteresis band so marginal sweeps cannot flip it."""
    if nb is None or len(nb) == 0:
        return 0, 0, "NIFTY regime: NA (no index bars)"
    votes = regime_votes(nb, nv)
    score = sum(votes.values())
    raw = 1 if score >= REG_SCORE_TH else (-1 if score <= -REG_SCORE_TH else 0)
    prev = load_regime(day)
    if prev == "UP" and score >= REG_SCORE_TH - 1:
        side = 1
    elif prev == "DOWN" and score <= -(REG_SCORE_TH - 1):
        side = -1
    else:
        side = raw
    save_regime(day, REG_NAME[side])
    vv = " ".join(f"{k}{'+' if x > 0 else '-' if x < 0 else '0'}"
                  for k, x in votes.items())
    held = "  [held]" if side != raw else ""
    line = (f"NIFTY regime: {REG_NAME[side]} (score {score:+d}; votes {vv})"
            f"{held}  -  UP=longs only, DOWN=shorts only, NEUTRAL=both")
    return side, score, line


def load_regime(day):
    try:
        with open(STATE_FILE) as f:
            st = json.load(f)
        if st.get("date") == str(day):
            return st.get("regime")
    except (OSError, ValueError):
        pass
    return None


def save_regime(day, name):
    st = load_state(day)
    if st.get("regime") != name:
        st["regime"] = name
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(st, f)
        except OSError:
            pass


def atr_pct_daily(daily, n=14):
    """ATR(n) of the daily series as a fraction of the last close."""
    if len(daily) < n + 1:
        return None
    h, l, c = daily["High"], daily["Low"], daily["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1).dropna()
    atr = tr.tail(n).mean()
    return float(atr / c.iloc[-1]) if c.iloc[-1] > 0 else None


def yesterday_flags(daily):
    """NR7 / inside-day flags + key levels for the LAST CLOSED session."""
    d = daily.copy()
    try:
        d.index = d.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    if len(d) < 8:
        return None
    rng = d["High"] - d["Low"]
    y = d.iloc[-1]
    p = d.iloc[-2]
    return {
        "prev_close": float(y["Close"]),
        "prev_high": float(y["High"]),
        "prev_low": float(y["Low"]),
        "nr7": bool(rng.iloc[-1] == rng.tail(7).min()),
        "inside": bool(y["High"] < p["High"] and y["Low"] > p["Low"]),
    }


# ------------------------------------------------------------------
# Risk engine - strict 1:5 geometry with guard-rails
# ------------------------------------------------------------------

def risk_math(entry, stop, side, apct):
    """Turn an entry/stop idea into a sized trade, or None if it breaks
    any SOP guard-rail (stop width, reachability, margin)."""
    if entry <= 0:
        return None
    d = abs(entry - stop)
    dpct = d / entry
    if dpct < MIN_STOP_PCT - 1e-9 or dpct > MAX_STOP_PCT + 1e-9:
        return None                       # too tight (costs) or too wide
    if apct and RR * d > REACH_FRAC * apct * entry + 1e-9:
        return None                       # 5R does not fit a normal day
    qty = int(RISK_RS // d)
    qty = min(qty, int((CAPITAL_RS * MARGIN_MULT) // entry))
    if qty < 1:
        return None
    sign = 1.0 if side == "LONG" else -1.0
    target = entry + sign * RR * d
    return {"entry": round(entry, 2), "stop": round(stop, 2),
            "target": round(target, 2), "qty": qty,
            "notional": round(qty * entry), "risk_rs": round(qty * d),
            "target_rs": round(qty * abs(target - entry)),
            "stop_pct": round(100 * d / entry, 2)}


# ------------------------------------------------------------------
# The four strategies. Each returns a signal dict or None.
# ------------------------------------------------------------------

def sig_s1(sym, b, vwap, nifty_side, apct):
    """S1: close beyond the 09:15-09:30 opening range with volume."""
    orb = b[b.index.time < dt.time(9, 30)]
    if len(orb) < 2:
        return None
    oh, ol = float(orb["High"].max()), float(orb["Low"].min())
    if oh <= 0 or not (0.004 <= (oh - ol) / oh <= 0.02):
        return None                       # degenerate range
    t0 = b.index[-1].time()
    if not (dt.time(9, 30) <= t0 <= dt.time(10, 45)):
        return None
    c = float(b["Close"].iloc[-1])
    vw = float(vwap.iloc[-1]) if not math.isnan(vwap.iloc[-1]) else None
    vol = float(b["Volume"].iloc[-1])
    avg_vol = float(b["Volume"].mean()) or 1.0
    if vw is None:
        return None
    long_ok = c > oh and c > vw and nifty_side >= 0 and vol >= 2.0 * avg_vol
    short_ok = c < ol and c < vw and nifty_side <= 0 and vol >= 2.0 * avg_vol
    if long_ok:
        r = risk_math(c, ol, "LONG", apct)
    elif short_ok:
        r = risk_math(c, oh, "SHORT", apct)
    else:
        return None
    return make_signal("S1", sym, r, "ORB break, 2x volume, VWAP/NIFTY aligned")


def sig_s2(sym, b, vwap, ctx, nifty_side, apct):
    """S2: trend day pulls back to VWAP and reclaims it (both sides)."""
    t0 = b.index[-1].time()
    if not (dt.time(9, 45) <= t0 <= dt.time(13, 0)) or len(b) < 12:
        return None
    c = float(b["Close"].iloc[-1])
    vw = float(vwap.iloc[-1])
    if math.isnan(vw):
        return None
    open_px = float(b["Open"].iloc[0])
    vol = float(b["Volume"].iloc[-1])
    avg_vol = float(b["Volume"].mean()) or 1.0
    if vol < 1.5 * avg_vol:
        return None
    if open_px >= 1.002 * ctx["prev_close"]:          # trend-UP day
        if (b["Close"] > vwap).mean() < 0.7:
            return None
        touched = float(b["Low"].iloc[-1]) <= vw and c > vw
        if not touched or nifty_side < 0:
            return None
        stop = min(float(b["Low"].iloc[-3:].min()), c * (1 - MIN_STOP_PCT))
        r = risk_math(c, stop, "LONG", apct)
    elif open_px <= 0.998 * ctx["prev_close"]:        # trend-DOWN day
        if (b["Close"] < vwap).mean() < 0.7:
            return None
        touched = float(b["High"].iloc[-1]) >= vw and c < vw
        if not touched or nifty_side > 0:
            return None
        stop = max(float(b["High"].iloc[-3:].max()), c * (1 + MIN_STOP_PCT))
        r = risk_math(c, stop, "SHORT", apct)
    else:
        return None
    return make_signal("S2", sym, r, "VWAP pullback rejection on trend day")


def sig_s3(sym, b, vwap, nifty_side, apct):
    """S3: fresh 15-min Supertrend flip with volume and VWAP support.
    Stop = the nearer of the Supertrend line and the flip candle's extreme
    (floored/ceilinged at the 0.3% cost boundary)."""
    t0 = b.index[-1].time()
    if not (dt.time(9, 45) <= t0 <= dt.time(14, 0)) or len(b) < 40:
        return None
    st = supertrend(resample15(b))
    if len(st) < 4:
        return None
    d = st["dir"]
    flips = np.flatnonzero(np.diff(d.values) != 0)
    if len(flips) == 0 or (len(d) - 1 - flips[-1]) > 2:
        return None                       # flip must be fresh (<= 3 bars)
    flip_bar = st.iloc[flips[-1] + 1]     # the candle that started the new trend
    c = float(b["Close"].iloc[-1])
    vw = float(vwap.iloc[-1])
    if math.isnan(vw):
        return None
    vol = float(b["Volume"].iloc[-1])
    avg_vol = float(b["Volume"].mean()) or 1.0
    if vol < 1.5 * avg_vol:
        return None
    if d.iloc[-1] == 1 and c > vw and nifty_side >= 0:
        nearer = max(float(st["st"].iloc[-1]), float(flip_bar["Low"]))
        stop = min(nearer, c * (1 - MIN_STOP_PCT))
        r = risk_math(c, stop, "LONG", apct)
    elif d.iloc[-1] == -1 and c < vw and nifty_side <= 0:
        nearer = min(float(st["st"].iloc[-1]), float(flip_bar["High"]))
        stop = max(nearer, c * (1 + MIN_STOP_PCT))
        r = risk_math(c, stop, "SHORT", apct)
    else:
        return None
    return make_signal("S3", sym, r, "Supertrend(10,3) 15-min flip + volume")


def sig_s4(sym, b, vwap, ctx, nifty_side, apct):
    """S4: NR7 / inside-day expansion breakout through yesterday's range."""
    if not (ctx["nr7"] or ctx["inside"]):
        return None
    t0 = b.index[-1].time()
    if not (dt.time(9, 45) <= t0 <= dt.time(13, 30)):
        return None
    c = float(b["Close"].iloc[-1])
    vw = float(vwap.iloc[-1])
    if math.isnan(vw):
        return None
    vol = float(b["Volume"].iloc[-1])
    avg_vol = float(b["Volume"].mean()) or 1.0
    if vol < 1.5 * avg_vol:
        return None
    bar_lo = float(b["Low"].iloc[-1])
    bar_hi = float(b["High"].iloc[-1])
    if c > ctx["prev_high"] and c > vw and nifty_side >= 0:
        stop = min(bar_lo, c * (1 - MIN_STOP_PCT))
        r = risk_math(c, stop, "LONG", apct)
    elif c < ctx["prev_low"] and c < vw and nifty_side <= 0:
        stop = max(bar_hi, c * (1 + MIN_STOP_PCT))
        r = risk_math(c, stop, "SHORT", apct)
    else:
        return None
    trig = "NR7" if ctx["nr7"] else "Inside-day"
    return make_signal("S4", sym, r, trig + " range expansion")


def make_signal(sid, sym, r, why):
    if r is None:
        return None
    r.update({"strategy": sid, "symbol": sym.replace(".NS", ""), "why": why,
              "key": f"{sym}|{sid}|{'LONG' if r['target'] > r['entry'] else 'SHORT'}"})
    return r


# ------------------------------------------------------------------
# Reporting, state (de-duplication) and Telegram
# ------------------------------------------------------------------

def load_state(day):
    try:
        with open(STATE_FILE) as f:
            st = json.load(f)
        if st.get("date") == str(day):
            return st
    except (OSError, ValueError):
        pass
    return {"date": str(day), "fired": {}}


def write_report(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def send_telegram(text):
    tok = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return
    for i in range(0, len(text), 3500):
        try:
            requests.post(
                f"https://api.telegram.org/bot{tok}/sendMessage",
                data={"chat_id": chat, "text": text[i:i + 3500]},
                timeout=15)
        except Exception:
            pass


def card(s, new):
    tag = "[NEW] " if new else ""
    return (f"{tag}[{s['strategy']}][{s['key'].split('|')[-1]}] {s['symbol']}  "
            f"@ {s['entry']} | SL {s['stop']} ({s['stop_pct']}%) | "
            f"TGT {s['target']} (5R) | qty {s['qty']} | "
            f"risk Rs {s['risk_rs']} | reward Rs {s['target_rs']}\n"
            f"     setup: {s['why']}")


# ------------------------------------------------------------------
# Trade journal - journal/trade_journal.csv + journal/daily_summary.csv
# Mirrors the manual journal from the SOP, maintained automatically:
#   * a fired signal appends an OPEN row to trade_journal.csv
#   * every later run grades open rows against completed 5-minute bars
#     (STOP_HIT / TARGET_HIT; stop checked first inside one bar, because
#     a single bar cannot reveal which level was touched first - the
#     scanner always grades conservatively)
#   * the WRAP sweep force-closes leftovers as EOD_SQUAREOFF at the
#     last traded price; rows stranded from earlier sessions are closed
#     against that date's daily candle
#   * daily_summary.csv keeps one evolving row per trading day
#   * circuit breaker (SOP money rule): day P/L <= -2R or 2 consecutive
#     losses stop NEW signal journaling until the next session
# The journal grades the SIGNAL (entry at signal price, no slippage),
# not your broker fill - your execution ledger stays with your broker.
# ------------------------------------------------------------------

JOURNAL_DIR = "journal"
TJ_FILE = os.path.join(JOURNAL_DIR, "trade_journal.csv")
DS_FILE = os.path.join(JOURNAL_DIR, "daily_summary.csv")
DAILY_LOSS_R = 2.0        # circuit breaker: day P/L that trips the breaker
LOSS_STREAK_TRIP = 2      # circuit breaker: consecutive losses that trip it
RR_LABEL = "1:5"
SQUAREOFF_GATE = dt.time(15, 5)   # WRAP keeps grading until 15:05 IST
REMARK_EXIT_LABELS = {"STOP_HIT": "STOP HIT", "TARGET_HIT": "TARGET HIT",
                      "EOD_SQUAREOFF": "AUTO SQUARE-OFF"}

TJ_COLS = ("TradeID", "Date", "SignalTime", "Symbol", "Direction", "Setup",
           "Entry", "StopLoss", "Target", "Qty", "RiskPerShare", "RiskAmt",
           "RewardAmt", "RR", "Status", "ExitPrice", "ExitTime", "ExitReason",
           "Result", "PnL_R", "PnL_Amt", "UpdatedAt", "Remarks", "Notes")
DS_COLS = ("Date", "Trades", "Wins", "Losses", "Flats", "Open", "WinRatePct",
           "NetR", "NetPnlAmt", "BreakerTripped", "DayVerdict", "Notes")
SETUP_NAMES = {"S1": "S1_ORB_BREAKOUT", "S2": "S2_VWAP_PULLBACK",
               "S3": "S3_SUPERTREND_FLIP", "S4": "S4_NR7_EXPANSION"}


def read_csv_rows(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except (OSError, UnicodeDecodeError):
        return []


def write_csv_rows(path, cols, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cols), extrasaction="ignore",
                           restval="")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    os.replace(tmp, path)


def _fmt_signed(v, dp):
    if abs(v) < 0.5 * 10 ** (-dp):
        return f"{0:.{dp}f}"
    return f"{v:+.{dp}f}" if v > 0 else f"{v:.{dp}f}"


def unique_trade_id(rows, day, sym, side):
    base = f"{day:%Y%m%d}-{sym}-{side}"
    existing = {r.get("TradeID", "") for r in rows}
    tid, n = base, 2
    while tid in existing:
        tid = f"{base}-{n}"
        n += 1
    return tid


def entry_remark(entry, signal_time, regime=""):
    """First half of the Remarks narrative (written when the row opens).
    regime appends the NIFTY regime snapshot at fire time for audit."""
    txt = f"Entry triggered @ {entry} at {signal_time} IST"
    if regime:
        txt += f" | {regime}"
    return txt


def exit_remark(reason, px, exit_time, pnl_r):
    """Second half of the Remarks narrative (appended when the row closes)."""
    label = REMARK_EXIT_LABELS.get(reason, reason)
    return f"{label} @ {px} at {exit_time} IST ({pnl_r:+.2f}R)"


def backfill_remarks(rows):
    """Rebuild the Remarks narrative for rows written by an older journal
    version (idempotent: rows that already carry Remarks are untouched)."""
    n = 0
    for r in rows:
        if r.get("Remarks") or not r.get("Entry"):
            continue
        txt = entry_remark(r["Entry"], r.get("SignalTime") or "?")
        if r.get("Status") == "CLOSED" and r.get("ExitPrice"):
            reason = r.get("ExitReason") or "EXIT"
            label = REMARK_EXIT_LABELS.get(reason, reason)
            try:
                pnl = f" ({float(r['PnL_R']):+.2f}R)" if r.get("PnL_R") else ""
            except ValueError:
                pnl = ""
            txt += (f"; {label} @ {r['ExitPrice']} at "
                    f"{r.get('ExitTime') or '?'} IST{pnl}")
        r["Remarks"] = txt
        n += 1
    return n


def signal_to_row(sig, day, now, trade_id):
    """Turn a fired signal into an OPEN journal row."""
    side = "LONG" if sig["target"] > sig["entry"] else "SHORT"
    rps = round(abs(sig["entry"] - sig["stop"]), 2)
    qty = int(sig["qty"])
    return {
        "TradeID": trade_id,
        "Date": str(day),
        "SignalTime": f"{now:%H:%M}",
        "Symbol": sig["symbol"],
        "Direction": side,
        "Setup": SETUP_NAMES.get(sig["strategy"], sig["strategy"]),
        "Entry": f"{sig['entry']:.2f}",
        "StopLoss": f"{sig['stop']:.2f}",
        "Target": f"{sig['target']:.2f}",
        "Qty": qty,
        "RiskPerShare": f"{rps:.2f}",
        "RiskAmt": f"{qty * rps:.2f}",
        "RewardAmt": f"{qty * abs(sig['target'] - sig['entry']):.2f}",
        "RR": RR_LABEL,
        "Status": "OPEN",
        "ExitPrice": "", "ExitTime": "", "ExitReason": "",
        "Result": "", "PnL_R": "", "PnL_Amt": "",
        "UpdatedAt": f"{now:%Y-%m-%d %H:%M:%S}",
        "Remarks": entry_remark(f"{sig['entry']:.2f}", f"{now:%H:%M}",
                                sig.get("regime", "")),
        "Notes": sig.get("why", ""),
    }


def close_row(row, px, exit_time, reason, now, remark_note=""):
    """Grade one row at an exit price, stamp the audit columns and append
    the exit narrative to Remarks. remark_note adds an honesty suffix
    (e.g. 'graded on daily candle' for stale rows)."""
    sign = 1.0 if row["Direction"] == "LONG" else -1.0
    entry = float(row["Entry"])
    rps = float(row["RiskPerShare"]) or 1e-9
    qty = int(float(row["Qty"]))
    pnl_r = round(sign * (px - entry) / rps, 3)
    row["Status"] = "CLOSED"
    row["ExitPrice"] = f"{px:.2f}"
    row["ExitTime"] = exit_time
    row["ExitReason"] = reason
    row["PnL_R"] = f"{pnl_r:.3f}"
    row["PnL_Amt"] = f"{round(sign * (px - entry) * qty, 2):.2f}"
    row["Result"] = ("WIN" if pnl_r > 0.0005
                     else ("LOSS" if pnl_r < -0.0005 else "FLAT"))
    row["UpdatedAt"] = f"{now:%Y-%m-%d %H:%M:%S}"
    base = row.get("Remarks") or entry_remark(
        row.get("Entry", ""), row.get("SignalTime") or "?")
    tail = f" ({remark_note})" if remark_note else ""
    row["Remarks"] = f"{base}; {exit_remark(reason, f'{px:.2f}', exit_time, pnl_r)}{tail}"


def _bar_hit(row, hi, lo):
    """(stop_touched, target_touched) for one bar; stop checked first."""
    sl, tg = float(row["StopLoss"]), float(row["Target"])
    if row["Direction"] == "LONG":
        return lo <= sl + 1e-9, hi >= tg - 1e-9
    return hi >= sl - 1e-9, lo <= tg + 1e-9


def _intraday_for_date(intra, sym, rdate):
    """One past session's 5-minute bars for a symbol, or None.
    yfinance keeps roughly five sessions of 5-minute history."""
    if not intra:
        return None
    raw = intra.get(sym)
    if raw is None or not len(raw):
        return None
    df = raw.copy()
    try:
        df.index = df.index.tz_convert(IST)
    except (TypeError, AttributeError):
        pass
    df = df[[str(t.date()) == rdate for t in df.index]]
    return df.sort_index() if len(df) else None


def _first_touch(row, bb):
    """Walk completed bars chronologically; return (px, timestamp, reason)
    for the first bar touching stop or target (stop-first inside one bar),
    or None."""
    for ts, br in bb.iterrows():
        hit_stop, hit_tgt = _bar_hit(row, float(br["High"]),
                                     float(br["Low"]))
        if hit_stop:
            return float(row["StopLoss"]), ts, "STOP_HIT"
        if hit_tgt:
            return float(row["Target"]), ts, "TARGET_HIT"
    return None


def monitor_journal(rows, bars, daily, day, now, intra=None):
    """Grade OPEN rows. Today's rows are checked against completed 5-min
    bars every sweep. Rows stranded from earlier sessions are re-graded
    on THAT DATE's own 5-minute bars (exact trigger price and time), and
    only fall back to the daily candle once the intraday history has
    expired - the daily candle cannot reveal intraday order, so there the
    stop wins conservatively if both levels sit inside the day's range.
    Returns (graded_today, stale_closed)."""
    graded = stale = 0
    for row in rows:
        if row.get("Status") != "OPEN":
            continue
        rdate = row.get("Date", "")
        sym = row.get("Symbol", "") + ".NS"
        if rdate == str(day):
            b = bars.get(sym)
            if b is None or not len(b):
                continue
            try:
                hh, mm = map(int, str(row.get("SignalTime", "9:15")).split(":"))
            except ValueError:
                hh, mm = 9, 15
            bb = b[[t.time() >= dt.time(hh, mm) for t in b.index]]
            hit = _first_touch(row, bb)
            if hit:
                close_row(row, hit[0], f"{hit[1]:%H:%M}", hit[2], now)
                graded += 1
        else:
            px = reason = tlabel = None
            note = "graded on daily candle"
            b5 = _intraday_for_date(intra, sym, rdate)
            if b5 is not None and len(b5):
                try:
                    hh, mm = map(int,
                                 str(row.get("SignalTime", "9:15")).split(":"))
                except ValueError:
                    hh, mm = 9, 15
                bb = b5[[t.time() >= dt.time(hh, mm) for t in b5.index]]
                hit = _first_touch(row, bb)
                if hit:
                    px, reason, tlabel = hit[0], hit[2], f"{hit[1]:%H:%M}"
                else:
                    px = float(b5["Close"].iloc[-1])
                    reason = "EOD_SQUAREOFF"
                    tlabel = f"{b5.index[-1]:%H:%M}"
                note = ""
            if px is None:
                d = daily.get(sym)
                if d is not None and len(d):
                    idx = pd.to_datetime(d.index)
                    try:
                        idx = idx.tz_localize(None)
                    except (TypeError, AttributeError):
                        pass
                    d2 = d[[str(t.date()) == rdate for t in idx]]
                    if len(d2):
                        r = d2.iloc[-1]
                        hit_stop, hit_tgt = _bar_hit(row, float(r["High"]),
                                                     float(r["Low"]))
                        if hit_stop:
                            px, reason = float(row["StopLoss"]), "STOP_HIT"
                        elif hit_tgt:
                            px, reason = float(row["Target"]), "TARGET_HIT"
                        else:
                            px, reason = float(r["Close"]), "EOD_SQUAREOFF"
                if px is None:              # no data at all: scratch flat
                    px, reason = float(row["Entry"]), "EOD_SQUAREOFF"
                tlabel = "15:05"
            close_row(row, px, tlabel, reason, now, remark_note=note)
            stale += 1
    return graded, stale


def force_close_eod(rows, bars, daily, day, now):
    """WRAP sweep: close every OPEN row of today at the last traded price."""
    closed = 0
    for row in rows:
        if row.get("Status") != "OPEN" or row.get("Date") != str(day):
            continue
        sym = row.get("Symbol", "") + ".NS"
        px, tlabel = None, None
        b = bars.get(sym)
        if b is not None and len(b):
            px = float(b["Close"].iloc[-1])
            tlabel = f"{b.index[-1]:%H:%M}"
        else:
            d = daily.get(sym)
            if d is not None and len(d):
                idx = pd.to_datetime(d.index)
                try:
                    idx = idx.tz_localize(None)
                except (TypeError, AttributeError):
                    pass
                d2 = d[[t.date() == day for t in idx]]
                if len(d2):
                    px = float(d2.iloc[-1]["Close"])
        if px is None:
            px = float(row["Entry"])
        close_row(row, px, tlabel or "15:05", "EOD_SQUAREOFF", now)
        closed += 1
    return closed


def breaker_state(rows, day):
    """Circuit breaker computed from CLOSED rows only (naturally sticky:
    closed rows never change). Triggers: day P/L <= -2R, or 2 consecutive
    losses. Returns (tripped, note)."""
    todays = sorted([r for r in rows
                     if r.get("Date") == str(day) and r.get("Status") == "CLOSED"],
                    key=lambda r: r.get("ExitTime") or "")
    realized = sum(float(r["PnL_R"]) for r in todays if r.get("PnL_R"))
    streak = 0
    for r in reversed(todays):
        if r.get("Result") == "LOSS":
            streak += 1
        else:
            break
    parts = []
    if realized <= -DAILY_LOSS_R - 1e-9:
        parts.append(f"day limit -{DAILY_LOSS_R:g}R")
    if streak >= LOSS_STREAK_TRIP:
        parts.append(f"{streak}-loss streak")
    if parts:
        return True, "circuit breaker: " + " / ".join(parts)
    return False, ""


def upsert_daily_summary(ds_rows, tj_rows, day):
    """One evolving row per day in daily_summary.csv."""
    todays = [r for r in tj_rows if r.get("Date") == str(day)]
    closed = [r for r in todays if r.get("Status") == "CLOSED"]
    wins = sum(1 for r in closed if r.get("Result") == "WIN")
    losses = sum(1 for r in closed if r.get("Result") == "LOSS")
    flats = sum(1 for r in closed if r.get("Result") == "FLAT")
    netr = round(sum(float(r["PnL_R"]) for r in closed), 3)
    netamt = round(sum(float(r["PnL_Amt"]) for r in closed), 2)
    wr = round(100.0 * wins / len(closed), 1) if closed else 0.0
    tripped, note = breaker_state(tj_rows, day)
    row = {"Date": str(day), "Trades": len(todays), "Wins": wins,
           "Losses": losses, "Flats": flats, "Open": len(todays) - len(closed),
           "WinRatePct": f"{wr:.1f}", "NetR": _fmt_signed(netr, 3),
           "NetPnlAmt": _fmt_signed(netamt, 2),
           "BreakerTripped": "YES" if tripped else "NO",
           "DayVerdict": "FAILED" if tripped else "PASSED",
           "Notes": note if tripped else
                    ("" if todays else "no qualifying setups today")}
    for i, r in enumerate(ds_rows):
        if r.get("Date") == str(day):
            ds_rows[i] = row
            break
    else:
        ds_rows.append(row)
    ds_rows.sort(key=lambda r: r.get("Date", ""))


def journal_status(rows, day, tripped, bnote):
    """Short journal block appended to every live-scan report."""
    todays = [r for r in rows if r.get("Date") == str(day)]
    closed = [r for r in todays if r.get("Status") == "CLOSED"]
    open_n = len(todays) - len(closed)
    wins = sum(1 for r in closed if r.get("Result") == "WIN")
    losses = sum(1 for r in closed if r.get("Result") == "LOSS")
    netr = sum(float(r["PnL_R"]) for r in closed if r.get("PnL_R"))
    line = (f"Journal today: {len(todays)} signal(s) | open {open_n} | "
            f"closed {len(closed)} ({wins}W/{losses}L) | "
            f"day P/L {_fmt_signed(netr, 3)}R")
    if tripped:
        line += " | BREAKER: " + bnote
    return [line]


def journal_lines(rows, day, tripped, bnote):
    """Detailed per-trade block for the WRAP report."""
    todays = [r for r in rows if r.get("Date") == str(day)]
    if not todays:
        return ["No signals fired today - nothing to journal."]
    out = [f"Journaled today: {len(todays)} signal(s)."]
    for r in todays:
        if r.get("Status") == "CLOSED":
            out.append(f"  {r['TradeID']}: entry {r['Entry']} -> exit "
                       f"{r['ExitPrice']} | {r['Result']} {r['PnL_R']}R "
                       f"({r['ExitReason']})")
        else:
            out.append(f"  {r['TradeID']}: entry {r['Entry']} | still OPEN")
    closed = [r for r in todays if r.get("Status") == "CLOSED"]
    netr = sum(float(r["PnL_R"]) for r in closed if r.get("PnL_R"))
    out.append(f"Day P/L: {_fmt_signed(netr, 3)}R | breaker: "
               + (f"YES ({bnote})" if tripped else "no"))
    return out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    now = now_ist()
    ph = phase_of(now)
    print(f"scanner at {now:%Y-%m-%d %H:%M} IST -> phase {ph}")
    if ph is None:
        print("Nothing to do right now. Exiting.")
        return

    day = now.date()
    uni = [s + ".NS" for s in FNO_STOCKS]
    limit = int(os.environ.get("SCANNER_TEST_LIMIT", "0") or 0)
    if limit:
        uni = uni[:limit]
        print(f"TEST LIMIT active: scanning {len(uni)} symbols only")

    intra = fetch(uni, "5d", "5m")
    daily = fetch(uni, "3mo", "1d")
    nifty = fetch(["^NSEI"], "5d", "5m").get("^NSEI")

    # Market-closed / holiday guard: no fresh bar for today anywhere.
    sample = intra.get(uni[0])
    if sample is None and intra:
        sample = next(iter(intra.values()))
    if sample is not None and not os.environ.get("SCANNER_FORCE_DATE"):
        last = max(t.date() for t in sample.index)
        if last != day:
            print(f"No market data for today ({day}); last data {last}. "
                  "Holiday or closed market - exiting.")
            return
    if not intra:
        print("No data returned at all - exiting.")
        return

    nifty_side, nifty_score = 0, 0
    nifty_tag = "NIFTY regime: NA (no index data yet)"
    if nifty is not None:
        nb = completed_bars(nifty, day, now)
        if len(nb):
            nifty_side, nifty_score, nifty_tag = nifty_regime(
                nb, vwap_of(nb), day)

    ctx, bars, vwaps = {}, {}, {}
    minb = 1 if ph == "PREP" else 6   # PREP only needs the first bar (gaps)
    for s in intra:
        b = completed_bars(intra[s], day, now)
        if len(b) < minb or float(b["Volume"].sum()) <= 0:
            continue
        bars[s] = b
        vwaps[s] = vwap_of(b)
        d = daily.get(s)
        if d is not None:
            try:
                d = d[[t.date() < day for t in d.index]]   # drop today's row
            except TypeError:
                pass
            f = yesterday_flags(d) if len(d) >= 8 else None
            if f:
                f["apct"] = atr_pct_daily(d)
                ctx[s] = f

    # ---- trade journal: load, grade open rows, circuit breaker ----
    tj_rows = read_csv_rows(TJ_FILE)
    backfill_remarks(tj_rows)          # fill Remarks for pre-upgrade rows
    ds_rows = read_csv_rows(DS_FILE)
    graded, stale = monitor_journal(tj_rows, bars, daily, day, now,
                                    intra=intra)
    if graded or stale:
        print(f"journal: graded {graded} row(s) today, closed {stale} stale row(s)")
    tripped, bnote = breaker_state(tj_rows, day)

    lines = [f"# NSE F&O Scanner - {day:%d %b %Y} {now:%H:%M} IST (phase {ph})",
             "", f"{nifty_tag}  |  universe scanned: {len(bars)} stocks", ""]

    if ph == "PREP":
        lines += ["## Pre-market watchlist (PREP phase)", ""]
        gaps, nr7 = [], []
        for s, b in bars.items():
            f = ctx.get(s)
            if not f:
                continue
            gap = 100 * (float(b["Open"].iloc[0]) / f["prev_close"] - 1)
            if abs(gap) >= 1.5:
                gaps.append((s.replace(".NS", ""), round(gap, 1)))
            if f["nr7"] or f["inside"]:
                nr7.append(s.replace(".NS", "") + (" (NR7)" if f["nr7"] else " (ID)"))
        lines.append("**Gap movers (>= 1.5%):** " +
                     (", ".join(f"{a} {g:+g}%" for a, g in sorted(gaps, key=lambda x: -abs(x[1]))[:12]) or "none"))
        lines.append("")
        lines.append("**NR7 / Inside-day (S4 candidates today):** " +
                     (", ".join(sorted(nr7)) or "none"))
        lines.append("")
        lines.append("Wait for 09:30. The LIVE sweep will start flagging S1-S4 setups after that.")
        if stale:
            lines += ["", f"Journal: force-closed {stale} stale open signal(s) left "
                          "over from earlier sessions (graded on daily candles)."]
    elif ph == "WRAP":
        lines += ["## Square-off window (WRAP)", ""]
        auto_closed = 0
        if now.time() >= SQUAREOFF_GATE:
            auto_closed = force_close_eod(tj_rows, bars, daily, day, now)
            if auto_closed:
                lines.append(f"Auto square-off: force-closed {auto_closed} "
                             "signal(s) still open at 15:05 (last traded "
                             "price, ExitReason=EOD_SQUAREOFF).")
                lines.append("")
        else:
            lines.append("Before 15:05 - stop and target levels are still "
                         "live; the journal keeps grading on every sweep.")
            lines.append("")
        tripped, bnote = breaker_state(tj_rows, day)
        lines += journal_lines(tj_rows, day, tripped, bnote)
        lines += ["", "Square off any open MIS positions now. The broker auto "
                      "square-off follows; never widen a stop."]
    else:
        lines += ["## Strategy signals (live scan)", ""]
        found = []
        for s, b in bars.items():
            f = ctx.get(s)
            if not f:
                continue
            v = vwaps[s]
            for fn, args in ((sig_s1, (b, v, nifty_side, f["apct"])),
                             (sig_s2, (b, v, f, nifty_side, f["apct"])),
                             (sig_s3, (b, v, nifty_side, f["apct"])),
                             (sig_s4, (b, v, f, nifty_side, f["apct"]))):
                try:
                    r = fn(s, *args)
                except Exception:
                    r = None
                if r:
                    found.append(r)
        state = load_state(day)
        new = [r for r in found if r["key"] not in state["fired"]]
        take = new[:MAX_SIGNALS]
        for r in take:
            state["fired"][r["key"]] = f"{now:%H:%M}"
        with open(STATE_FILE, "w") as fh:
            json.dump(state, fh)

        # Journal newly fired signals unless the circuit breaker is tripped.
        journaled = 0
        if tripped:
            lines += [f"CIRCUIT BREAKER ACTIVE ({bnote}). New setups are watch-only: "
                      "take nothing new today; manage open positions normally.", ""]
        else:
            for r in take:
                try:
                    r["regime"] = (f"NIFTY {REG_NAME[nifty_side]} "
                                   f"(score {nifty_score:+d})")
                    side = "LONG" if r["target"] > r["entry"] else "SHORT"
                    tid = unique_trade_id(tj_rows, day, r["symbol"], side)
                    tj_rows.append(signal_to_row(r, day, now, tid))
                    journaled += 1
                except Exception as exc:
                    print("journal append failed:", exc)

        if found:
            fresh = {r["key"] for r in take}
            lines += [card(r, r["key"] in fresh) for r in
                      sorted(found, key=lambda r: r["key"])]
        else:
            lines.append("No qualifying setups right now.")
        lines += ["", f"Signals detected this run: {len(found)}; "
                      f"new today: {len(new)} (reporting cap {MAX_SIGNALS}/run); "
                      f"journaled this run: {journaled}."]
        lines += journal_status(tj_rows, day, tripped, bnote)
        lines += ["", "Execution rules: place a bracket order (entry + SL + 5R target) "
                      "or a market/limit entry plus a separate SL-M order. "
                      "If neither level hits by 14:45, square off by 15:05. "
                      "Never widen a stop. Never take a second entry in the same "
                      "stock on the same strategy today."]

    upsert_daily_summary(ds_rows, tj_rows, day)
    write_csv_rows(TJ_FILE, TJ_COLS, tj_rows)
    write_csv_rows(DS_FILE, DS_COLS, ds_rows)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    write_report(os.path.join(REPORTS_DIR, f"report_{day:%Y-%m-%d}.md"), lines)

    send_telegram("\n".join(lines))
    print("\n".join(lines))
    print(f"\nReport written: {REPORTS_DIR}/report_{day:%Y-%m-%d}.md")


if __name__ == "__main__":
    main()
