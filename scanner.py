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
sweep and prints it in the report header. It is ADVISORY ONLY: it
never blocks a side, because single stocks can be strong while the
index is weak and vice versa - the stock-level filters in S1-S4
decide every trade.

Two self-correcting layers sit on top of the four strategies:

  * Fill-confirmation gate (slippage log). A fired signal becomes a
    journal trade only if price retraces to its trigger price within
    0.10% after the signal - i.e. a limit order at the signal price
    would really have filled. Signals whose price runs away and never
    comes back are logged in journal/slippage_log.csv instead and
    never pollute the win/loss statistics.

  * Auto-tune engine (trend dominance). Over the last 5 trading
    sessions every (strategy, side) bucket is graded from the journal
    and the slippage log on every sweep: buckets with negative net R
    (>= 2 graded fills) or chronic slippage (>= 3 runaways) are muted;
    winners keep firing. The scanner thus keeps re-aligning itself with
    the dominant recent trend of its own accuracy. At the 15:05+ sweep
    an end-of-day learning snapshot (window INCLUDING today) shows
    exactly what the scanner will apply from the next session.

Daily trade cap and the tracking lab (journal everything):

  * every fired signal carries a Mode. TRADE = actionable, counts
    toward the daily budget of CAP_TRADES_DAY = 10 trades. TRACKING =
    journaled and graded exactly like a trade, but explicitly NOT to
    be taken: it keeps measuring the accuracy of strategies that the
    auto-tuner muted, that fire on a circuit-breaker day, or that fire
    after the cap. When the 10th TRADE fills, the scanner sends an
    immediate Telegram "stop trading for the day" message; signals
    keep flowing as TRACKING until the 13:30 signal cutoff. The budget
    counts committed trade slots (fills + signals still awaiting fill
    confirmation), so a burst cannot overshoot the cap, and a signal
    that ends up in the slippage log frees its slot.

  * tracking rows feed the auto-tuner like real rows, so a muted
    strategy can EARN ITS WAY BACK by paper-winning inside the journal
    - the learning loop never starves.

This script runs inside GitHub Actions every 10 minutes between 08:30
and 15:45 IST, Monday-Friday. It reads the IST clock itself:

  before 09:20        -> exits silently (nothing to do yet)
  09:20 - 09:29 PREP  -> pre-market watchlist (gaps, NR7/inside-day names)
  09:30 - 13:29 LIVE  -> full strategy scan; the signal window ends at
                         13:30 IST - after that no NEW signals fire and
                         the scanner only grades and manages open rows
  13:30 - 15:45 WRAP  -> grading only; force-closes whatever is still
                         open at the 15:05+ sweep (auto square-off) and
                         prints the EOD learning snapshot. Extended past
                         15:20 so GitHub cron delays/skips cannot strand
                         a row overnight.
  weekends / holidays -> exits (no fresh market data)

Any sweep that wakes up more than 25 minutes after the previous one
prints a "Sweep gap detected" warning in the report and Telegram, so
GitHub's cron delays and skipped runs are visible instead of silently
missing signals.

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
CAP_TRADES_DAY = 10    # daily trade budget: the 10th TRADE fill stops trading
COST_RS       = 60      # estimated round-trip cost per trade (reporting only)
MARGIN_MULT   = 5       # SEBI intraday (MIS) leverage ceiling for sizing cap

# --- NIFTY intraday regime engine (recomputed on every sweep) ---
REG_MOMO_TH  = 0.0012   # 1-hour NIFTY rate-of-change threshold (0.12%)
REG_MOVE_TH  = 0.002    # min open->extreme move (0.2%) for the retrace vote
REG_RETRACE  = 0.5      # retrace fraction of that move to flag a reversal
REG_SCORE_TH = 2        # |net votes| needed for a directional regime
REG_NAME     = {1: "UP", -1: "DOWN", 0: "NEUTRAL"}

# --- fill-confirmation gate (slippage tracking) ---
FILL_TOL_PCT  = 0.10    # signal must retrace to within 0.10% of trigger
FILL_DEADLINE = dt.time(15, 0)   # unfilled signals expire to slippage here

# --- auto-tune engine (last-N-session win/loss dominance) ---
AT_WINDOW_DAYS = 5      # trading sessions of history feeding the tuner
AT_MIN_TRADES  = 2      # graded fills needed before net R can mute a bucket
AT_SLIP_MIN    = 3      # runaway signals that mark a bucket as chronic chase

# --- cron health: warn in the report when sweeps were skipped/delayed ---
SWEEP_GAP_WARN_MIN = 25   # > 2 skipped 10-min slots before the report warns

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
    if dt.time(9, 30) <= t < dt.time(13, 30):     # signal window ends 13:30
        return "LIVE"
    if dt.time(13, 30) <= t <= dt.time(15, 45):   # 15:20 -> 15:45: cron-delay
        return "WRAP"                             # safety for the EOD close
    return None


def sweep_delay_note(prev_hhmm, now):
    """Report-line warning when GitHub's scheduler fired late or skipped
    runs. The scanner self-heals (it reads the IST clock and processes
    whatever slot it wakes up in), but bars between the missed slots were
    never scanned - the user should SEE that in the report instead of
    wondering why a signal is missing. prev and now are always the same
    day because state.json is day-scoped."""
    try:
        hh, mm = map(int, str(prev_hhmm).split(":"))
    except ValueError:
        return ""
    prev = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    gap = (now - prev).total_seconds() / 60.0
    if gap <= SWEEP_GAP_WARN_MIN:
        return ""
    missed = int(gap // 10)
    return (f"**Sweep gap detected:** previous sweep {prev_hhmm} IST, this "
            f"one {now:%H:%M} IST - about {missed} x 10-min slot(s) were "
            "skipped or badly delayed by GitHub's scheduler, so bars in "
            "between were never scanned. Isolated delays are normal "
            "(GitHub cron is best-effort); if this appears every day, see "
            "README > 'Scheduled runs not firing on GitHub'.")


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
# Net score >= +2 -> UP, <= -2 -> DOWN, in between -> NEUTRAL. A one-tick
# hysteresis band (a held regime releases only when the score crosses 0)
# stops UP<->NEUTRAL flapping on marginal sweeps.
#
# The regime is ADVISORY ONLY: it is printed in every report header and
# stamped into journal Remarks for audit, but it never blocks a side -
# a weak tape still has strong stocks worth buying and a strong tape
# still has weak stocks worth shorting. Stock-level filters decide, and
# the auto-tune engine mutes whatever keeps losing or slipping.
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
    side: +1 UP, -1 DOWN, 0 NEUTRAL - ADVISORY ONLY, never a gate (a
    weak tape still has strong stocks and vice versa). The resolved
    regime is persisted in state.json with a one-tick hysteresis band
    so marginal sweeps cannot flip it."""
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
            f"{held}  -  advisory only, never blocks a side")
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

def sig_s1(sym, b, vwap, apct):
    """S1: close beyond the 09:15-09:30 opening range with volume. The
    stock's own break decides the side - never the index regime."""
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
    long_ok = c > oh and c > vw and vol >= 2.0 * avg_vol
    short_ok = c < ol and c < vw and vol >= 2.0 * avg_vol
    if long_ok:
        r = risk_math(c, ol, "LONG", apct)
    elif short_ok:
        r = risk_math(c, oh, "SHORT", apct)
    else:
        return None
    return make_signal("S1", sym, r, "ORB break, 2x volume, VWAP aligned")


def sig_s2(sym, b, vwap, ctx, apct):
    """S2: trend day pulls back to VWAP and reclaims it (both sides,
    decided by the stock's own open vs yesterday's close)."""
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
        if not touched:
            return None
        stop = min(float(b["Low"].iloc[-3:].min()), c * (1 - MIN_STOP_PCT))
        r = risk_math(c, stop, "LONG", apct)
    elif open_px <= 0.998 * ctx["prev_close"]:        # trend-DOWN day
        if (b["Close"] < vwap).mean() < 0.7:
            return None
        touched = float(b["High"].iloc[-1]) >= vw and c < vw
        if not touched:
            return None
        stop = max(float(b["High"].iloc[-3:].max()), c * (1 + MIN_STOP_PCT))
        r = risk_math(c, stop, "SHORT", apct)
    else:
        return None
    return make_signal("S2", sym, r, "VWAP pullback rejection on trend day")


def sig_s3(sym, b, vwap, apct):
    """S3: fresh 15-min Supertrend flip with volume and VWAP support.
    Stop = the nearer of the Supertrend line and the flip candle's extreme
    (floored/ceilinged at the 0.3% cost boundary). The stock's own
    Supertrend direction decides the side - never the index regime."""
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
    if d.iloc[-1] == 1 and c > vw:
        nearer = max(float(st["st"].iloc[-1]), float(flip_bar["Low"]))
        stop = min(nearer, c * (1 - MIN_STOP_PCT))
        r = risk_math(c, stop, "LONG", apct)
    elif d.iloc[-1] == -1 and c < vw:
        nearer = min(float(st["st"].iloc[-1]), float(flip_bar["High"]))
        stop = max(nearer, c * (1 + MIN_STOP_PCT))
        r = risk_math(c, stop, "SHORT", apct)
    else:
        return None
    return make_signal("S3", sym, r, "Supertrend(10,3) 15-min flip + volume")


def sig_s4(sym, b, vwap, ctx, apct):
    """S4: NR7 / inside-day expansion breakout through yesterday's range.
    The stock's own breakout decides the side - never the index regime."""
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
    if c > ctx["prev_high"] and c > vw:
        stop = min(bar_lo, c * (1 - MIN_STOP_PCT))
        r = risk_math(c, stop, "LONG", apct)
    elif c < ctx["prev_low"] and c < vw:
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
    if s.get("mode") == "TRACKING":
        tag = "[NEW][TRACKING] " if new else "[TRACKING] "
    elif new:
        tag = f"[NEW][TRADE {s.get('budget', '?')}] "
    else:
        tag = ""
    tail = ""
    if s.get("mode") == "TRACKING" and s.get("mode_why"):
        tail = f" | tracking: {s['mode_why']}"
    return (f"{tag}[{s['strategy']}][{s['key'].split('|')[-1]}] {s['symbol']}  "
            f"@ {s['entry']} | SL {s['stop']} ({s['stop_pct']}%) | "
            f"TGT {s['target']} (5R) | qty {s['qty']} | "
            f"risk Rs {s['risk_rs']} | reward Rs {s['target_rs']}\n"
            f"     setup: {s['why']}{tail}")


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
SLIP_FILE = os.path.join(JOURNAL_DIR, "slippage_log.csv")
DAILY_LOSS_R = 2.0        # circuit breaker: day P/L that trips the breaker
LOSS_STREAK_TRIP = 2      # circuit breaker: consecutive losses that trip it
RR_LABEL = "1:5"
SQUAREOFF_GATE = dt.time(15, 5)   # WRAP keeps grading until 15:05 IST
REMARK_EXIT_LABELS = {"STOP_HIT": "STOP HIT", "TARGET_HIT": "TARGET HIT",
                      "EOD_SQUAREOFF": "AUTO SQUARE-OFF"}

TJ_COLS = ("TradeID", "Date", "SignalTime", "Symbol", "Direction", "Setup",
           "Entry", "StopLoss", "Target", "Qty", "RiskPerShare", "RiskAmt",
           "RewardAmt", "RR", "Status", "Mode", "ExitPrice", "ExitTime",
           "ExitReason", "Result", "PnL_R", "PnL_Amt", "UpdatedAt",
           "Remarks", "Notes")
DS_COLS = ("Date", "Trades", "Tracking", "Wins", "Losses", "Flats", "Open",
           "WinRatePct", "NetR", "NetPnlAmt", "BreakerTripped",
           "DayVerdict", "Notes")
SLIP_COLS = ("Date", "SignalTime", "Symbol", "Direction", "Setup",
             "Trigger", "BestRunPct", "ExpiredAt", "Outcome", "Mode",
             "Notes")
SETUP_NAMES = {"S1": "S1_ORB_BREAKOUT", "S2": "S2_VWAP_PULLBACK",
               "S3": "S3_SUPERTREND_FLIP", "S4": "S4_NR7_EXPANSION"}


def read_csv_rows(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except (OSError, UnicodeDecodeError):
        return []


def normalize_modes(rows):
    """Journal rows written before the Mode column existed were all real
    trades - stamp them TRADE so the cap/breaker math stays correct."""
    for r in rows:
        if not r.get("Mode"):
            r["Mode"] = "TRADE"
    return rows


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
        "Mode": sig.get("mode", "TRADE"),
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
    """Circuit breaker computed from CLOSED TRADE rows only (tracking rows
    are accuracy data, not money, so they never trip the breaker; closed
    rows never change, so the breaker is naturally sticky). Triggers: day
    P/L <= -2R, or 2 consecutive losses. Returns (tripped, note)."""
    todays = sorted([r for r in rows
                     if r.get("Date") == str(day)
                     and r.get("Status") == "CLOSED"
                     and r.get("Mode", "TRADE") == "TRADE"],
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
    """One evolving row per day in daily_summary.csv. Trades = TRADE-mode
    rows only (the real book); TRACKING rows are the accuracy lab and get
    their own column plus a Notes suffix."""
    todays = [r for r in tj_rows if r.get("Date") == str(day)]
    trades = [r for r in todays if r.get("Mode", "TRADE") == "TRADE"]
    tracks = [r for r in todays if r.get("Mode", "TRADE") == "TRACKING"]
    closed = [r for r in trades if r.get("Status") == "CLOSED"]
    wins = sum(1 for r in closed if r.get("Result") == "WIN")
    losses = sum(1 for r in closed if r.get("Result") == "LOSS")
    flats = sum(1 for r in closed if r.get("Result") == "FLAT")
    netr = round(sum(float(r["PnL_R"]) for r in closed), 3)
    netamt = round(sum(float(r["PnL_Amt"]) for r in closed), 2)
    wr = round(100.0 * wins / len(closed), 1) if closed else 0.0
    tripped, note = breaker_state(tj_rows, day)
    tclosed = [r for r in tracks if r.get("Status") == "CLOSED"]
    tw = sum(1 for r in tclosed if r.get("Result") == "WIN")
    tnet = round(sum(float(r["PnL_R"]) for r in tclosed
                     if r.get("PnL_R")), 3)
    notes = note if tripped else ""
    if tracks:
        lab = (f"tracking lab {len(tclosed)} graded ({tw}W/"
               f"{len(tclosed) - tw}L, {_fmt_signed(tnet, 2)}R)")
        notes = f"{notes}; {lab}" if notes else lab
    if not todays and not notes:
        notes = "no qualifying setups today"
    row = {"Date": str(day), "Trades": len(trades), "Tracking": len(tracks),
           "Wins": wins, "Losses": losses, "Flats": flats,
           "Open": len(trades) - len(closed),
           "WinRatePct": f"{wr:.1f}", "NetR": _fmt_signed(netr, 3),
           "NetPnlAmt": _fmt_signed(netamt, 2),
           "BreakerTripped": "YES" if tripped else "NO",
           "DayVerdict": "FAILED" if tripped else "PASSED",
           "Notes": notes}
    for i, r in enumerate(ds_rows):
        if r.get("Date") == str(day):
            ds_rows[i] = row
            break
    else:
        ds_rows.append(row)
    ds_rows.sort(key=lambda r: r.get("Date", ""))


def journal_status(rows, day, tripped, bnote, pending_n=0, slip_n=0):
    """Short journal block appended to every live-scan report. The trade
    budget counts TRADE rows only; tracking rows are shown separately."""
    todays = [r for r in rows if r.get("Date") == str(day)]
    trades = [r for r in todays if r.get("Mode", "TRADE") == "TRADE"]
    tracks = [r for r in todays if r.get("Mode", "TRADE") == "TRACKING"]
    closed = [r for r in trades if r.get("Status") == "CLOSED"]
    open_n = len(trades) - len(closed)
    wins = sum(1 for r in closed if r.get("Result") == "WIN")
    losses = sum(1 for r in closed if r.get("Result") == "LOSS")
    netr = sum(float(r["PnL_R"]) for r in closed if r.get("PnL_R"))
    line = (f"Journal today: {len(trades)}/{CAP_TRADES_DAY} trades | "
            f"tracking {len(tracks)} | open {open_n} | "
            f"closed {len(closed)} ({wins}W/{losses}L) | "
            f"day P/L {_fmt_signed(netr, 3)}R | "
            f"fill gate: {pending_n} pending | slippage {slip_n}")
    if trades_taken_today(rows, day) >= CAP_TRADES_DAY:
        line += " | TRADE CAP: trading stopped for today"
    if tripped:
        line += " | BREAKER: " + bnote
    return [line]


def journal_lines(rows, day, tripped, bnote):
    """Detailed per-row block for the WRAP report. TRADE rows are the real
    book; TRACKING rows are the accuracy lab - both graded, only TRADE
    rows count toward the budget and the breaker."""
    todays = [r for r in rows if r.get("Date") == str(day)]
    if not todays:
        return ["No signals fired today - nothing to journal."]
    trades = [r for r in todays if r.get("Mode", "TRADE") == "TRADE"]
    tracks = [r for r in todays if r.get("Mode", "TRADE") == "TRACKING"]
    out = [f"Journaled today: {len(trades)} trade(s) + {len(tracks)} "
           "tracking row(s) (accuracy lab)."]
    for r in todays:
        mode = r.get("Mode", "TRADE")
        if r.get("Status") == "CLOSED":
            out.append(f"  {r['TradeID']} [{mode}]: entry {r['Entry']} -> "
                       f"exit {r['ExitPrice']} | {r['Result']} {r['PnL_R']}R "
                       f"({r['ExitReason']})")
        else:
            out.append(f"  {r['TradeID']} [{mode}]: entry {r['Entry']} | "
                       "still OPEN")
    closed = [r for r in trades if r.get("Status") == "CLOSED"]
    netr = sum(float(r["PnL_R"]) for r in closed if r.get("PnL_R"))
    tclosed = [r for r in tracks if r.get("Status") == "CLOSED"]
    tnetr = sum(float(r["PnL_R"]) for r in tclosed if r.get("PnL_R"))
    out.append(f"Trade P/L: {_fmt_signed(netr, 3)}R "
               f"({len(closed)}/{CAP_TRADES_DAY} budget used) | tracking "
               f"net {_fmt_signed(tnetr, 3)}R | breaker: "
               + (f"YES ({bnote})" if tripped else "no"))
    return out


# ------------------------------------------------------------------
# Fill-confirmation gate + slippage log, and the auto-tune engine.
#
# A fired signal is NOT a trade yet. It first waits in state.json's
# 'pending' list; on every later sweep check_pending() looks at the
# stock's completed 5-minute bars SINCE the signal:
#   LONG : filled iff some bar's Low <= trigger * (1 + FILL_TOL_PCT%)
#   SHORT: filled iff some bar's High >= trigger * (1 - FILL_TOL_PCT%)
# i.e. price really came back to the signal price (a limit order at the
# trigger would have filled). A fill is journaled with the fill bar as
# SignalTime so grading starts there; Remarks keep the original signal
# time for audit. A signal that never retraces by FILL_DEADLINE (15:00)
# is written to journal/slippage_log.csv with how far the price ran
# away, and never counts as a trade.
#
# autotune_buckets() then reads BOTH files over the last 5 sessions and
# mutes the (strategy, side) buckets that are net-losing or chronically
# slipping - the scanner's own accuracy trend steers what it fires.
# ------------------------------------------------------------------

def save_state(st):
    try:
        with open(STATE_FILE, "w") as fh:
            json.dump(st, fh)
    except OSError:
        pass


def pending_new(sig, day, now):
    """A fired signal awaiting fill confirmation (state.json 'pending')."""
    return {"date": str(day), "time": f"{now:%H:%M}", "sig": sig, "best": 0.0}


def check_pending(pend, bars, day, now):
    """The fill-confirmation gate, run on every LIVE/WRAP sweep BEFORE the
    journal grading. Promotes a pending signal to a journal trade only when
    price retraces to the trigger price (within FILL_TOL_PCT); runaways
    that never retrace by FILL_DEADLINE expire to the slippage log.
    Mutates pend in place; returns (filled_pends, slipped_pends)."""
    tol = FILL_TOL_PCT / 100.0
    filled, slipped, keep = [], [], []
    for p in pend:
        sig = p["sig"]
        e = float(sig["entry"])
        b = bars.get(sig["symbol"] + ".NS")
        touch = None
        if b is not None and len(b):
            try:
                hh, mm = map(int, p["time"].split(":"))
            except ValueError:
                hh, mm = 9, 15
            w = b[[t.time() >= dt.time(hh, mm) for t in b.index]]
            if len(w):
                if sig["target"] > sig["entry"]:            # LONG
                    p["best"] = max(p.get("best", 0.0),
                                    100.0 * (float(w["High"].max()) - e) / e)
                    hit = w[w["Low"] <= e * (1 + tol)]
                else:                                       # SHORT
                    p["best"] = max(p.get("best", 0.0),
                                    100.0 * (e - float(w["Low"].min())) / e)
                    hit = w[w["High"] >= e * (1 - tol)]
                if len(hit):
                    touch = f"{hit.index[0]:%H:%M}"
        if touch is not None:
            p["fill_time"] = touch
            filled.append(p)
        elif now.time() >= FILL_DEADLINE:
            p["expired"] = f"{now:%H:%M}"
            slipped.append(p)
        else:
            keep.append(p)
    pend[:] = keep
    return filled, slipped


def fill_to_row(p, day, now, trade_id):
    """Journal row for a signal that CONFIRMED by retracing to its trigger.
    SignalTime becomes the fill bar time, so grading starts at the fill;
    Remarks narrate both the original signal time and the fill."""
    sig = p["sig"]
    row = signal_to_row(sig, day, now, trade_id)
    ft = p.get("fill_time") or f"{now:%H:%M}"
    row["SignalTime"] = ft
    row["UpdatedAt"] = f"{now:%Y-%m-%d %H:%M:%S}"
    row["Remarks"] = (f"Entry triggered @ {sig['entry']:.2f} at {ft} IST "
                      f"(signal fired {p['time']}; filled on retrace to trigger)")
    if sig.get("regime"):
        row["Remarks"] += f" | {sig['regime']}"
    if row.get("Mode") == "TRACKING":
        why = sig.get("mode_why") or "not tradeable"
        row["Remarks"] += f" | TRACKING ({why}) - accuracy data, do not take"
    return row


def slip_to_row(p, day, now):
    """Slippage-log row: a fired signal that never retraced to its trigger,
    so it was never really takeable at the signal price."""
    sig = p["sig"]
    side = "LONG" if sig["target"] > sig["entry"] else "SHORT"
    return {"Date": str(day), "SignalTime": p["time"],
            "Symbol": sig["symbol"], "Direction": side,
            "Setup": SETUP_NAMES.get(sig["strategy"], sig["strategy"]),
            "Trigger": f"{sig['entry']:.2f}",
            "BestRunPct": f"{p.get('best', 0.0):.2f}",
            "ExpiredAt": p.get("expired", f"{now:%H:%M}"),
            "Outcome": "SLIPPAGE (never retraced to trigger)",
            "Mode": sig.get("mode", "TRADE"),
            "Notes": sig.get("why", "") +
                     (f" | {sig['regime']}" if sig.get("regime") else "")}


def autotune_buckets(tj_rows, slip_rows, day):
    """The auto-correct engine. Over the last AT_WINDOW_DAYS trading
    sessions STRICTLY BEFORE today, grade every (Setup, Direction) bucket
    from the journal (graded fills) and the slippage log (runaways):
      * net R below zero with >= AT_MIN_TRADES fills -> MUTED (losing logic)
      * >= AT_SLIP_MIN runaways and slips > fills    -> MUTED (chronic chase)
      * anything else, including buckets with no history -> ENABLED
    Today's own rows are never judged (no intra-day feedback loop). The
    scanner re-reads the CSVs on every sweep, so the mute list follows the
    dominant recent trend of its own accuracy automatically.
    Returns (muted_keys, report_lines)."""
    today = str(day)
    dates = {r.get("Date", "") for r in list(tj_rows) + list(slip_rows)}
    window = sorted(d for d in dates if d and d < today)[-AT_WINDOW_DAYS:]
    wset = set(window)
    buckets = {}
    for r in tj_rows:
        if r.get("Date") not in wset or r.get("Status") != "CLOSED":
            continue
        k = (r.get("Setup", "?"), r.get("Direction", "?"))
        b = buckets.setdefault(k, {"fills": 0, "wins": 0, "losses": 0,
                                   "netr": 0.0, "slips": 0})
        b["fills"] += 1
        try:
            b["netr"] += float(r.get("PnL_R") or 0.0)
        except (TypeError, ValueError):
            pass
        if r.get("Result") == "WIN":
            b["wins"] += 1
        elif r.get("Result") == "LOSS":
            b["losses"] += 1
    for r in slip_rows:
        if r.get("Date") not in wset:
            continue
        k = (r.get("Setup", "?"), r.get("Direction", "?"))
        b = buckets.setdefault(k, {"fills": 0, "wins": 0, "losses": 0,
                                   "netr": 0.0, "slips": 0})
        b["slips"] += 1
    muted, lines = set(), []
    for k in sorted(buckets):
        b = buckets[k]
        why = ""
        if b["fills"] >= AT_MIN_TRADES and b["netr"] < -1e-9:
            why = "losing net R"
        elif b["slips"] >= AT_SLIP_MIN and b["slips"] > b["fills"]:
            why = "chronic slippage"
        if why:
            muted.add(k)
        lines.append(f"  {k[0]:<19} {k[1]:<5} fills {b['fills']:>2} "
                     f"({b['wins']}W/{b['losses']}L) "
                     f"net {_fmt_signed(b['netr'], 2)}R | slips {b['slips']}"
                     f"  ->  {'MUTED (' + why + ')' if why else 'enabled'}")
    if not lines:
        lines.append("  no history yet - every setup/side enabled by default")
    return muted, lines


def trades_taken_today(rows, day):
    """TRADE-mode journal rows for the day - the real book. Tracking rows
    and slippage never count toward the daily budget."""
    return sum(1 for r in rows
               if r.get("Date") == str(day)
               and r.get("Mode", "TRADE") == "TRADE")


def fire_mode(muted, tripped, committed):
    """Mode of a newly fired signal. TRADE = actionable and counts against
    the daily cap. TRACKING = journaled + graded for accuracy but explicitly
    not to be taken (auto-tune muted bucket / circuit-breaker day / cap
    exhausted). Tracking rows keep the learning loop fed so a muted strategy
    can win its way back into circulation."""
    if muted:
        return "TRACKING", "auto-tune muted bucket"
    if tripped:
        return "TRACKING", "circuit breaker active"
    if committed >= CAP_TRADES_DAY:
        return "TRACKING", f"daily cap {CAP_TRADES_DAY}/{CAP_TRADES_DAY}"
    return "TRADE", ""


def next_session(day):
    """Next weekday after `day` (weekends skipped; NSE holidays ignored)."""
    d = day + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


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

    # ---- journal, slippage log, fill gate, circuit breaker, auto-tune ----
    tj_rows = normalize_modes(read_csv_rows(TJ_FILE))
    backfill_remarks(tj_rows)          # fill Remarks for pre-upgrade rows
    ds_rows = read_csv_rows(DS_FILE)
    slip_rows = normalize_modes(read_csv_rows(SLIP_FILE))
    state = load_state(day)
    delay_line = sweep_delay_note(state.get("last_sweep", ""), now)
    state["last_sweep"] = f"{now:%H:%M}"
    save_state(state)
    pend = state.setdefault("pending", [])
    filled_p, slip_p = [], []
    cap_line = ""
    if ph in ("LIVE", "WRAP"):
        # Fill-confirmation gate BEFORE grading: signals that retraced to
        # their trigger become journal rows (graded from the fill bar on);
        # runaways that expired at 15:00 go to the slippage log instead.
        filled_p, slip_p = check_pending(pend, bars, day, now)
        for p in filled_p:
            sig = p["sig"]
            side = "LONG" if sig["target"] > sig["entry"] else "SHORT"
            tid = unique_trade_id(tj_rows, day, sig["symbol"], side)
            tj_rows.append(fill_to_row(p, day, now, tid))
        for p in slip_p:
            slip_rows.append(slip_to_row(p, day, now))
        if filled_p or slip_p:
            save_state(state)
        if filled_p:
            print(f"fill gate: {len(filled_p)} signal(s) confirmed on retrace")
        if slip_p:
            print(f"fill gate: {len(slip_p)} signal(s) -> slippage log "
                  "(never retraced to trigger)")
        # ---- daily trade cap: the 10th TRADE fill stops trading ----
        if trades_taken_today(tj_rows, day) >= CAP_TRADES_DAY:
            if state.get("cap_notified") != str(day):
                state["cap_notified"] = str(day)
                save_state(state)
                send_telegram(
                    f"DAILY TRADE CAP REACHED ({CAP_TRADES_DAY}/"
                    f"{CAP_TRADES_DAY} trades on {day:%d-%b-%Y}).\n"
                    "STOP TRADING for the rest of the day - protect the "
                    "book; manage open positions normally.\n"
                    "Signals keep firing as TRACKING only until the 13:30 "
                    "IST cutoff to track accuracy - they are journaled but "
                    "must NOT be taken.")
                cap_line = (f"DAILY TRADE CAP REACHED ({CAP_TRADES_DAY}/"
                            f"{CAP_TRADES_DAY} trades) - stop-trading message "
                            "sent. TRACKING rows continue for accuracy only.")
    graded, stale = monitor_journal(tj_rows, bars, daily, day, now,
                                    intra=intra)
    if graded or stale:
        print(f"journal: graded {graded} row(s) today, closed {stale} stale row(s)")
    tripped, bnote = breaker_state(tj_rows, day)

    # ---- auto-tune: mute strategy/side buckets losing or slipping lately ----
    muted, tune_lines = autotune_buckets(tj_rows, slip_rows, day)

    lines = [f"# NSE F&O Scanner - {day:%d %b %Y} {now:%H:%M} IST (phase {ph})",
             "", f"{nifty_tag}  |  universe scanned: {len(bars)} stocks", ""]
    if delay_line:
        lines += [delay_line, ""]

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
        today_slips = [r for r in slip_rows if r.get("Date") == str(day)]
        if today_slips:
            lines += ["", f"Slippage today: {len(today_slips)} fired signal(s) "
                          "never retraced to their trigger price - logged in "
                          "journal/slippage_log.csv, not in the trade journal:"]
            for r in today_slips[:12]:
                lines.append(f"  {r['SignalTime']} {r['Symbol']} "
                             f"{r['Setup'].split('_')[0]} {r['Direction']} "
                             f"@ {r['Trigger']} (ran {r['BestRunPct']}% away)")
        if pend:
            lines += ["", f"Fill gate: {len(pend)} signal(s) still waiting for a "
                          "retrace to trigger (expires 15:00)."]
        if now.time() >= SQUAREOFF_GATE:
            # ---- end-of-day learning: what tomorrow's sweeps will apply ----
            # Window INCLUDES today: this is exactly the table the next
            # session's first LIVE sweep will act on.
            nxt = next_session(day)
            _, eod_lines = autotune_buckets(tj_rows, slip_rows, nxt)
            tr_today = [r for r in tj_rows if r.get("Date") == str(day)
                        and r.get("Mode", "TRADE") == "TRADE"
                        and r.get("Status") == "CLOSED"]
            tk_today = [r for r in tj_rows if r.get("Date") == str(day)
                        and r.get("Mode", "TRADE") == "TRACKING"
                        and r.get("Status") == "CLOSED"]
            twn = sum(1 for r in tr_today if r.get("Result") == "WIN")
            kwn = sum(1 for r in tk_today if r.get("Result") == "WIN")
            slip_today_n = sum(1 for r in slip_rows
                               if r.get("Date") == str(day))
            lines += ["", f"## EOD learning - applied from the next session "
                          f"({nxt:%d %b})", ""]
            lines.append(f"Learned from the last {AT_WINDOW_DAYS} sessions "
                         f"including today: {len(tr_today)} graded trades "
                         f"({twn}W/{len(tr_today) - twn}L) + {len(tk_today)} "
                         f"tracking rows ({kwn}W/{len(tk_today) - kwn}L) + "
                         f"{slip_today_n} slippage rows. Winning buckets keep "
                         "firing; muted buckets fire TRACKING-only (journaled "
                         "for accuracy, not traded) until their results "
                         "recover:")
            lines += eod_lines
        lines += ["", "Square off any open MIS positions now. The broker auto "
                      "square-off follows; never widen a stop."]
    else:
        lines += ["## Strategy signals (live scan)", ""]
        lines += [f"Auto-tune, last {AT_WINDOW_DAYS} sessions (win/loss + "
                  "slippage dominance; muted buckets fire TRACKING-only):"]
        lines += tune_lines
        lines.append("")
        found, tune_blocked = [], []
        for s, b in bars.items():
            f = ctx.get(s)
            if not f:
                continue
            v = vwaps[s]
            for fn, args in ((sig_s1, (b, v, f["apct"])),
                             (sig_s2, (b, v, f, f["apct"])),
                             (sig_s3, (b, v, f["apct"])),
                             (sig_s4, (b, v, f, f["apct"]))):
                try:
                    r = fn(s, *args)
                except Exception:
                    r = None
                if not r:
                    continue
                side = "LONG" if r["target"] > r["entry"] else "SHORT"
                if (SETUP_NAMES.get(r["strategy"], r["strategy"]),
                        side) in muted:
                    # muted bucket: the signal still fires, but as a
                    # TRACKING row - journaled for accuracy so it can win
                    # its way back into the live book
                    r["mode"] = "TRACKING"
                    r["mode_why"] = "auto-tune muted bucket"
                    tune_blocked.append(f"{r['strategy']} {side} {r['symbol']}")
                found.append(r)
        new = [r for r in found if r["key"] not in state["fired"]]
        # daily budget = committed trade slots: TRADE rows already journaled
        # today + TRADE signals still awaiting fill confirmation (a signal
        # that ends up as slippage frees its slot on the next sweep)
        committed = (trades_taken_today(tj_rows, day)
                     + sum(1 for p in pend
                           if p.get("sig", {}).get("mode", "TRADE") == "TRADE"))
        for r in new:
            state["fired"][r["key"]] = f"{now:%H:%M}"
            r["regime"] = (f"NIFTY {REG_NAME[nifty_side]} "
                           f"(score {nifty_score:+d})")
            if r.get("mode") != "TRACKING":
                r["mode"], r["mode_why"] = fire_mode(False, tripped, committed)
            if r["mode"] == "TRADE":
                committed += 1
                r["budget"] = f"{committed}/{CAP_TRADES_DAY}"
            else:
                r["budget"] = f"{trades_taken_today(tj_rows, day)}/{CAP_TRADES_DAY}"

        # EVERY fired signal enters the fill-confirmation gate, whatever its
        # mode: TRADE rows become trades on a retrace; TRACKING rows become
        # accuracy rows on a retrace; runaways go to the slippage log.
        pended = 0
        if tripped:
            lines += [f"CIRCUIT BREAKER ACTIVE ({bnote}). New setups fire as "
                      "TRACKING only - journaled for accuracy; take nothing "
                      "new today; manage open positions normally.", ""]
        for r in new:
            try:
                pend.append(pending_new(r, day, now))
                pended += 1
            except Exception as exc:
                print("fill-gate append failed:", exc)
        save_state(state)

        if tune_blocked:
            lines += [f"Auto-tune demoted {len(tune_blocked)} setup(s) to "
                      "TRACKING on recent losses/slippage: "
                      + ", ".join(tune_blocked[:6])
                      + (" ..." if len(tune_blocked) > 6 else ""), ""]
        if found:
            fresh = {r["key"] for r in new}
            lines += [card(r, r["key"] in fresh) for r in
                      sorted(found, key=lambda r: r["key"])]
        else:
            lines.append("No qualifying setups right now.")
        tr_new = sum(1 for r in new if r.get("mode") == "TRADE")
        lines += ["", f"Signals detected this run: {len(found)}; "
                      f"new today: {len(new)} ({tr_new} TRADE / "
                      f"{len(new) - tr_new} TRACKING); all sent to the "
                      f"fill gate; confirmed on retrace this run: "
                      f"{len(filled_p)}; slippage this run: {len(slip_p)}."]
        today_slip_n = sum(1 for r in slip_rows if r.get("Date") == str(day))
        lines += journal_status(tj_rows, day, tripped, bnote,
                                len(pend), today_slip_n)
        lines += ["", "Execution rules: place a bracket order (entry + SL + 5R target) "
                      "or a market/limit entry plus a separate SL-M order. "
                      f"A signal is journaled only after price retraces to its "
                      f"trigger (within {FILL_TOL_PCT:g}%); runaways are logged "
                      "as slippage and never counted as trades. "
                      f"TRADE rows count toward the daily budget of "
                      f"{CAP_TRADES_DAY}; the {CAP_TRADES_DAY}th TRADE sends "
                      "the stop-trading alert. TRACKING rows (cap / breaker / "
                      "muted) are accuracy data - do not take them. New "
                      "signals stop at 13:30 IST. If neither level hits by "
                      "14:45, square off by 15:05. Never widen a stop. Never "
                      "take a second entry in the same stock on the same "
                      "strategy today."]

    if cap_line:
        lines += ["", cap_line]

    upsert_daily_summary(ds_rows, tj_rows, day)
    write_csv_rows(TJ_FILE, TJ_COLS, tj_rows)
    write_csv_rows(DS_FILE, DS_COLS, ds_rows)
    write_csv_rows(SLIP_FILE, SLIP_COLS, slip_rows)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    write_report(os.path.join(REPORTS_DIR, f"report_{day:%Y-%m-%d}.md"), lines)

    send_telegram("\n".join(lines))
    print("\n".join(lines))
    print(f"\nReport written: {REPORTS_DIR}/report_{day:%Y-%m-%d}.md")


if __name__ == "__main__":
    main()
