#!/usr/bin/env python3
"""
NSE F&O Intraday Scanner - SOP Volume IV (strict 1:5 reward:risk)
=================================================================
Scans the NSE F&O cash universe for four intraday setups and writes a
Markdown report plus a CSV log. Optional Telegram alerts.

Strategies
  S1  Opening Range Breakout (09:15-09:30 range) to a strict 5R target
  S2  VWAP Pullback Continuation (trend-day pullback to VWAP)
  S3  Supertrend (10,3) 15-min flip continuation
  S4  NR7 / Inside-Day range expansion (yesterday's contraction)

This script runs inside GitHub Actions every 10 minutes between 08:30
and 15:20 IST, Monday-Friday. It reads the IST clock itself:

  before 09:20        -> exits silently (nothing to do yet)
  09:20 - 09:29 PREP  -> pre-market watchlist (gaps, NR7/inside-day names)
  09:30 - 14:29 LIVE  -> full strategy scan
  14:30 - 15:20 WRAP  -> square-off reminder
  weekends / holidays -> exits (no fresh market data)

Data source: Yahoo Finance via the free `yfinance` library (no API key).
Optional test hooks (environment variables):
  SCANNER_FORCE_DATE=YYYY-MM-DD   dry-run a past session (evaluated 12:30 IST)
  SCANNER_TEST_LIMIT=N            scan only the first N symbols (debug)
"""

import os
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

IST = ZoneInfo("Asia/Kolkata")   # NSE runs on India Standard Time (UTC+5:30)

STATE_FILE   = "state.json"
REPORTS_DIR  = "reports"

# ------------------------------------------------------------------
# Small clock helpers
# ------------------------------------------------------------------

def now_ist():
    """Current time in IST (or a forced past time for dry-runs)."""
    force = os.environ.get("SCANNER_FORCE_DATE", "")
    if force:
        d = dt.datetime.strptime(force, "%Y-%m-%d").date()
        return dt.datetime(d.year, d.month, d.day, 12, 30, tzinfo=IST)
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
    if dt.time(14, 30) <= t <= dt.time(15, 20):
        return "WRAP"
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
    """Session VWAP series (typical price weighted by volume)."""
    tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
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

    nifty_side = 0
    if nifty is not None:
        nb = completed_bars(nifty, day, now)
        if len(nb):
            nv = vwap_of(nb)
            c = float(nb["Close"].iloc[-1])
            nifty_side = 1 if c > float(nv.iloc[-1]) else -1

    ctx, bars, vwaps = {}, {}, {}
    for s in intra:
        b = completed_bars(intra[s], day, now)
        if len(b) < 6 or float(b["Volume"].sum()) <= 0:
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

    lines = [f"# NSE F&O Scanner - {day:%d %b %Y} {now:%H:%M} IST (phase {ph})",
             "", f"NIFTY intraday side: {'UP' if nifty_side > 0 else 'DOWN' if nifty_side < 0 else 'FLAT/NA'}"
             f"  |  universe scanned: {len(bars)} stocks", ""]

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

        if found:
            fresh = {r["key"] for r in take}
            lines += [card(r, r["key"] in fresh) for r in
                      sorted(found, key=lambda r: r["key"])]
        else:
            lines.append("No qualifying setups right now.")
        lines += ["", f"Signals detected this run: {len(found)}; "
                      f"new today: {len(new)} (reporting cap {MAX_SIGNALS}/run)."]
        lines += ["", "Execution rules: place a bracket order (entry + SL + 5R target) "
                      "or a market/limit entry plus a separate SL-M order. "
                      "If neither level hits by 14:45, square off by 15:05. "
                      "Never widen a stop. Never take a second entry in the same "
                      "stock on the same strategy today."]

    os.makedirs(REPORTS_DIR, exist_ok=True)
    write_report(os.path.join(REPORTS_DIR, f"report_{day:%Y-%m-%d}.md"), lines)

    send_telegram("\n".join(lines))
    print("\n".join(lines))
    print(f"\nReport written: {REPORTS_DIR}/report_{day:%Y-%m-%d}.md")


if __name__ == "__main__":
    main()
