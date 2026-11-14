1:  #!/usr/bin/env python3
   2:  # -*- coding: utf-8 -*-
   3:  """
   4:  VOL5 — Extended-Hours Dual-Frame 1:5 Reward : Risk Intraday Scanner (NSE F&O)
   5:  ================================================================================
   6:  A standalone, self-improving companion system to Vol4 (market-sweep.yml).
   7:  Runs on GitHub Actions free tier. No server, no local machine required.
   8:  
   9:  WHAT CHANGES vs VOL4 (Vol4 untouched — everything here is NEW files only):
  10:    * Signal window extends to 14:30 IST (Vol4 stops new signals at 13:30).
  11:    * Dual timeframe: 15-minute bars set direction/structure, 5-minute bars fire triggers.
  12:    * Self-learning window extended from 5 to 10 sessions (AT_WINDOW_DAYS = 10).
  13:    * Own journals (journal_v5/), own reports (reports_v5/), own memory
  14:      (state_v5.json, tuner_state_v5.json) — zero interference with Vol4.
  15:    * Own GitHub workflow (.github/workflows/extended-scan.yml) in its own
  16:      concurrency group (vol5-extended), so the two scanners never step on each other.
  17:    * Five dual-frame strategies (V1-V5) derived from Vol4 learnings (see SOP).
  18:    * "Tomorrow priority" recommendation list printed at EOD — the system tells
  19:      you which methods earned their slot yesterday, so it self-improves each night.
  20:  
  21:  EDUCATIONAL USE ONLY. NOT INVESTMENT ADVICE. Every signal is an alert; you decide.
  22:  """
  23:  
  24:  from __future__ import annotations
  25:  
  26:  import csv
  27:  import json
  28:  import logging
  29:  import math
  30:  import os
  31:  import sys
  32:  from datetime import datetime, timedelta, timezone
  33:  from pathlib import Path
  34:  
  35:  import numpy as np
  36:  import pandas as pd
  37:  import requests
  38:  from zoneinfo import ZoneInfo
  39:  
  40:  INDIA = ZoneInfo("Asia/Kolkata")
  41:  
  42:  # --------------------------------------------------------------------------- #
  43:  # CONFIGURATION (change these to tune behaviour; every number is explained    #
  44:  # in the SOP_Vol5 document. Capital / risk settings mirror Vol4 for parity.) #
  45:  # --------------------------------------------------------------------------- #
  46:  
  47:  CAPITAL_RUPEES   = 20_000.0      # model capital
  48:  RISK_PCT         = 0.01          # 1% of capital risked per trade
  49:  RISK_AMT         = 200.0         # Rs 200 per trade (= CAPITAL * RISK_PCT)
  50:  RR_MULT          = 5.0           # strict 1 : 5 reward : risk
  51:  MIN_STOP_PCT     = 0.003         # stop floor: 0.3 % (keeps costs from eating edge)
  52:  MAX_STOP_PCT     = 0.015         # stop ceiling: 1.5 % (keeps 5R reachable)
  53:  ATR_PERIOD       = 14
  54:  REACH_FRAC       = 1.0           # 5R target must fit inside the day's ATR(14) range
  55:  MIS_LEVERAGE     = 5             # SEBI MIS intraday leverage ceiling (final qty cap)
  56:  EST_COST_RT      = 60.0          # estimated round-trip cost per trade (Rs)
  57:  
  58:  # --- Trading hours (IST) --------------------------------------------------- #
  59:  PHASE_PREP_START_MIN  = 9 * 60 + 25   # 09:25  pre-market watchlist
  60:  PHASE_LIVE_START_MIN  = 9 * 60 + 30   # 09:30  full scanning starts
  61:  PHASE_CUTOFF_MIN      = 14 * 60 + 30  # 14:30  NEW signals cut off (extended window)
  62:  PHASE_WRAP_END_MIN    = 15 * 60 + 45  # 15:45  journal finalize last chance
  63:  
  64:  # --- Fill gate & lifecycle -------------------------------------------------- #
  65:  FILL_TOL_PCT     = 0.001             # fill confirms when retraced within 0.10 %
  66:  FILL_DEADLINE_MIN = 15 * 60          # unfilled signals expire to slippage log at 15:00
  67:  SQ_OFF_MIN       = 15 * 60 + 5       # force-close leftovers at first sweep >= 15:05
  68:  FINALIZE_TIMES   = (15 * 60 + 10,     # 15:10, 15:20, 15:35, 15:45 repeat passes
  69:                      15 * 60 + 20,
  70:                      15 * 60 + 35,
  71:                      15 * 60 + 45)
  72:  
  73:  # --- Guard rails ------------------------------------------------------------ #
  74:  CAP_TRADES_DAY   = 10                # maximum committed TRADE slots per day
  75:  DAILY_LOSS_R     = -2.0              # circuit-breaker threshold (-2R)
  76:  LOSS_STREAK_TRIP = 2                 # two consecutive losses trips breaker
  77:  SWEEP_GAP_WARN_MIN = 25              # warn if a sweep wakes > 25 min late
  78:  
  79:  # --- Auto-tune (self-improvement) window: 10 SESSIONS ----------------------- #
  80:  AT_WINDOW_DAYS   = 10                # learns from last 10 completed sessions
  81:  AT_MIN_TRADES    = 2                 # mute needs >= 2 graded fills at negative netR
  82:  AT_SLIP_MIN      = 3                 # mute needs >= 3 runaway signals (chronic chase)
  83:  
  84:  # --- Directory layout (ALL under repo root; Vol4 uses journal/, state.json — #
  85:  #     NOTHING here collides) ------------------------------------------------- #
  86:  BASE_DIR        = Path(os.path.dirname(os.path.abspath(__file__)))
  87:  JOURNAL_DIR     = BASE_DIR / "journal_v5"
  88:  REPORTS_DIR     = BASE_DIR / "reports_v5"
  89:  STATE_FILE      = BASE_DIR / "state_v5.json"
  90:  TUNER_STATE_FILE = BASE_DIR / "tuner_state_v5.json"
  91:  
  92:  TEST_OUT_BASE   = Path("/tmp/vol5_test")
  93:  
  94:  # --- Journal schema (kept identical to Vol4 for audit continuity) ---------- #
  95:  JOURNAL_COLS = ["TradeID", "Date", "SignalTime", "Symbol", "Direction", "Setup",
  96:                  "Entry", "StopLoss", "Target", "Qty", "RiskPerShare", "RiskAmt",
  97:                  "RewardAmt", "RR", "Status", "Mode", "ExitPrice", "ExitTime",
  98:                  "ExitReason", "Result", "PnL_R", "PnL_Amt", "UpdatedAt",
  99:                  "Remarks", "Notes"]
 100:  TUNER_COLS   = ["Date", "Bucket", "Decision", "NetR", "Wins", "Losses",
 101:                  "Flats", "Trades", "WinRatePct", "SlipCount", "Reason"]
 102:  
 103:  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 104:  logger = logging.getLogger(__name__)
 105:  
 106:  
 107:  # --------------------------------------------------------------------------- #
 108:  # IST CLOCK & PHASES                                                          #
 109:  # --------------------------------------------------------------------------- #
 110:  def ist_now():
 111:      """Current instant in India Standard Time."""
 112:      return datetime.now(INDIA)
 113:  
 114:  
 115:  def ist_now_min() -> int:
 116:      return ist_now().hour * 60 + ist_now().minute
 117:  
 118:  
 119:  def ist_clock_override() -> int | None:
 120:      """SCANNER_FORCE_TIME=HH:MM overrides the internal clock for dry-runs."""
 121:      raw = os.environ.get("SCANNER_FORCE_TIME")
 122:      if not raw:
 123:          return None
 124:      h, m = map(int, raw.split(":"))
 125:      return h * 60 + m
 126:  
 127:  
 128:  def market_phase(mins: int) -> str:
 129:      if mins < PHASE_LIVE_START_MIN:
 130:          return "PREP" if mins >= PHASE_PREP_START_MIN else "EARLY"
 131:      if mins <= PHASE_CUTOFF_MIN:
 132:          return "LIVE"
 133:      return "WRAP"
 134:  
 135:  
 136:  # --------------------------------------------------------------------------- #
 137:  # YAHOO FINANCE DATA LADDER                                                   #
 138:  # --------------------------------------------------------------------------- #
 139:  def _fetch(ticker: str, interval: str, period: str):
 140:      url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={period}"
 141:      r = requests.get(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=30)
 142:      try:
 143:          data = r.json()
 144:      except Exception as e:
 145:          logger.warning(f"{ticker} fetch failed ({interval}): {e}")
 146:          return None, None
 147:      try:
 148:          res = data["chart"]["result"][0]
 149:          ts = res["timestamp"]
 150:          q = res["indicators"]["quote"][0]
 151:          df = pd.DataFrame({
 152:              "open":  q.get("open",  [np.nan]*len(ts)),
 153:              "high":  q.get("high",   [np.nan]*len(ts)),
 154:              "low":   q.get("low",    [np.nan]*len(ts)),
 155:              "close": q.get("close",  [np.nan]*len(ts)),
 156:              "volume":q.get("volume", [0]*len(ts)),
 157:          }, index=pd.to_datetime(ts, unit="s", utc=True)).dropna(subset=["close"])
 158:          if df.empty:
 159:              return None, None
 160:          df = df.sort_index()
 161:          return df
 162:      except Exception as e:
 163:          logger.warning(f"{ticker} parse failed ({interval}): {e}")
 164:          return None, None
 165:  
 166:  
 167:  def _bars(ticker: str, interval: str, period: str):
 168:      """Return (df) where df has TZ-aware UTC datetime index."""
 169:      df = _fetch(ticker, interval, period)
 170:      if isinstance(df, tuple):          # _fetch returned (None, None) on parse failure
 171:          return pd.DataFrame(columns=["open","high","low","close","volume"])
 172:      if df is None or df.empty:
 173:          return pd.DataFrame(columns=["open","high","low","close","volume"])
 174:      return df.tz_localize(None).tz_localize("UTC")
 175:  
 176:  
 177:  def symbol_tickers():
 178:      """Symbols to scan. Tries universe.UNIVERSE_TICKERS from the existing repo;
 179:      falls back to a hard-coded liquid NSE F&O list so the scanner runs anywhere."""
 180:      try:
 181:          from universe import UNIVERSE_TICKERS  # type: ignore
 182:          tickers = [t.rstrip().upper() for t in UNIVERSE_TICKERS if t.strip()]
 183:      except Exception:
 184:          tickers = []
 185:      if len(tickers) >= 10:
 186:          return sorted(set(tickers))
 187:      fallback = [
 188:          "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HDFC","SBIN","BHARTIARTN",
 189:          "KOTAKBANK","LT","AXISBANK","HCLTECH","ITC","MARUTI","WIPRO","ULTRACEMCO",
 190:          "SUNPHARMA","BAJAJ-FINANCE","HDFCLIFE","NTPC","POWERGRID","ASIANPAINT","TITAN",
 191:          "ZEEL","UPL","JSWSTEEL","ADANIENT","TATASTEEL","TATACONSUM","TATAMOTORS",
 192:          "MARICO","CIPLA","DRREDDY","BRITANNIA","APOLLOHOSP","EICHERMOT","GRASIM",
 193:          "BAJAJFINSV","M&M","HINDUNILVR","NESTLEIND","SHRIRAMFIN","BPCL","COALINDIA",
 194:          "IOC","GAIL","HINDALCO","ADANIPORTS","VEDL","SUDHARMO",
 195:      ]
 196:      return [f"{x}.NS" for x in fallback]
 197:  
 198:  
 199:  def fetch_dual_bars(symbol: str):
 200:      """15-min bars (trend frame, last 20d) + 5-min bars (trigger frame, last 10d)."""
 201:      b15 = _bars(symbol, "15m", "20d")
 202:      b5  = _bars(symbol, "5m",  "10d")
 203:      return b5, b15
 204:  
 205:  
 206:  # --------------------------------------------------------------------------- #
 207:  # SESSION GROUPING (IST session boundaries)                                  #
 208:  # --------------------------------------------------------------------------- #
 209:  def to_ist(df: pd.DataFrame):
 210:      """Index -> timezone-aware IST."""
 211:      out = df.copy()
 212:      out.index = out.index.tz_convert("Asia/Kolkata")
 213:      return out
 214:  
 215:  
 216:  def sessions(df: pd.DataFrame):
 217:      """Split into a dict IST-date -> session dataframe (IST index)."""
 218:      if df.empty:
 219:          return {}
 220:      ist = to_ist(df)
 221:      return {dt: grp.copy() for dt, grp in ist.groupby(ist.index.date)}
 222:  
 223:  
 224:  def today_session(bars):
 225:      """The session that belongs to today (may be partially filled)."""
 226:      sess = sessions(bars)
 227:      return sess.get(ist_now().date()), sess
 228:  
 229:  
 230:  # --------------------------------------------------------------------------- #
 231:  # INDICATORS (vectorised; session-aware where needed)                          #
 232:  # --------------------------------------------------------------------------- #
 233:  def typical_price(o, h, l, c):
 234:      return (h + l + c) / 3.0
 235:  
 236:  
 237:  def vwap_session(df):
 238:      """Session VWAP (reset on each IST date); uses expanding mean if volume missing."""
 239:      tp = typical_price(df["open"], df["high"], df["low"], df["close"])
 240:      vol = df["volume"].astype(float).replace(0, np.nan)
 241:      if vol.isna().all():
 242:          return tp.expanding().mean()
 243:      cv = vol.cumsum()
 244:      w = tp * vol
 245:      return (w.cumsum() / cv.replace(0, np.nan))
 246:  
 247:  
 248:  def sma(df, n):
 249:      return df["close"].rolling(n).mean()
 250:  
 251:  
 252:  def stdma(df, n):
 253:      return df["close"].rolling(n).std(ddof=1)
 254:  
 255:  
 256:  def true_range(o, h, l, prev_close):
 257:      return np.maximum.reduce([h - l, (h - prev_close).abs(), (l - prev_close).abs()])
 258:  
 259:  
 260:  def atr(df, n):
 261:      tr = true_range(df["open"], df["high"], df["low"], df["close"].shift(1))
 262:      s = pd.Series(tr)
 263:      return s.ewm(span=n, adjust=False).mean().values
 264:  
 265:  
 266:  def rsi(df, n=14):
 267:      delta = df["close"].diff()
 268:      up = delta.clip(lower=0).astype(float)
 269:      down = (-delta.clip(upper=0)).astype(float)
 270:      ups = up.ewm(span=n, adjust=False).mean()
 271:      downs = down.ewm(span=n, adjust=False).mean()
 272:      rs = ups / downs.replace(0, np.nan)
 273:      return 100 - 100 / (1 + rs)
 274:  
 275:  
 276:  def di(df, n=14):
 277:      plus_dm, minus_dm = np.zeros(len(df.index)), np.zeros(len(df.index))
 278:      hp = df["high"].values; lp = df["low"].values
 279:      for i in range(1, len(df)):
 280:          plus_dm[i] = (hp[i] - hp[i-1]) if ((hp[i]-hp[i-1]) > abs(lp[i]-lp[i-1])) and (hp[i]-hp[i-1]) > 0 else 0.0
 281:          minus_dm[i] = (lp[i-1]-lp[i]) if ((lp[i-1]-lp[i]) > (hp[i]-hp[i-1])) and (lp[i-1]-lp[i]) > 0 else 0.0
 282:      atrv = atr(df, n)
 283:      plus_di = 100 * (pd.Series(plus_dm).ewm(span=n, adjust=False).mean() / pd.Series(atrv).replace(0, np.nan))
 284:      minus_di = 100 * (pd.Series(minus_dm).ewm(span=n, adjust=False).mean() / pd.Series(atrv).replace(0, np.nan))
 285:      return plus_di, minus_di
 286:  
 287:  
 288:  def adx(df, n=14):
 289:      pdi, mdi = di(df, n)
 290:      dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
 291:      return dx.ewm(span=n, adjust=False).mean()
 292:  
 293:  
 294:  def supertrend(high, low, close, period=10, mult=3.0):
 295:      """Classic Supertrend returning (upper, lower, trend_direction) arrays."""
 296:      high, low, close = np.asarray(high), np.asarray(low), np.asarray(close)
 297:      n = len(close)
 298:      if n < period:
 299:          return np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
 300:      d = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
 301:      atrv = pd.Series(atr(d, period)).bfill().fillna(0).values
 302:      hl2 = (high + low) / 2.0
 303:      up = hl2 + mult * atrv
 304:      lo = hl2 - mult * atrv
 305:      upper, lower, trend = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
 306:      upper[0], lower[0] = up[0], lo[0]
 307:      trend[0] = 1
 308:      for i in range(1, n):
 309:          if trend[i-1] == 1:
 310:              upper[i] = min(up[i], upper[i-1]); lower[i] = max(lo[i], lower[i-1])
 311:          else:
 312:              upper[i] = up[i]; lower[i] = lo[i]
 313:          if trend[i-1] == 1:
 314:              trend[i] = 1 if low[i] < lower[i] else (-1 if high[i] > upper[i] else trend[i-1])
 315:          else:
 316:              trend[i] = -1 if high[i] > upper[i] else (1 if low[i] < lower[i] else trend[i-1])
 317:      return upper, lower, trend
 318:  
 319:  
 320:  def bollinger(df, n=20, k=2.0):
 321:      m = sma(df, n); s = stdma(df, n)
 322:      return m, m + k * s, m - k * s
 323:  
 324:  
 325:  # --------------------------------------------------------------------------- #
 326:  # STRUCTURAL BUILDING BLOCKS                                                  #
 327:  # --------------------------------------------------------------------------- #
 328:  def or_range(session_df, nbars=2):
 329:      """Opening-range high/low from first `nbars` bars of the session."""
 330:      d = session_df.iloc[:nbars]
 331:      if d.empty:
 332:          return np.nan, np.nan, np.nan
 333:      return float(d["high"].max()), float(d["low"].min()), float(d["open"].iloc[0])
 334:  
 335:  
 336:  def inside_bar(series):
 337:      """Boolean: each bar is inside the previous bar (15m frame)."""
 338:      return (series["high"] <= series["high"].shift(1)) & (series["low"] >= series["low"].shift(1))
 339:  
 340:  
 341:  def atr_ratio(series, n=5):
 342:      """ATR of this bar relative to median ATR of prior n bars (contraction flag)."""
 343:      return atr(series, n) / atr(series, n).shift(n + 1).bfill()
 344:  
 345:  
 346:  def squeeze_bars(df15, threshold=0.75):
 347:      """15m bars flagged as contractions (inside-bar OR ATR-ratio <= threshold)."""
 348:      ib = inside_bar(df15).astype(int)
 349:      ar = (atr_ratio(df15, 5) <= threshold).astype(int)
 350:      return ib | ar
 351:  
 352:  
 353:  # --------------------------------------------------------------------------- #
 354:  # THE 1 : 5 RISK MODEL                                                        #
 355:  # --------------------------------------------------------------------------- #
 356:  def build_position(direction, entry, ctx):
 357:      """Size a position around Rs 200 risk, strict 1:5, within stop-band + ATR."""
 358:      long_side = direction == "LONG"
 359:      entry = float(entry)
 360:      if not entry or np.isnan(entry) or entry <= 0:
 361:          return None
 362:  
 363:      struct = ctx.get("struct") or entry * 0.01
 364:      if long_side:
 365:          stop = round(entry - max(struct, entry * MIN_STOP_PCT), 2)
 366:      else:
 367:          stop = round(entry + max(struct, entry * MIN_STOP_PCT), 2)
 368:  
 369:      dist = abs(entry - stop)
 370:      stop_pct = dist / entry
 371:      if stop_pct < MIN_STOP_PCT or stop_pct > MAX_STOP_PCT:
 372:          return None
 373:  
 374:      reward = RR_MULT * dist
 375:      target = round(entry + reward if long_side else entry - reward, 2)
 376:  
 377:      daily_atr = ctx.get("atr") or np.nan
 378:      if daily_atr and not np.isnan(daily_atr) and reward > daily_atr * REACH_FRAC:
 379:          return None
 380:  
 381:      raw_qty = RISK_AMT / dist
 382:      if raw_qty <= 0:
 383:          return None
 384:      qty = int(math.floor(raw_qty))
 385:      margin_cap = int(math.floor(CAPITAL_RUPEES * MIS_LEVERAGE / entry))
 386:      qty = min(qty, margin_cap)
 387:      if qty <= 0:
 388:          return None
 389:  
 390:      return {
 391:          "entry": entry, "stop": stop, "target": target, "qty": qty,
 392:          "risk_per_share": round(dist, 2), "risk_amt": round(dist * qty, 2),
 393:          "reward_amt": round(reward * qty, 2), "rr": round(reward / dist, 2),
 394:      }
 395:  
 396:  
 397:  # --------------------------------------------------------------------------- #
 398:  # FRAME PREPARATION                                                           #
 399:  # --------------------------------------------------------------------------- #
 400:  def prepare_frames(m5, b15):
 401:      """Attach indicators to both frames; returns dicts keyed 'm5'/'b15'."""
 402:      frames = {}
 403:      for key, df in (("m5", m5), ("b15", b15)):
 404:          out = df.copy()
 405:          out["_vwap"] = vwap_session(out)
 406:          out["_rsi"] = rsi(out)
 407:          out["_atr"] = atr(out, 14)
 408:          out["_bb_mid"], out["_bb_up"], out["_bb_lo"] = bollinger(out)
 409:          if key == "m5":
 410:              u, l, t = supertrend(out["high"].bfill(), out["low"].bfill(), out["close"], period=7, mult=2.0)
 411:          else:
 412:              u, l, t = supertrend(out["high"].bfill(), out["low"].bfill(), out["close"], period=10, mult=3.0)
 413:          out["_st_up"], out["_st_lo"], out["_st_dir"] = u, l, t
 414:          out["_adx"] = adx(out)
 415:          frames[key] = out
 416:      return frames
 417:  
 418:  
 419:  def conf_score(ctx):
 420:      """0-1 confidence from confluence features (used by tuner gating)."""
 421:      weights = {"vol_mult": 0.20, "vwap_align": 0.15, "st_confluence": 0.25,
 422:                 "trend_slope": 0.15, "bb_extreme": 0.10, "adx_strong": 0.15}
 423:      score = 0.0
 424:      if ctx.get("vol_mult", 1.0) >= 1.8: score += weights["vol_mult"]
 425:      elif ctx.get("vol_mult", 1.0) >= 1.4: score += weights["vol_mult"] / 1.8
 426:      if ctx.get("vwap_aligned"): score += weights["vwap_align"]
 427:      if ctx.get("st_confl"): score += weights["st_confluence"]
 428:      if ctx.get("trend_rising") or ctx.get("trend_falling"): score += weights["trend_slope"]
 429:      if ctx.get("bb_at_extreme"): score += weights["bb_extreme"]
 430:      if ctx.get("adx", 0) > 25: score += weights["adx_strong"]
 431:      return min(score, 1.0)
 432:  
 433:  
 434:  def safe_dir(x):
 435:      """Cast supertrend direction to int safely."""
 436:      try:
 437:          v = float(x)
 438:          return 1 if v > 0 else (-1 if v < 0 else 0)
 439:      except Exception:
 440:          return 0
 441:  
 442:  
 443:  def make_signal(setup, direction, entry, ctx, note):
 444:      """Build a canonical signal record."""
 445:      pos = build_position(direction, entry, ctx)
 446:      if pos is None or pos["qty"] <= 0:
 447:          return []
 448:      adxv = float(ctx.get("adx", 0)) if isinstance(ctx.get("adx"), (int, float)) else 0
 449:      return [{
 450:          "setup": setup, "direction": direction, "entry": entry, "stop": pos["stop"],
 451:          "target": pos["target"], "qty": pos["qty"], "risk_per_share": pos["risk_per_share"],
 452:          "risk_amt": pos["risk_amt"], "reward_amt": pos["reward_amt"], "rr": pos["rr"],
 453:          "conf": conf_score(ctx), "note": note,
 454:          "vol_mult": float(ctx.get("vol_mult", 1.0)), "vwap_aligned": bool(ctx.get("vwap_aligned")),
 455:          "st_confl": bool(ctx.get("st_confl")), "trend_rising": bool(ctx.get("trend_rising")),
 456:          "trend_falling": bool(ctx.get("trend_falling")), "adx": adxv,
 457:      }]
 458:  
 459:  
 460:  # --------------------------------------------------------------------------- #
 461:  # STRATEGIES V1-V5 (dual-frame, side decided by the STOCK, not the index)     #
 462:  # --------------------------------------------------------------------------- #
 463:  def V1_15MIN_ORB_BREAKOUT(m5, b15, today):
 464:      """
 465:      Vol4 S1 lesson, upgraded: 15-minute opening range + 5-minute confirmation
 466:      with volume & VWAP confluence. Side = stock's own breakout.
 467:      Runs 09:35 - 14:25 (needs at least 2 ORB bars + 1 confirm).
 468:      """
 469:      ohi, oli, _ = or_range(today, 2)
 470:      if np.isnan(ohi) or np.isnan(oli) or len(m5) < 20:
 471:          return []
 472:      st15 = b15.copy()
 473:      st15["_dir"] = st15["_st_dir"].fillna(0).apply(safe_dir)
 474:      vw = float(today["_vwap"].iloc[-1]) if len(today) else np.nan
 475:      vol_avg = float(m5["volume"].rolling(20).mean().iloc[-1]) if len(m5) >= 20 else 0
 476:      last = m5.iloc[-1]
 477:      vm = last["volume"] / max(vol_avg, 1)
 478:      outs = []
 479:      for direction, rg in (("LONG", ohi), ("SHORT", oli)):
 480:          ok_close = (last["close"] > rg * 1.001) if direction == "LONG" else (last["close"] < rg * 0.999)
 481:          vwap_aligned = (vw > 0) and ((direction == "LONG") == (last["close"] > vw))
 482:          st_confl = safe_dir(st15["_dir"].iloc[-1]) != 0
 483:          cond = ok_close and vwap_aligned and st_confl and vm >= 1.5
 484:          if cond:
 485:              outs += make_signal("V1_15MIN_ORB_BREAKOUT", direction, last["close"], {
 486:                  "struct": ohi - oli, "atr": float(b15["_atr"].iloc[-1]),
 487:                  "vwap_aligned": True, "st_confl": True, "vol_mult": vm,
 488:                  "trend_rising": True if direction == "LONG" else False, "adx": float(st15["_adx"].iloc[-1])},
 489:                  f"OR-BREAK {direction} of 15m range [{oli:.0f},{ohi:.0f}] vol {vm:.1f}x")
 490:      return outs
 491:  
 492:  
 493:  def V2_15M_SUPERTREND_MOMENTUM(m5, b15, today):
 494:      """
 495:      Vol4 S3 reloaded: 15-min Supertrend flip + 5-min pullback-to-ST-level entry
 496:      + ADX momentum filter. Wider stop logic than Vol4 so it fires more often
 497:      while respecting the 0.3%-1.5% band and the ATR reachability rule.
 498:      Runs until 14:25.
 499:      """
 500:      if len(b15) < 40:
 501:          return []
 502:      st15 = b15.copy()
 503:      st15["_dir"] = st15["_st_dir"].fillna(0).apply(safe_dir)
 504:      adx15 = float(st15["_adx"].iloc[-1]) if "_adx" in st15.columns else 0
 505:      st_dir = safe_dir(st15["_dir"].iloc[-1])
 506:      if st_dir == 0 or adx15 < 20:
 507:          return []
 508:      st_level = float(st15["_st_lo"].iloc[-1]) if st_dir == 1 else float(st15["_st_up"].iloc[-1])
 509:      if np.isnan(st_level):
 510:          return []
 511:  
 512:      hist = m5[m5.index <= m5.index[-2]]
 513:      direction = "LONG" if st_dir == 1 else "SHORT"
 514:      for _, row in hist.iterrows():
 515:          inside = (st_dir == 1 and row["low"] <= st_level * 1.002) or (st_dir == -1 and row["high"] >= st_level * 0.998)
 516:          resume = (st_dir == 1 and row["close"] > row["open"]) or (st_dir == -1 and row["close"] < row["open"])
 517:          if inside and resume:
 518:              return make_signal("V2_15M_SUPERTREND_MOMENTUM", direction, row["close"], {
 519:                  "struct": 3.0, "atr": float(b15["_atr"].iloc[-1]), "st_confl": True},
 520:                  f"15M-ST {direction} pullback to {st_level:.1f}")
 521:      return []
 522:  
 523:  
 524:  def V3_VWAP_TREND_PULLBACK(m5, b15, today):
 525:      """
 526:      Vol4 S2 lesson upgraded: only takes VWAP pullbacks WITH 15-min trend bias
 527:      (Supertrend + rising/falling VWAP slope), rejecting counter-trend noise.
 528:      Runs until 14:25.
 529:      """
 530:      if len(b15) < 30 or len(today) < 5:
 531:          return []
 532:      st15 = b15.copy()
 533:      st15["_dir"] = st15["_st_dir"].fillna(0).apply(safe_dir)
 534:      vw = float(today["_vwap"].iloc[-1]) if len(today) else np.nan
 535:      if np.isnan(vw):
 536:          return []
 537:      vwslope = float(today["_vwap"].iloc[-1] - today["_vwap"].iloc[-5])
 538:      st_dir = safe_dir(st15["_dir"].iloc[-1])
 539:      trend_up = (st_dir == 1 and vwslope > 0) or (st_dir == -1 and vwslope < 0)
 540:      if st_dir == 0 or abs(vwslope) < 0.0001 or not trend_up:
 541:          return []
 542:  
 543:      hist = m5[m5.index <= m5.index[-2]]
 544:      direction = "LONG" if st_dir == 1 else "SHORT"
 545:      for _, row in hist.tail(30).iterrows():
 546:          near_vwap = abs(row["close"] - vw) / vw < 0.0015
 547:          rejects = ((st_dir == 1 and row["close"] >= vw and row["close"] >= row["open"]) or
 548:                     (st_dir == -1 and row["close"] <= vw and row["close"] <= row["open"]))
 549:          if near_vwap and rejects:
 550:              return make_signal("V3_VWAP_TREND_PULLBACK", direction, row["close"], {
 551:                  "struct": abs(vw - row["low"]), "atr": float(b15["_atr"].iloc[-1]),
 552:                  "vwap_aligned": True}, f"15M-trend pullback into VWAP {vw:.1f}")
 553:      return []
 554:  
 555:  
 556:  def V4_INSIDE_DAY_EXPANSION(m5, b15, today):
 557:      """
 558:      Vol4 S4 lesson upgraded: 15-min inside-bar/narrow-range contraction
 559:      resolving in the direction of 15-min trend, with volume surge.
 560:      Runs 09:50 - 14:25.
 561:      """
 562:      if len(b15) < 25:
 563:          return []
 564:      st15 = b15.copy()
 565:      st15["_dir"] = st15["_st_dir"].fillna(0).apply(safe_dir)
 566:      st_dir = safe_dir(st15["_dir"].iloc[-1])
 567:      if st_dir == 0:
 568:          return []
 569:      vol_avg = float(m5["volume"].rolling(20).mean().iloc[-1]) if len(m5) >= 20 else 0
 570:      direction = "LONG" if st_dir == 1 else "SHORT"
 571:  
 572:      def scan(days_back=3):
 573:          cand = m5[(m5.index.date >= (ist_now().date() - timedelta(days=days_back))) & (m5.index != m5.index[-1])]
 574:          hits = []
 575:          prev_h, prev_l = st15.iloc[-2]["high"], st15.iloc[-2]["low"]
 576:          for _, row in cand.tail(25).iterrows():
 577:              break_out = (st_dir == 1 and row["close"] > prev_h) or (st_dir == -1 and row["close"] < prev_l)
 578:              vols = row["volume"] / max(vol_avg, 1)
 579:              if break_out and vols >= 1.8:
 580:                  hits.append((row, vols))
 581:          return hits[:1]
 582:  
 583:      hits = scan()
 584:      if not hits:
 585:          return []
 586:      row, vols = hits[0]
 587:      return make_signal("V4_INSIDE_DAY_EXPANSION", direction, row["close"], {
 588:          "struct": 2.0, "atr": float(b15["_atr"].iloc[-1]), "vol_mult": vols, "st_confl": True},
 589:          f"inside-expansion {direction} vol {vols:.1f}x")
 590:  
 591:  
 592:  def V5_OPENING_RANGE_TAIL_BREAK(m5, b15, today):
 593:      """
 594:      New Vol5 strategy: tail-break of the 09:30-10:00 15-min block in the
 595:      direction of 15-min trend. Captures continuation days after the initial
 596:      ORB fight settles (fills well in afternoon hours).
 597:      Runs 10:15 - 14:25.
 598:      """
 599:      if today is None or len(today) < 4:
 600:          return []
 601:      tail_hi, tail_lo, _ = or_range(today.iloc[-4:], 4)
 602:      st15 = b15.copy()
 603:      st15["_dir"] = st15["_st_dir"].fillna(0).apply(safe_dir)
 604:      st_dir = safe_dir(st15["_dir"].iloc[-1])
 605:      if st_dir == 0:
 606:          return []
 607:      vw15 = float(b15["_vwap"].iloc[-1]) if len(b15) else np.nan
 608:      last5 = m5.iloc[-1]
 609:      vol_avg = float(m5["volume"].rolling(20).mean().iloc[-1]) if len(m5) >= 20 else 0
 610:      vm = last5["volume"] / max(vol_avg, 1)
 611:      direction = "LONG" if st_dir == 1 else "SHORT"
 612:      cond_long = st_dir == 1 and last5["close"] > tail_hi * 1.002 and vm >= 1.6 and vw15 > 0 and last5["close"] > vw15
 613:      cond_short = st_dir == -1 and last5["close"] < tail_lo * 0.998 and vm >= 1.6 and vw15 > 0 and last5["close"] < vw15
 614:      if not (cond_long or cond_short):
 615:          return []
 616:      ctx = {"struct": tail_hi - tail_lo, "atr": float(b15["_atr"].iloc[-1]),
 617:             "vol_mult": vm, "st_confl": True,
 618:             "trend_rising": cond_long, "trend_falling": cond_short}
 619:      return make_signal("V5_OPENING_RANGE_TAIL_BREAK", direction, last5["close"], ctx,
 620:                         f"tail-break {direction} 10:00 block [{tail_lo:.0f},{tail_hi:.0f}]")
 621:  
 622:  
 623:  STRATEGY_FUNCS = {
 624:      "V1_15MIN_ORB_BREAKOUT": V1_15MIN_ORB_BREAKOUT,
 625:      "V2_15M_SUPERTREND_MOMENTUM": V2_15M_SUPERTREND_MOMENTUM,
 626:      "V3_VWAP_TREND_PULLBACK": V3_VWAP_TREND_PULLBACK,
 627:      "V4_INSIDE_DAY_EXPANSION": V4_INSIDE_DAY_EXPANSION,
 628:      "V5_OPENING_RANGE_TAIL_BREAK": V5_OPENING_RANGE_TAIL_BREAK,
 629:  }
 630:  # --------------------------------------------------------------------------- #
 631:  # NIFTY REGIME ENGINE (six-vote, advisory only — never blocks a side)          #
 632:  # Mirrors Vol4's rebuilt regime design; votes recomputed every sweep.          #
 633:  # --------------------------------------------------------------------------- #
 634:  def nifty_regime():
 635:      """Six independent votes from NIFTY 5-min bars; score >= +2 UP, <= -2 DOWN."""
 636:      b5 = _bars("^NSEI", "5m", "7d")
 637:      if b5 is None or b5.empty or len(b5) < 60:
 638:          return {"label": "NEUTRAL", "score": 0, "votes": {},
 639:                  "note": "insufficient index bars"}
 640:      df = b5.copy()
 641:      df["_tp"] = typical_price(df["open"], df["high"], df["low"], df["close"])
 642:      vol = df["volume"].astype(float)
 643:      if vol.replace(0, np.nan).isna().all():
 644:          df["_vwap"] = df["_tp"].expanding().mean()
 645:      else:
 646:          cv = vol.replace(0, np.nan).cumsum()
 647:          df["_vwap"] = (df["_tp"] * vol).cumsum() / cv.replace(0, np.nan)
 648:      open_price = float(df["open"].iloc[0])
 649:      high_of_day = float(df["high"].max())
 650:      low_of_day = float(df["low"].min())
 651:      close = float(df["close"].iloc[-1])
 652:      vwap = float(df["_vwap"].iloc[-1])
 653:      votes = {}
 654:      votes["vwap"] = 1 if close > vwap else (-1 if close < vwap else 0)
 655:      votes["open"] = 1 if close > open_price else (-1 if close < open_price else 0)
 656:      if len(df) >= 12:
 657:          h_lag = float(df["high"].shift(6).iloc[-1])
 658:          l_lag = float(df["low"].shift(6).iloc[-1])
 659:          if close > h_lag and float(df["high"].iloc[-1]) > h_lag:
 660:              votes["struct"] = 1
 661:          elif close < l_lag and float(df["low"].iloc[-1]) < l_lag:
 662:              votes["struct"] = -1
 663:          else:
 664:              votes["struct"] = 0
 665:      else:
 666:          votes["struct"] = 0
 667:      if len(df) >= 12:
 668:          roc = close / float(df["close"].iloc[-12]) - 1
 669:          votes["momo"] = 1 if roc > 0.0012 else (-1 if roc < -0.0012 else 0)
 670:      else:
 671:          votes["momo"] = 0
 672:      move = (high_of_day - open_price) / open_price
 673:      if abs(move) >= 0.002:
 674:          recovered = (close - open_price) / move if move != 0 else 0
 675:          votes["retrace"] = 1 if recovered >= 0.5 else 0
 676:      else:
 677:          votes["retrace"] = 0
 678:      vwaps = df["_vwap"].tail(12).values
 679:      closes = df["close"].tail(12).values
 680:      reclaim = 0
 681:      for pv, c in zip(vwaps[:-1], closes[1:]):
 682:          if ((pv < vwaps[-2]) and (c >= vwaps[-1])) or ((pv > vwaps[-2]) and (c <= vwaps[-1])):
 683:              reclaim = 1 if c >= vwaps[-1] else -1
 684:              break
 685:      votes["reclaim"] = reclaim if reclaim else 0
 686:      score = sum(votes.values())
 687:      stored = _state_load().get("regime", {}).get("label", "NEUTRAL")
 688:      label = "UP" if score >= 2 else ("DOWN" if score <= -2 else "NEUTRAL")
 689:      if stored in ("UP", "DOWN") and score == 0:
 690:          label = stored
 691:      state = {"label": label, "score": score, "votes": {k: str(v) for k, v in votes.items()},
 692:               "note": f"{len(df)} bars | range {low_of_day:.0f}-{high_of_day:.0f}"}
 693:      _state_store("regime", state)
 694:      return state
 695:  
 696:  
 697:  # --------------------------------------------------------------------------- #
 698:  # STATE & TUNER MEMORY                                                        #
 699:  # --------------------------------------------------------------------------- #
 700:  def _sig_ts(s):
 701:      """Serialize signal dict datetime fields to UTC-ISO strings."""
 702:      out = dict(s)
 703:      for k in ("signal_time", "filled_at"):
 704:          if k in out and isinstance(out[k], datetime):
 705:              out[k] = out[k].astimezone(timezone.utc).isoformat()
 706:      return out
 707:  
 708:  
 709:  def _sig_dt(s):
 710:      """Restore UTC datetime fields in loaded signal dicts."""
 711:      for k in ("signal_time", "filled_at"):
 712:          if k in s and isinstance(s[k], str):
 713:              try:
 714:                  s[k] = datetime.fromisoformat(s[k]).replace(tzinfo=timezone.utc)
 715:              except Exception:
 716:                  pass
 717:      return s
 718:  
 719:  
 720:  def _state_load():
 721:      try:
 722:          return json.loads((STATE_FILE.read_text() if STATE_FILE.exists() else "{}"))
 723:      except Exception:
 724:          return {}
 725:  
 726:  
 727:  def _state_store(key, val):
 728:      d = _state_load()
 729:      d[key] = val
 730:      STATE_FILE.write_text(json.dumps(d, indent=1))
 731:  
 732:  
 733:  def _tuner_load():
 734:      try:
 735:          return json.loads((TUNER_STATE_FILE.read_text() if TUNER_STATE_FILE.exists() else "{}"))
 736:      except Exception:
 737:          return {}
 738:  
 739:  
 740:  def _tuner_store(d):
 741:      TUNER_STATE_FILE.write_text(json.dumps(d, indent=1))
 742:  
 743:  
 744:  def _today_date_str():
 745:      return ist_now().date().isoformat()
 746:  
 747:  
 748:  # --------------------------------------------------------------------------- #
 749:  # AUTO-TUNE ENGINE (learns from the LAST 10 SESSIONS)                         #
 750:  # --------------------------------------------------------------------------- #
 751:  def load_journal():
 752:      """Full historical trade journal (all sessions) for grading buckets."""
 753:      rows = []
 754:      p = JOURNAL_DIR / "trade_journal.csv"
 755:      if not p.exists():
 756:          return rows
 757:      with p.open(newline="") as f:
 758:          for r in csv.DictReader(f):
 759:              rows.append(r)
 760:      return rows
 761:  
 762:  
 763:  def session_cutoff_rows(rows):
 764:      """Rows strictly before today (closed sessions)."""
 765:      today = _today_date_str()
 766:      return [r for r in rows if r.get("Date", "") < today]
 767:  
 768:  
 769:  def grade_buckets(rows):
 770:      """Per-bucket stats over the closed-session history."""
 771:      buckets = {}
 772:      for r in rows:
 773:          key = f"{r['Setup']}_{r['Direction']}"
 774:          b = buckets.setdefault(key, {"wins": 0, "losses": 0, "flats": 0,
 775:                                       "trades": 0, "netR": 0.0, "slip": 0})
 776:          b["trades"] += 1
 777:          res = r.get("Result", "")
 778:          if res == "WIN": b["wins"] += 1
 779:          elif res == "LOSS": b["losses"] += 1
 780:          elif res == "FLAT": b["flats"] += 1
 781:          try: b["netR"] += float(r.get("PnL_R") or 0)
 782:          except ValueError: pass
 783:      out = {}
 784:      for k, b in buckets.items():
 785:          tot = b["wins"] + b["losses"]
 786:          out[k] = dict(b, win_rate=(b["wins"] / tot if tot else 0.0),
 787:                        netR=round(b["netR"], 4), tot_grades=tot)
 788:      return out
 789:  
 790:  
 791:  def run_auto_tune():
 792:      """Mute losers / chronic chases; re-enable improving buckets. Log to tuning_log."""
 793:      today = _today_date_str()
 794:      rows = load_journal()
 795:      bucket_stats = grade_buckets(session_cutoff_rows(rows))
 796:      for sr in list(csv.DictReader((JOURNAL_DIR / "slippage_log.csv").open(newline=""))
 797:                     if (JOURNAL_DIR / "slippage_log.csv").exists() else []):
 798:          key = f"{sr['Setup']}_{sr['Direction']}"
 799:          if key in bucket_stats:
 800:              bucket_stats[key]["slip"] += 1
 801:      tuner = _tuner_load()
 802:      decisions = []
 803:      active_prev = set(tuner.get("buckets", {}).keys())
 804:      for key, st in sorted(bucket_stats.items()):
 805:          prev_status = tuner.get("buckets", {}).get(key, {}).get("status", "enabled")
 806:          muted_loss = st["netR"] < 0 and st["tot_grades"] >= AT_MIN_TRADES
 807:          chronic_chase = (st["slip"] >= AT_SLIP_MIN) and (st["slip"] > st["trades"])
 808:          do_mute = muted_loss or chronic_chase
 809:          new_status = "muted" if do_mute else "enabled"
 810:          if new_status != prev_status:
 811:              reason = "losing logic" if muted_loss else ("chronic chase" if chronic_chase else "")
 812:              rec = {"Date": today, "Bucket": key, "Decision": new_status,
 813:                     "NetR": st["netR"], "Wins": st["wins"], "Losses": st["losses"],
 814:                     "Flats": st["flats"], "Trades": st["trades"],
 815:                     "WinRatePct": round(st["win_rate"] * 100, 1),
 816:                     "SlipCount": st["slip"], "Reason": reason}
 817:              decisions.append(rec)
 818:              logger.info(f"TUNER {key}: {new_status} ({reason or 'no trigger'}) netR={st['netR']:.2f} WR={st['win_rate']*100:.0f}%")
 819:          tuner.setdefault("buckets", {})[key] = {"status": new_status, "scored_at": today}
 820:      for key in list(active_prev):
 821:          if key not in tuner["buckets"]:
 822:              tuner["buckets"][key] = {"status": "enabled", "scored_at": today}
 823:      if decisions:
 824:          _log_tuning(decisions)
 825:      _tuner_store(tuner)
 826:      return decisions
 827:  
 828:  
 829:  def _log_tuning(records):
 830:      p = JOURNAL_DIR / "tuning_log.csv"
 831:      first = not p.exists()
 832:      with p.open("a", newline="") as f:
 833:          w = csv.DictWriter(f, fieldnames=TUNER_COLS)
 834:          if first:
 835:              w.writeheader()
 836:          w.writerows(records)
 837:  
 838:  
 839:  def ranked_buckets():
 840:      """Order of strategies that earned their slot — printed as tomorrow priority."""
 841:      rows = load_journal()
 842:      stats = grade_buckets(session_cutoff_rows(rows))
 843:      enabled = [(k, s) for k, s in stats.items()
 844:                 if _tuner_load().get("buckets", {}).get(k, {}).get("status") == "enabled"]
 845:      enabled.sort(key=lambda x: (x[1]["netR"], x[1]["win_rate"]), reverse=True)
 846:      return enabled[:6]
 847:  
 848:  
 849:  # --------------------------------------------------------------------------- #
 850:  # SESSION TREND CLASSIFIER (over last 10 SESSIONS)                            #
 851:  # --------------------------------------------------------------------------- #
 852:  def classify_session_trend(days=10):
 853:      """Bullish / bearish / range based on recent daily closes vs opens + VWAP."""
 854:      ticks = ["RELIANCE.NS", "INFY.NS", "ICICIBANK.NS", "HDFCBANK.NS", "SBIN.NS",
 855:               "TCS.NS", "TATASTEEL.NS", "ITC.NS", "WIPRO.NS", "BHARTIARTN.NS"]
 856:      score = 0
 857:      for t in ticks:
 858:          df = _bars(t, "1d", f"{days+5}d")
 859:          if df is None or df.empty or len(df) < days:
 860:              continue
 861:          sess = sessions(df)
 862:          for dt, s in list(sess.items())[:days]:
 863:              o = float(s["open"].iloc[0]); c = float(s["close"].iloc[-1])
 864:              v = float(s["_vwap"].iloc[-1]) if "_vwap" in s.columns else float("nan")
 865:              if not (np.isnan(c) or np.isnan(o) or np.isnan(v)):
 866:                  if c > o and c > v: score += 1
 867:                  elif c < o and c < v: score -= 1
 868:      if score > days * 0.55: return "BULLISH"
 869:      if score < -days * 0.55: return "BEARISH"
 870:      return "RANGE-BOUND"
 871:  
 872:  
 873:  # --------------------------------------------------------------------------- #
 874:  # SIGNAL DISPATCHER: cap, breaker, mode resolution                            #
 875:  # --------------------------------------------------------------------------- #
 876:  def day_ledger(rows):
 877:      """Current trading-day ledger: graded counts, slots used, cap/breaker state."""
 878:      today = _today_date_str()
 879:      day_rows = [r for r in rows if r.get("Date", "") == today]
 880:      graded = [r for r in day_rows if r.get("Status") == "CLOSED"]
 881:      open_rows = [r for r in day_rows if r.get("Status") == "OPEN"]
 882:      wins = losses = flat = streak = 0
 883:      net_r = 0.0
 884:      for r in graded:
 885:          res = r.get("Result", "")
 886:          net_r += float(r.get("PnL_R") or 0)
 887:          if res == "WIN": wins += 1
 888:          elif res == "LOSS": losses += 1; streak += 1
 889:          elif res == "FLAT": flat += 1; streak = 0
 890:      tripped = (net_r <= DAILY_LOSS_R) or (streak >= LOSS_STREAK_TRIP)
 891:      return {"wins": wins, "losses": losses, "flats": flat, "netR": round(net_r, 4),
 892:              "tripped": tripped, "open": open_rows, "committed": sum(1 for r in day_rows if r.get("Mode") == "TRADE")}
 893:  
 894:  
 895:  def mode_for(bucket, ledger, slots_free):
 896:      """Decide TRADE vs TRACKING for a newly fired signal."""
 897:      tuner = _tuner_load()
 898:      if tuner.get("buckets", {}).get(bucket, {}).get("status") == "muted":
 899:          return "TRACKING", "auto-tuned (negative 10-session netR)"
 900:      if slots_free <= 0 or ledger["tripped"]:
 901:          return "TRACKING", "circuit breaker tripped" if ledger["tripped"] else "daily trade cap reached"
 902:      return "TRADE", ""
 903:  
 904:  
 905:  # --------------------------------------------------------------------------- #
 906:  # FILL GATE                                                                   #
 907:  # --------------------------------------------------------------------------- #
 908:  def check_fills(symbol, pending, bars_m5):
 909:      """Mark pending LONG fills when a bar low touches trigger (tol) and vice versa."""
 910:      filled, kept = [], pending[:]
 911:      now = ist_now().astimezone(timezone.utc)
 912:      bars = bars_m5[bars_m5.index <= now]
 913:      for sig in pending:
 914:          keep = True
 915:          st = _sig_dt(dict(sig))
 916:          target_t = float(st["trigger"])
 917:          best_run = 0.0
 918:          if st["direction"] == "LONG":
 919:              ok = any(float(b["low"]) <= target_t * (1 + FILL_TOL_PCT) for _, b in bars.iterrows())
 920:              if len(bars):
 921:                  best_run = max(((float(b["high"]) - target_t) / target_t) for _, b in bars.iterrows())
 922:          else:
 923:              ok = any(float(b["high"]) >= target_t * (1 - FILL_TOL_PCT) for _, b in bars.iterrows())
 924:              if len(bars):
 925:                  best_run = max(((target_t - float(b["low"])) / target_t) for _, b in bars.iterrows())
 926:          if ok:
 927:              filled.append({k: st.get(k) for k in list(st) + ["filled_at", "best_run_pct"]} |
 928:                            {"filled_at": now, "best_run_pct": round(best_run * 100, 2)})
 929:              keep = False
 930:      return filled, [s for s in kept if s not in filled]
 931:  
 932:  
 933:  # --------------------------------------------------------------------------- #
 934:  # JOURNAL WRITING                                                             #
 935:  # --------------------------------------------------------------------------- #
 936:  def next_trade_id(today):
 937:      base = 0
 938:      p = JOURNAL_DIR / "trade_journal.csv"
 939:      if p.exists():
 940:          with p.open() as f:
 941:              try:
 942:                  ids = sorted(list(set(r["TradeID"].split("-")[3] for r in csv.DictReader(f)
 943:                                        if r.get("TradeID", "").startswith("V5-" + today))))
 944:                  if ids:
 945:                      base = int(ids[-1])
 946:              except Exception:
 947:                  base = 0
 948:      return f"V5-{today}-{base+1:05d}"
 949:  
 950:  
 951:  def append_journal(rows_new):
 952:      p = JOURNAL_DIR / "trade_journal.csv"
 953:      first = not p.exists()
 954:      with p.open("a", newline="") as f:
 955:          w = csv.DictWriter(f, fieldnames=JOURNAL_COLS)
 956:          if first:
 957:              w.writeheader()
 958:          w.writerows(rows_new)
 959:  
 960:  
 961:  def slippage_append(rows):
 962:      p = JOURNAL_DIR / "slippage_log.csv"
 963:      first = not p.exists()
 964:      cols = ["Date", "SignalTime", "Symbol", "Direction", "Setup", "Trigger",
 965:              "BestRunPct", "ExpiredAt", "Outcome", "Mode", "Notes"]
 966:      with p.open("a", newline="") as f:
 967:          w = csv.DictWriter(f, fieldnames=cols)
 968:          if first:
 969:              w.writeheader()
 970:          w.writerows(rows)
 971:  
 972:  
 973:  def pnl_from_row(row):
 974:      """Compute PnL_R / PnL_Amt given entry/stop/target/exit, direction, qty."""
 975:      try:
 976:          entry = float(row["Entry"]); stop = float(row["StopLoss"]); qty = int(float(row["Qty"]))
 977:          exitp = float(row["ExitPrice"]); direction = row["Direction"]
 978:      except Exception:
 979:          return 0.0, 0.0
 980:      dist = abs(entry - stop)
 981:      r = ((exitp - entry) / dist) if direction == "LONG" else ((entry - exitp) / dist)
 982:      return round(r, 3), round(r * dist * qty, 2)
 983:  
 984:  
 985:  def write_daily_summary(ledger, tracking_graded=0, notes=""):
 986:      today = _today_date_str()
 987:      p = JOURNAL_DIR / "daily_summary.csv"
 988:      rows = []
 989:      if p.exists():
 990:          rows = list(csv.DictReader(p.open()))
 991:      row = None
 992:      for r in rows:
 993:          if r.get("Date") == today:
 994:              row = r; break
 995:      if row is None:
 996:          row = {c: "" for c in ["Date","Trades","Tracking","Wins","Losses","Flats","Open","WinRatePct","NetR","NetPnlAmt","BreakerTripped","DayVerdict","Notes"]}
 997:          rows.append(row)
 998:      totals = {"Trades": ledger["wins"] + ledger["losses"] + ledger["flats"],
 999:                "Tracking": tracking_graded, "Wins": ledger["wins"],
1000:                "Losses": ledger["losses"], "Flats": ledger["flats"],
1001:                "Open": 0, "WinRatePct": 0.0, "NetR": ledger["netR"], "NetPnlAmt": 0.0,
1002:                "BreakerTripped": "YES" if ledger["tripped"] else "NO",
1003:                "DayVerdict": "FAILED" if ledger["tripped"] else "OK",
1004:                "Date": today, "Notes": notes}
1005:      row.update(totals)
1006:      with p.open("w", newline="") as f:
1007:          w = csv.DictWriter(f, fieldnames=["Date","Trades","Tracking","Wins","Losses","Flats","Open","WinRatePct","NetR","NetPnlAmt","BreakerTripped","DayVerdict","Notes"])
1008:          w.writeheader()
1009:          w.writerow(row)
1010:      return row
1011:  
1012:  
1013:  # --------------------------------------------------------------------------- #
1014:  # REPORT GENERATION                                                           #
1015:  # --------------------------------------------------------------------------- #
1016:  def render_card(sig, committed, mode):
1017:      tag = "TRACKING" if mode == "TRACKING" else "NEW"
1018:      card = (f"*   [{tag}][{'TRADE' if mode=='TRADE' else 'TRACKING'} {committed}/{CAP_TRADES_DAY}] "
1019:              f"{sig['setup']} {sig['direction']}\n")
1020:      card += (f"       Entry {sig['entry']:.2f} · Stop {sig['stop']:.2f} · Target {sig['target']:.2f} · Qty {sig['qty']} "
1021:               f"(Rs {sig['risk_amt']:.0f} risk / Rs {sig['reward_amt']:.0f} target)\n")
1022:      card += f"       @ {sig['signal_time'].strftime('%H:%M')} IST · Conf {sig['conf']:.2f} · *{sig['note']}*\n"
1023:      return card
1024:  
1025:  
1026:  def write_report(phase, regime, bucket_table, cards, recommendations, trend, notes=""):
1027:      lines = [
1028:          "# VOL5 — Extended-Hours Dual-Frame Intraday Watchlist",
1029:          "**NSE F&O · 1 : 5 Reward : Risk · Dual timeframes (15-min trend / 5-min trigger)**",
1030:          "*Generated by scanner_v5.py on GitHub Actions · Educational use only — signals are alerts, not commands.*",
1031:          "", "## Session snapshot",
1032:          "| Item | Value |", "|---|---|",
1033:          f"| Date (IST) | {_today_date_str()} |", f"| Phase | {phase} |",
1034:          f"| IST Clock | {ist_now().strftime('%H:%M IST')} |",
1035:          f"| Regime | **{regime['label']}** (score {regime['score']}) — {json.dumps(regime['votes'])} |",
1036:          "| Trade budget | 10 committed TRADE slots per day |",
1037:          f"| Sessions learned (auto-tune) | last {AT_WINDOW_DAYS} completed sessions |",
1038:          "", "## Signal cards"]
1039:      for c in cards:
1040:          lines.append(c)
1041:      lines += ["\n## Method accuracy over last 10 sessions",
1042:                "| Bucket | Status | NetR | Win Rate | Trades | Reason |",
1043:                "|---|---|---|---|---|---|"]
1044:      for name, status, netr, wr, tr, reason in bucket_table:
1045:          lines.append(f"| {name} | `{status}` | {netr:+.2f} | {wr:.0f}% | {tr} | {reason} |")
1046:      lines += ["\n### EOD learning — recommended method priority for tomorrow",
1047:                "(methods that proved profitable are listed first; the auto-tune will favour them.)"]
1048:      if recommendations:
1049:          for i, (bk, st) in enumerate(recommendations, 1):
1050:              lines.append(f"{i}. **{bk}** — netR {st['netR']:+.2f}, win rate {st['win_rate']*100:.0f}% ({st['tot_grades']} graded)")
1051:      else:
1052:          lines.append("- No graded history yet this session — watchlist starts neutral.")
1053:      lines += [f"\n*Tape bias over last {AT_WINDOW_DAYS} sessions: **{trend}**. Regime is advisory only; the stock's own breakout decides every side.{notes}*"]
1054:      REPORTS_DIR.mkdir(parents=True, exist_ok=True)
1055:      (REPORTS_DIR / f"{_today_date_str()}_vol5_report.md").write_text("\n".join(lines))
1056:  
1057:  
1058:  # --------------------------------------------------------------------------- #
1059:  # TELEGRAM ALERTS (optional)                                                  #
1060:  # --------------------------------------------------------------------------- #
1061:  TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
1062:  TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID")
1063:  
1064:  
1065:  def tg_send(text):
1066:      if not TG_TOKEN or not TG_CHAT:
1067:          return
1068:      try:
1069:          url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage?chat_id={TG_CHAT}&text={requests.utils.quote(text)}"
1070:          requests.get(url, timeout=15)
1071:      except Exception as e:
1072:          logger.warning(f"Telegram failed: {e}")
1073:  
1074:  
1075:  # --------------------------------------------------------------------------- #
1076:  # UNIT TEST BATTERY (SCANNER_TEST_MODE=1) — no network needed                  #
1077:  # --------------------------------------------------------------------------- #
1078:  def run_tests():
1079:      """Deterministic smoke tests covering strategy math, guard rails, tune rules."""
1080:      fails = []
1081:      rng = np.random.default_rng(7)
1082:  
1083:      def chk(name, cond):
1084:          print(("PASS" if cond else "FAIL"), "-", name)
1085:          if not cond:
1086:              fails.append(name)
1087:  
1088:      # Test 1: supertrend produces finite arrays on realistic data
1089:      close = rng.normal(2500, 15, 100); high = close + rng.uniform(1, 8, 100); low = close - rng.uniform(1, 8, 100)
1090:      up, lo, tr = supertrend(high, low, close)
1091:      chk("supertrend: shape + sign values", up.shape == lo.shape == tr.shape == (100,) and set(np.unique(tr)).issubset({-1, 0, 1}))
1092:  
1093:      # Test 2: build_position enforces the 0.3%-1.5% stop band + ATR reachability
1094:      ctx = {"struct": 2.0, "atr": 60.0}
1095:      pos = build_position("LONG", 2500.0, ctx)
1096:      chk("build_position: base case returns qty", pos is not None and pos["qty"] > 0)
1097:      chk("build_position: stop within band", MIN_STOP_PCT <= abs(pos["entry"]-pos["stop"])/pos["entry"] <= MAX_STOP_PCT)
1098:      chk("build_position: strict 5R target", abs(pos["target"]-pos["entry"]) == RR_MULT * abs(pos["entry"]-pos["stop"]))
1099:      chk("build_position: rejects >1.5% stops", build_position("LONG", 2500.0, {"struct": 45.0, "atr": 30.0}) is None)
1100:      chk("build_position: rejects <0.3% stops", build_position("LONG", 2500.0, {"struct": 0.005, "atr": 30.0}) is None)
1101:      chk("build_position: rejects unreachable 5R (small ATR)", build_position("LONG", 2500.0, {"struct": 2.0, "atr": 1.5}) is None)
1102:      chk("build_position: risk amount ~ Rs 200", pos is not None and abs(pos["risk_amt"] - RISK_AMT) < 10)
1103:  
1104:      # Test 3: V1 strategy produces valid signals only after ORB exists
1105:      rng2 = np.random.default_rng(3)
1106:      t0 = pd.Timestamp("2026-11-13 09:15:00", tz="UTC")
1107:      idx = pd.date_range(t0, periods=60, freq="5min")
1108:      close5 = pd.Series(rng2.uniform(200, 210, 60), index=idx)
1109:      open5 = close5 * rng2.uniform(0.99, 1.01, 60); high5 = close5 * rng2.uniform(1.0, 1.005, 60)
1110:      low5 = close5 * rng2.uniform(0.995, 1.0, 60); vol5 = pd.Series(rng2.integers(1000, 5000, 60), index=idx)
1111:      m5 = pd.DataFrame({"open":open5,"high":high5,"low":low5,"close":close5,"volume":vol5})
1112:      m15 = pd.DataFrame({"open":open5.iloc[::3],"high":high5.iloc[::3],"low":low5.iloc[::3],
1113:                          "close":close5.iloc[::3],"volume":vol5.iloc[::3]}).reset_index(drop=True)
1114:      m15["_vwap"] = vwap_session(m15); m15["_atr"] = atr(m15, 14)
1115:      _, _, dirs = supertrend(m15["high"], m15["low"], m15["close"]); m15["_st_dir"] = dirs
1116:      m15["_adx"] = adx(m15)
1117:      today = m5[m5.index.date == m5.index[-1].date]
1118:      signals = V1_15MIN_ORB_BREAKOUT(m5, m15, today)
1119:      chk("V1: returns well-formed signals", all(s["qty"] > 0 and s["setup"] == "V1_15MIN_ORB_BREAKOUT" for s in signals))
1120:  
1121:      # Test 4: auto-tune mutes negative-netR buckets & re-enables improved ones
1122:      JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
1123:      test_rows = [{"TradeID": f"t{i}", "Date": "2026-11-01" if i % 2 == 0 else "2026-11-05",
1124:                    "Symbol": "RELIANCE.NS", "Direction": "LONG", "Setup": "V1_15MIN_ORB_BREAKOUT",
1125:                    "Entry": 2000, "StopLoss": 1995, "Target": 2025, "Qty": 40, "RiskPerShare": 5,
1126:                    "RiskAmt": 200, "RewardAmt": 1000, "RR": 5.0, "Status": "CLOSED", "Mode": "TRADE",
1127:                    "ExitPrice": 1990, "ExitTime": "10:00", "ExitReason": "STOP_HIT",
1128:                    "Result": "LOSS" if i < 5 else "WIN", "PnL_R": -5.0 if i < 5 else 5.0,
1129:                    "PnL_Amt": -1000 if i < 5 else 1000, "UpdatedAt": "2026-11-05T10:00",
1130:                    "Remarks": "", "Notes": ""} for i in range(8)]
1131:      with (JOURNAL_DIR / "trade_journal.csv").open("w", newline="") as f:
1132:          w = csv.DictWriter(f, fieldnames=JOURNAL_COLS); w.writeheader(); w.writerows(test_rows)
1133:      run_auto_tune()
1134:      chk("tune: losing bucket muted", _tuner_load().get("buckets", {}).get("V1_15MIN_ORB_BREAKOUT_LONG", {}).get("status") == "muted")
1135:      extra = [{"TradeID": f"t9", "Date": "2026-11-07", "Symbol": "RELIANCE.NS", "Direction": "LONG",
1136:                "Setup": "V1_15MIN_ORB_BREAKOUT", "Entry": 2000, "StopLoss": 1995, "Target": 2025,
1137:                "Qty": 40, "RiskPerShare": 5, "RiskAmt": 200, "RewardAmt": 1000, "RR": 5.0,
1138:                "Status": "CLOSED", "Mode": "TRADE", "ExitPrice": 2025, "ExitTime": "11:00",
1139:                "ExitReason": "TARGET_HIT", "Result": "WIN", "PnL_R": 5.0, "PnL_Amt": 1000,
1140:                "UpdatedAt": "2026-11-07T11:00", "Remarks": "", "Notes": ""},
1141:               {"TradeID": "t10", "Date": "2026-11-07", "Symbol": "RELIANCE.NS", "Direction": "LONG",
1142:                "Setup": "V1_15MIN_ORB_BREAKOUT", "Entry": 2000, "StopLoss": 1995, "Target": 2025,
1143:                "Qty": 40, "RiskPerShare": 5, "RiskAmt": 200, "RewardAmt": 1000, "RR": 5.0,
1144:                "Status": "CLOSED", "Mode": "TRADE", "ExitPrice": 2025, "ExitTime": "11:00",
1145:                "ExitReason": "TARGET_HIT", "Result": "WIN", "PnL_R": 5.0, "PnL_Amt": 1000,
1146:                "UpdatedAt": "2026-11-07T11:00", "Remarks": "", "Notes": ""}]
1147:      with (JOURNAL_DIR / "trade_journal.csv").open("a", newline="") as f:
1148:          w = csv.DictWriter(f, fieldnames=JOURNAL_COLS); w.writerows(extra)
1149:      run_auto_tune()
1150:      chk("tune: improved bucket re-enabled", _tuner_load().get("buckets", {}).get("V1_15MIN_ORB_BREAKOUT_LONG", {}).get("status") == "enabled")
1151:  
1152:      # Test 5: mode_for enforces cap + breaker
1153:      ledger_cap = {"wins": 0, "losses": 0, "flats": 0, "netR": 0.0, "tripped": False,
1154:                    "open": [], "committed": CAP_TRADES_DAY}
1155:      chk("mode_for: cap => TRACKING", mode_for("X_LONG", ledger_cap, 0)[0] == "TRACKING")
1156:      ledger_brk = {"wins": 0, "losses": 2, "flats": 0, "netR": -2.0, "tripped": True,
1157:                    "open": [], "committed": 5}
1158:      chk("mode_for: breaker => TRACKING", mode_for("X_LONG", ledger_brk, 3)[0] == "TRACKING")
1159:      chk("mode_for: healthy => TRADE", mode_for("X_LONG", {"wins":1,"losses":0,"flats":0,"netR":2.0,"tripped":False,"open":[],"committed":0}, 8)[0] == "TRADE")
1160:  
1161:      # Test 6: regime vote bounds
1162:      reg = nifty_regime()
1163:      chk("regime: bounded score", -6 <= reg["score"] <= 6 and reg["label"] in ("UP", "NEUTRAL", "DOWN"))
1164:  
1165:      if fails:
1166:          print(f"\n{len(fails)} TEST(S) FAILED:", fails)
1167:          return 1
1168:      print("\nALL TESTS PASSED")
1169:      return 0
1170:  
1171:  
1172:  # --------------------------------------------------------------------------- #
1173:  # MAIN SCANNER                                                                 #
1174:  # --------------------------------------------------------------------------- #
1175:  def main():
1176:      mode = os.environ.get("RUN_MODE", "live")
1177:      force_date = os.environ.get("SCANNER_FORCE_DATE")
1178:      override_min = ist_clock_override()
1179:      min_now = override_min if override_min is not None else ist_now_min()
1180:  
1181:          # ensure output directories exist (Vol4-compatible repo-root layout);
1182:      # test runs redirect writes into an ephemeral staging area instead of the repo.
1183:      JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
1184:      REPORTS_DIR.mkdir(parents=True, exist_ok=True)
1185:      if os.environ.get("RUN_MODE") == "test":
1186:          import shutil
1187:          if TEST_OUT_BASE.exists(): shutil.rmtree(TEST_OUT_BASE)
1188:          (TEST_OUT_BASE / "journal_v5").mkdir(parents=True, exist_ok=True)
1189:          (TEST_OUT_BASE / "reports_v5").mkdir(parents=True, exist_ok=True)
1190:          import sys
1191:          sys.modules[__name__].__dict__["JOURNAL_DIR"] = TEST_OUT_BASE / "journal_v5"
1192:          sys.modules[__name__].__dict__["REPORTS_DIR"] = TEST_OUT_BASE / "reports_v5"
1193:  
1194:      if os.environ.get("SCANNER_TEST_MODE") == "1":
1195:          return run_tests()
1196:  
1197:      finalize_note = ""
1198:      phase = market_phase(min_now)
1199:      logger.info(f"VOL5 scan starting — phase={phase} ist_clock={ist_now().strftime('%H:%M')} "
1200:                  f"test_mode={mode} date_force={force_date}")
1201:  
1202:      if phase == "EARLY":
1203:          return  # silent exit before watchlist
1204:  
1205:      if phase == "PREP":
1206:          logger.info("PREP — pre-market watchlist built; full scan starts at 09:30 IST")
1207:          return
1208:  
1209:      universe = symbol_tickers()
1210:      rows = load_journal()
1211:      ledger = day_ledger(rows)
1212:      tuner = _tuner_load()
1213:      pendings = [_sig_dt(s) for s in _state_load().get("pending", [])]
1214:      slots_free = CAP_TRADES_DAY - ledger["committed"]
1215:  
1216:      regime = nifty_regime()
1217:  
1218:      # sweep-gap warning
1219:      last = _state_load().get("last_sweep")
1220:      if last:
1221:          try:
1222:              prev = datetime.fromisoformat(last.replace("Z", "+00:00")).astimezone(INDIA)
1223:              gap = (ist_now() - prev).total_seconds() / 60
1224:              if gap > SWEEP_GAP_WARN_MIN:
1225:                  warn = f"Sweep gap detected: previous sweep {prev.strftime('%H:%M')} IST, this one {ist_now().strftime('%H:%M')} IST - about {int(gap/10)} skipped slot(s). Isolated delays are normal (GitHub cron is best-effort)."
1226:                  logger.warning(warn); tg_send(warn)
1227:          except Exception:
1228:              pass
1229:  
1230:      # strategy scan
1231:      fired_today = []
1232:      seen_keys = set()
1233:      for ticker in universe:
1234:          if os.environ.get("SCANNER_TEST_LIMIT"):
1235:              if len(fired_today) + len(pendings) >= int(os.environ["SCANNER_TEST_LIMIT"]):
1236:                  break
1237:          m5, b15 = fetch_dual_bars(ticker)
1238:          today_sess, _ = today_session(m5)
1239:          if today_sess is None:
1240:              continue  # no today bars yet (early LIVE sweeps wait for the open)
1241:          frames = prepare_frames(m5, b15)
1242:          m5f, b15f = frames["m5"], frames["b15"]
1243:          for name, func in STRATEGY_FUNCS.items():
1244:              t = today_sess.copy(); t["_vwap"] = vwap_session(t)
1245:              raw = func(m5f, b15f, t)
1246:              for sig in raw:
1247:                  sig["symbol"] = ticker
1248:                  sig["signal_time"] = ist_now()
1249:                  sig["bucket"] = f"{sig['setup']}_{sig['direction']}"
1250:                  sig["trigger"] = sig["entry"]
1251:                  bkt = sig["bucket"]
1252:                  key = (sig["setup"], sig["direction"], sig["entry"], sig["stop"], sig["target"], sig["signal_time"].isoformat())
1253:                  if key in seen_keys:
1254:                      continue
1255:                  seen_keys.add(key)
1256:                  mode_chosen, reason = mode_for(bkt, ledger, slots_free)
1257:                  sig["mode"] = mode_chosen
1258:                  sig["reason"] = reason
1259:                  fired_today.append(sig)
1260:                  if mode_chosen == "TRADE":
1261:                      slots_free -= 1
1262:                      ledger["committed"] += 1
1263:                  card = render_card(sig, ledger["committed"], mode_chosen)
1264:                  logger.info(card.strip())
1265:                  if mode_chosen == "TRADE" and slots_free == 0:
1266:                      tg_send(f"DAILY TRADE CAP REACHED ({CAP_TRADES_DAY}/{CAP_TRADES_DAY} trades on {_today_date_str()}). STOP TRADING for the rest of the day - manage open positions normally; later signals TRACKING only.")
1267:                      _state_store("cap_notified", True)
1268:                  tg_send(card.strip())
1269:  
1270:      # WRAP phase: grading, fills, slippage expiry, square-off, finalize
1271:      if phase == "WRAP":
1272:          ledger = day_ledger(load_journal())
1273:          pendings = [_sig_dt(s) for s in _state_load().get("pending", [])]
1274:          wrap_now = ist_now().astimezone(timezone.utc)
1275:          today_1500_ist = (datetime.combine(ist_now().date(), datetime.min.time())
1276:                            + timedelta(minutes=FILL_DEADLINE_MIN))
1277:          fill_deadline_utc = today_1500_ist.replace(tzinfo=INDIA).astimezone(timezone.utc)
1278:  
1279:          # slippage expiry: unfilled signals that never retraced by 15:00
1280:          for ticker in universe[:min(30, len(universe))]:
1281:              m5, b15 = fetch_dual_bars(ticker)
1282:              m5f = prepare_frames(m5, b15)["m5"]
1283:              until = min(fill_deadline_utc, m5f.index[-1] if not m5f.empty else wrap_now)
1284:              bars_before = m5f[m5f.index <= until]
1285:              kept = []
1286:              for sig in pendings:
1287:                  st = _sig_dt(dict(sig))
1288:                  if st["symbol"] != ticker:
1289:                      kept.append(sig); continue
1290:                  st_time = st["signal_time"]
1291:                  if st_time > fill_deadline_utc:
1292:                      kept.append(sig); continue
1293:                  target_t = float(st["trigger"])
1294:                  best_run = 0.0
1295:                  if st["direction"] == "LONG":
1296:                      ok = any(float(b["low"]) <= target_t * (1 + FILL_TOL_PCT) for _, b in bars_before.iterrows())
1297:                      if len(bars_before): best_run = max(((float(b["high"]) - target_t) / target_t) for _, b in bars_before.iterrows())
1298:                  else:
1299:                      ok = any(float(b["high"]) >= target_t * (1 - FILL_TOL_PCT) for _, b in bars_before.iterrows())
1300:                      if len(bars_before): best_run = max(((target_t - float(b["low"])) / target_t) for _, b in bars_before.iterrows())
1301:                  if not ok:
1302:                      slippage_append([{
1303:                          "Date": _today_date_str(), "SignalTime": st_time.strftime("%H:%M"),
1304:                          "Symbol": ticker, "Direction": st["direction"], "Setup": st["setup"],
1305:                          "Trigger": target_t, "BestRunPct": round(best_run * 100, 2),
1306:                          "ExpiredAt": fill_deadline.strftime("%H:%M"), "Outcome": "RUNAWAY",
1307:                          "Mode": st["mode"], "Notes": f"never retraced trigger; ran {'up' if st['direction']=='LONG' else 'down'} {best_run*100:.1f}%"}])
1308:                      if st["mode"] == "TRADE":
1309:                          slots_free += 1; ledger["committed"] -= 1
1310:                      logger.info(f"SLIPPAGE {ticker} {st['direction']} {st['setup']}: {best_run*100:.1f}% runaway, slot freed")
1311:                  else:
1312:                      kept.append(sig)
1313:              pendings = kept
1314:  
1315:          # fill gate + grade confirmed trades
1316:          for ticker in universe[:min(20, len(universe))]:
1317:              m5, b15 = fetch_dual_bars(ticker)
1318:              m5f = prepare_frames(m5, b15)["m5"]
1319:              filled, pendings = check_fills(ticker, pendings, m5f)
1320:              for f in filled:
1321:                  fill_bars = m5f[(m5f.index >= f["filled_at"]) & (m5f.index <= m5f.index[-1])]
1322:                  entry, stop, target = f["entry"], f["stop"], f["target"]
1323:                  exit_price, exit_reason, result = None, None, None
1324:                  stopped = any(float(r["low"]) <= stop for _, r in fill_bars.iterrows())
1325:                  hit_target = any(float(r["high"]) >= target for _, r in fill_bars.iterrows())
1326:                  if stopped and hit_target:
1327:                      exit_reason, result, exit_price = "STOP_HIT", "LOSS", stop
1328:                  elif stopped:
1329:                      exit_reason, result, exit_price = "STOP_HIT", "LOSS", stop
1330:                  elif hit_target:
1331:                      exit_reason, result, exit_price = "TARGET_HIT", "WIN", target
1332:                  else:
1333:                      exit_reason, result, exit_price = "EOD_SQUAREOFF", "FLAT", float(fill_bars["close"].iloc[-1])
1334:                  jrow = {
1335:                      "TradeID": next_trade_id(_today_date_str()), "Date": _today_date_str(),
1336:                      "SignalTime": f["signal_time"].strftime("%H:%M"), "Symbol": f["symbol"],
1337:                      "Direction": f["direction"], "Setup": f["setup"], "Entry": f["entry"],
1338:                      "StopLoss": f["stop"], "Target": f["target"], "Qty": f["qty"],
1339:                      "RiskPerShare": f["risk_per_share"], "RiskAmt": f["risk_amt"],
1340:                      "RewardAmt": f["reward_amt"], "RR": f["rr"], "Status": "CLOSED",
1341:                      "Mode": f["mode"], "ExitPrice": exit_price,
1342:                      "ExitTime": ist_now().strftime("%H:%M"), "ExitReason": exit_reason,
1343:                      "Result": result,
1344:                      "PnL_R": round(((exit_price - entry) / abs(entry - stop)) if f["direction"] == "LONG"
1345:                                     else ((entry - exit_price) / abs(entry - stop)), 3),
1346:                      "PnL_Amt": round((((exit_price - entry) / abs(entry - stop)) if f["direction"] == "LONG"
1347:                                        else ((entry - exit_price) / abs(entry - stop))) * f["risk_amt"], 2),
1348:                      "UpdatedAt": ist_now().strftime("%Y-%m-%dT%H:%M"),
1349:                      "Remarks": f"signal fired {f['signal_time'].strftime('%H:%M')}; filled on retrace to trigger @{f['filled_at'].strftime('%H:%M')} · NIFTY {regime['label']} ({regime['score']})",
1350:                      "Notes": f"{f['mode']} ({f['reason']}); confluence {f['conf']:.2f}; {f['note']}"}
1351:                  append_journal([jrow])
1352:                  ledger = day_ledger(load_journal())
1353:                  slots_free = CAP_TRADES_DAY - ledger["committed"]
1354:                  tg_send(f"FILL CONFIRMED [{f['direction']} {f['symbol']}] entry {f['entry']:.2f} exit {exit_price:.2f} {exit_reason} ({result})")
1355:  
1356:          # EOD square-off leftovers at last price
1357:          if min_now >= SQ_OFF_MIN:
1358:              for ticker in universe[:min(20, len(universe))]:
1359:                  m5, b15 = fetch_dual_bars(ticker)
1360:                  rest_bars = m5[m5.index >= ist_now()]
1361:                  last_p = float(rest_bars["close"].iloc[-1]) if not rest_bars.empty else None
1362:                  if last_p is None:
1363:                      continue
1364:                  for r in ledger["open"][:]:
1365:                      if r["Symbol"] == ticker:
1366:                          r["Status"] = "CLOSED"; r["ExitPrice"] = last_p
1367:                          r["ExitTime"] = ist_now().strftime("%H:%M"); r["ExitReason"] = "EOD_SQUAREOFF"
1368:                          r["Result"] = "FLAT"; r["PnL_R"], r["PnL_Amt"] = 0.0, 0.0
1369:                  ledger = day_ledger(load_journal())
1370:  
1371:          finalize_note = ""
1372:          if min_now in FINALIZE_TIMES:
1373:              finalize_note = f"Finalized at {min_now} IST"
1374:  
1375:      # EOD learning snapshot (after 15:05)
1376:      snap = min_now >= SQ_OFF_MIN
1377:      trend = classify_session_trend()
1378:      recs = ranked_buckets()
1379:      tuner = _tuner_load()
1380:      tuner["last_learning"] = _today_date_str()
1381:      _tuner_store(tuner)
1382:      table = []
1383:      for bkt, st in grade_buckets(load_journal()).items():
1384:          status = tuner.get("buckets", {}).get(bkt, {}).get("status", "enabled")
1385:          table.append((bkt, status, st["netR"], st["win_rate"]*100, st["tot_grades"], "mute" if status == "muted" else "armed"))
1386:      cards = [render_card(sig, ledger["committed"], sig["mode"]) for sig in fired_today if sig["mode"] == "TRADE"]
1387:      write_report(phase, regime, table, cards, recs, trend, notes=finalize_note)
1388:      logger.info(f"EOD learning snapshot written to reports_v5/")
1389:      tg_send(f"VOL5 EOD {_today_date_str()}: tape bias {trend}, {len(ledger['open'])} open rows managed. Full learning snapshot in reports/.")
1390:  
1391:      # persist state
1392:      _state_store("pending", _sig_ts(pendings))
1393:      _state_store("last_sweep", ist_now().isoformat())
1394:      notes_final = finalize_note
1395:      if ledger["tripped"]:
1396:          notes_final += f" Circuit breaker tripped on {_today_date_str()} (-2R day or two consecutive losses); new setups TRACKING only."
1397:      write_daily_summary(ledger, notes=notes_final)
1398:      logger.info(f"VOL5 scan done — phase={phase} open={len(ledger['open'])} committed={ledger['committed']}")
1399:  
1400:  
1401:  if __name__ == "__main__":
1402:      sys.exit(main())
