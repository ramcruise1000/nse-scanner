# NSE F&O Intraday Scanner - SOP Volume IV Companion

Automated scanner for four strict 1:5 reward-risk intraday strategies on the
NSE F&O universe. Runs free on GitHub Actions every 10 minutes from 08:30 to
15:20 IST (Mon-Fri), writes a Markdown report to `reports/`, and can send
Telegram alerts. No coding, no paid APIs, no server.

## Files in this repository

| File | Purpose |
|------|---------|
| `scanner.py` | The scanner: downloads data, runs S1-S4, writes reports |
| `universe.py` | The F&O stock list (edit monthly) |
| `requirements.txt` | Python libraries GitHub installs automatically |
| `.github/workflows/market-sweep.yml` | The schedule (cron) definition |
| `reports/` | Created automatically; one report per trading day |

## Quick start (browser only - no command line)

1. Create a GitHub account, then click **New repository** (Public, add README).
2. Click **Add file > Create new file** and recreate each file above by
   copy-pasting from the SOP document (the workflow file must be created at
   the exact path `.github/workflows/market-sweep.yml`).
3. Open the **Actions** tab and enable workflows if asked.
4. Click **Run workflow** (manual trigger) to test - a green tick and a
   `reports/` file mean the machine works.
5. Optional: create a Telegram bot and add `TELEGRAM_BOT_TOKEN` and
   `TELEGRAM_CHAT_ID` under **Settings > Secrets and variables > Actions**.

## Signals are alerts, not commands

Every signal card prints entry, stop (structural, 0.3-1.5% band), a 5R
target, quantity for Rs 200 risk, and the validity window. You execute in
your broker app. Read the SOP's verification gates before trading real
money: 100+ backtested signals, then 30 paper trades, then half size.

## Maintenance

Once a month verify the universe against NSE's official F&O list and update
`universe.py`. The scanner automatically skips symbols whose data is missing.

Educational use only. Nothing here is investment advice.

