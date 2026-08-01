# Northstar

Northstar is an interactive Streamlit app for US company valuation, portfolio risk planning, and auditable options screening. It presents valuation ranges and ranked option candidates as decision support, not investment advice.

## Run locally

Use Python 3.12 from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The command-line valuation workflow remains available through `python main.py --help`.

## Publish the interactive app

GitHub Pages cannot run Northstar because Pages serves static HTML, CSS, and JavaScript while this app requires a Python process. Use GitHub for the source repository and Streamlit Community Cloud for the public website:

1. Create a GitHub repository and push this project to its `main` branch.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with the GitHub account that owns the repository.
3. Choose **Create app**, select the repository and `main` branch, and set the entrypoint to `app.py`.
4. Keep Python at 3.12, choose an available `streamlit.app` subdomain, and deploy.

No API keys are required. Market and company data are fetched from public providers at runtime and may be delayed or temporarily unavailable.

### Turn on the daily option-snapshot refresh

The repository includes a GitHub Actions workflow that captures regular-session option data at **13:17 America/New_York every weekday**. It covers 52 curated underlyings across broad-market, sector, technology, semiconductor, macro, and alternative ETFs; semiconductor companies; memory/storage companies; and liquid mega-cap stocks. The Streamlit selectors show this scheduled list but still accept a custom US-listed ticker.

After pushing the repository to GitHub:

1. Open the repository's **Settings → Actions → General** page.
2. Under **Workflow permissions**, select **Read and write permissions** and save. This allows the workflow to commit validated snapshots back to the repository.
3. Open **Actions → refresh option snapshots → Run workflow** for an initial run during the US regular session, or wait for the next scheduled weekday run.
4. Open `data/option_snapshot_status.json` after the run to see attempted, saved, failed, and currently available coverage.

The workflow splits the 52 symbols into six throttled batches. It publishes one compact rolling file per symbol under `data/option_snapshots/`, and keeps the last good file when a holiday, closed market, provider error, incomplete chain, or unmarketable chain prevents a valid refresh. Each published snapshot must use an aligned regular-session underlying price and put/call chains, have no failed selected expirations, and contain at least one marketable out-of-the-money put.

Streamlit Community Cloud redeploys after the workflow's data commit reaches the GitHub repository, so off-hours scans can read these bundled snapshots. A custom ticker outside the curated list still needs one successful live scan during a regular session before that particular Streamlit runtime can replay it. To change the scheduled list, edit `data/option_universe.json` and keep every ticker unique.

## Privacy and persistence

The web app keeps each visitor's portfolio in an isolated Streamlit session. It is not written to a shared public-server file. Visitors can download a JSON backup and restore it later. Local command-line portfolio data uses `portfolio.json`, which is intentionally excluded from Git.

Provider caches and custom-ticker snapshots created inside Streamlit are runtime conveniences and may disappear when the app restarts. The scheduled 52-ticker snapshots are different: GitHub Actions validates and commits their rolling files to the repository, making them available again after a Streamlit restart. They remain historical planning snapshots, not executable quotes or a permanent market-data archive.

## Verification

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app.py main.py valuation scripts tests
python scripts/refresh_option_snapshots.py --dry-run --batch-index 0 --batch-count 6
```
