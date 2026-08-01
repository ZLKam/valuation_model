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

## Privacy and persistence

The web app keeps each visitor's portfolio in an isolated Streamlit session. It is not written to a shared public-server file. Visitors can download a JSON backup and restore it later. Local command-line portfolio data uses `portfolio.json`, which is intentionally excluded from Git.

Option-chain snapshots and provider caches are runtime conveniences only. Cloud-generated files may disappear when the app restarts, so historical replay should not be treated as durable cloud storage.

## Verification

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app.py main.py valuation tests
```
