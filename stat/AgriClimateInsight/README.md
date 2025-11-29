# AgriClimateInsight

Lightweight Streamlit app + data science toolkit for agricultural climate insights.

## Prerequisites
- Windows 10/11
- Python 3.10+ (recommended)
- Git (optional)

## Quick setup (PowerShell)
```powershell
cd "c:\Users\saisa\OneDrive\Desktop\Agriculture\stat\AgriClimateInsight"

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Run the app
Replace `app.py` below with the actual Streamlit entrypoint filename if different:
```powershell
streamlit run app.py
```

## Notes
- Requirements are listed in `requirements.txt`.
- Add data under a `data/` directory and check config or paths used by the app.
- Keep secrets out of the repo (use environment variables or a secure store).

## Contributing
1. Create a branch.
2. Add code + tests.
3. Open a PR with a clear description.

## License
Add your preferred license file (`LICENSE`) to the repo.
