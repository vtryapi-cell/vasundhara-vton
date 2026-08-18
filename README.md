# Vasundhara Pro9 Virtual Try-On

A simple Flask website for Vasundhara Pro9 using FASHN Try-On v1.6.

## Security
Set `FASHN_API_KEY` as a server environment variable. Never put the API key in frontend JavaScript or commit it to GitHub.

## Local run
```bash
pip install -r requirements.txt
export FASHN_API_KEY="YOUR_KEY"
python app.py
```

Windows PowerShell:
```powershell
$env:FASHN_API_KEY="YOUR_KEY"
python app.py
```

Then open http://localhost:8000

## Deployment
The included `render.yaml` can be used as a starting point for a Render web service.
