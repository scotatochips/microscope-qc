# MicroScope QC — Standalone Website

Drop-in microscopy image quality analyzer. FastAPI backend + static HTML/CSS/JS frontend. No Streamlit, no React build step, no nonsense.

## What it does

Detects four classes of image quality issues:

| Metric | Algorithms |
|--------|------------|
| **Sharpness** | Laplacian variance · Brenner gradient · Tenengrad |
| **Lighting** | Mean brightness · Tile coefficient-of-variation · Saturation fraction |
| **Noise** | Immerkær σ · SNR · Salt-and-pepper · FFT high-freq ringing |
| **Density** | Adaptive threshold + contour detection · Spatial CV grid |

Returns: weighted overall score (0–100), GOOD/NOT-SUITABLE label, full breakdown with raw measurements and human-readable issues, plus three visualizations (cell detection overlay, density heatmap, RGB histogram).

## Project structure

```
microscopy_qc_web/
├── server.py              ← FastAPI app
├── analyzer.py            ← analysis engine (unchanged)
├── templates/
│   └── index.html         ← single-page frontend
├── static/
│   ├── style.css
│   └── script.js
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Run locally (no Docker)

```bash
# Create venv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Run dev server (with auto-reload)
uvicorn server:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

## Run with Docker

```bash
docker compose up
```

Same URL: http://localhost:8000

## Deployment

### Railway (easiest, free tier)

1. Push this folder to a GitHub repo
2. Go to [railway.app](https://railway.app), sign in with GitHub
3. **New Project → Deploy from GitHub** → pick your repo
4. Railway auto-detects the Dockerfile and deploys it
5. Click "Generate Domain" — done. You'll get something like `microscope-qc.up.railway.app`

### Render (also free)

1. Push to GitHub
2. [render.com](https://render.com) → **New → Web Service** → connect repo
3. Pick **Docker** as the runtime, leave defaults
4. Click **Create Web Service**

### Fly.io

```bash
brew install flyctl    # or curl install
fly launch             # uses the Dockerfile, walks you through setup
fly deploy
```

### Plain VPS / EC2

```bash
git clone <your-repo>
cd microscopy_qc_web
docker compose up -d
```

Then put nginx in front for HTTPS, or use Caddy:

```caddyfile
your-domain.com {
    reverse_proxy localhost:8000
}
```

## API reference

### `POST /api/analyze`

Accepts: `multipart/form-data` with field `file` (any image).

Returns JSON:

```json
{
  "overall_score": 78.4,
  "label": "GOOD FOR ANALYSIS",
  "summary": ["Mild blur detected (Laplacian variance 104)"],
  "metrics": {
    "blur":     { "score": 65.2, "raw": {...}, "issues": [...], "severity": "warning" },
    "lighting": { "score": 92.1, "raw": {...}, "issues": [],     "severity": "ok" },
    "noise":    { "score": 88.7, "raw": {...}, "issues": [],     "severity": "ok" },
    "density":  { "score": 79.0, "raw": {...}, "issues": [...], "severity": "warning" }
  },
  "images": {
    "original":  "data:image/jpeg;base64,...",
    "annotated": "data:image/jpeg;base64,...",
    "heatmap":   "data:image/jpeg;base64,..."
  },
  "histogram":  { "Red":[...], "Green":[...], "Blue":[...] },
  "image_info": { "width": 1024, "height": 1024 }
}
```

### `GET /api/health`

```json
{ "status": "ok" }
```

## Customizing the frontend

- **Theme colors:** edit CSS variables at the top of `static/style.css` (`--accent`, `--danger`, etc.)
- **Thresholds:** tune the constants at the top of `analyzer.py` to match your microscopy modality
- **Score weights:** adjust `W_BLUR`, `W_LIGHTING`, `W_NOISE`, `W_DENSITY` in `analyzer.py`

## Tech stack

- **Backend:** FastAPI · Uvicorn · OpenCV · NumPy
- **Frontend:** Vanilla HTML/CSS/JS — no build step, no framework
- **Fonts:** Instrument Serif · Inter Tight · JetBrains Mono (Google Fonts)

## License

Use it however you want.
