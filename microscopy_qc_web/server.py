"""
MicroScope QC — FastAPI Backend
Serves the static frontend and exposes /api/analyze endpoint.
"""

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from analyzer import analyze_image, QualityReport

app = FastAPI(
    title="MicroScope QC API",
    description="Microscopy image quality analysis",
    version="1.0.0",
)

# Allow any origin for the API (so frontend can be hosted separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent

# Mount static folder for CSS / JS / images
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# ──────────────────────────────────────────────
def bgr_to_data_url(bgr_img: np.ndarray, fmt: str = ".jpg", quality: int = 88) -> str:
    """Encode a BGR numpy image to a base64 data URL."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == ".jpg" else []
    ok, buf = cv2.imencode(fmt, bgr_img, encode_params)
    if not ok:
        raise RuntimeError("Image encoding failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


def report_to_dict(report: QualityReport, original_bgr: np.ndarray) -> dict:
    """Serialise a QualityReport into a JSON-friendly dict for the frontend."""

    def metric_dict(m):
        return {
            "score":    m.score,
            "raw":      m.raw,
            "issues":   m.issues,
            "severity": m.severity,
        }

    payload = {
        "overall_score": report.overall_score,
        "label":         report.label,
        "summary":       report.summary,
        "metrics": {
            "blur":     metric_dict(report.blur),
            "lighting": metric_dict(report.lighting),
            "noise":    metric_dict(report.noise),
            "density":  metric_dict(report.density),
        },
        "images": {
            "original":  bgr_to_data_url(original_bgr),
            "annotated": bgr_to_data_url(report.annotated_image) if report.annotated_image is not None else None,
            "heatmap":   bgr_to_data_url(report.heatmap_image)   if report.heatmap_image   is not None else None,
        },
        "histogram": report.histogram_data,
        "image_info": {
            "width":  int(original_bgr.shape[1]),
            "height": int(original_bgr.shape[0]),
        },
    }
    return payload


# ══════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend index page."""
    index_path = BASE_DIR / "templates" / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Run quality analysis on the uploaded microscopy image."""
    # Validate
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    if len(contents) > 25 * 1024 * 1024:    # 25 MB cap
        raise HTTPException(status_code=413, detail="Image too large (>25 MB)")

    # Decode
    arr = np.frombuffer(contents, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Resize huge images for speed (but keep nice quality)
    h, w = img_bgr.shape[:2]
    MAX_DIM = 1600
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)

    # Analyze
    try:
        report = analyze_image(img_bgr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return JSONResponse(report_to_dict(report, img_bgr))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
