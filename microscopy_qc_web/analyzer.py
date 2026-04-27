"""
MicroScope QC — Advanced Microscopy Image Quality Analyzer
===========================================================
Detects: blur, lighting anomalies, noise/artifacts, cell density issues.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Thresholds (tuned for blood-smear microscopy)
# ─────────────────────────────────────────────
BLUR_GOOD       = 300.0
BLUR_BAD        = 60.0
BRIGHT_LOW      = 40
BRIGHT_HIGH     = 220
BRIGHT_IDEAL_LO = 80
BRIGHT_IDEAL_HI = 180
UNIFORMITY_THRESH = 50
NOISE_GOOD      = 2.5
NOISE_BAD       = 12.0
DENSITY_SPARSE  = 2.0
DENSITY_DENSE   = 65.0
DENSITY_IDEAL_LO = 8.0
DENSITY_IDEAL_HI = 45.0
DENSITY_CV_THRESH = 0.60

W_BLUR      = 0.35
W_LIGHTING  = 0.25
W_NOISE     = 0.20
W_DENSITY   = 0.20


@dataclass
class MetricResult:
    score: float
    raw: dict
    issues: list
    severity: str


@dataclass
class QualityReport:
    blur:     MetricResult = None
    lighting: MetricResult = None
    noise:    MetricResult = None
    density:  MetricResult = None
    overall_score: float = 0.0
    label: str = ""
    summary: list = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None
    heatmap_image:   Optional[np.ndarray] = None
    histogram_data:  Optional[dict] = None


# ══════════════════════════════════════════════
# 1. BLUR
# ══════════════════════════════════════════════
def _brenner_gradient(gray):
    diff_h = gray[2:, :].astype(np.float32) - gray[:-2, :].astype(np.float32)
    diff_v = gray[:, 2:].astype(np.float32) - gray[:, :-2].astype(np.float32)
    return float(np.mean(diff_h ** 2) + np.mean(diff_v ** 2))


def analyze_blur(gray):
    lap_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brenner   = _brenner_gradient(gray)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sx**2 + sy**2))

    s1 = np.clip((lap_var  - BLUR_BAD)  / (BLUR_GOOD  - BLUR_BAD)  * 100, 0, 100)
    s2 = np.clip((brenner  - 50)        / (2000 - 50)               * 100, 0, 100)
    s3 = np.clip((tenengrad - 50)       / (3000 - 50)               * 100, 0, 100)
    score = float((s1 + s2 + s3) / 3)

    issues = []
    if lap_var < BLUR_BAD:
        severity = "critical"
        issues.append(f"Severe blur detected (Laplacian variance {lap_var:.1f} < {BLUR_BAD})")
    elif lap_var < BLUR_GOOD * 0.5:
        severity = "warning"
        issues.append(f"Mild blur detected (Laplacian variance {lap_var:.1f})")
    else:
        severity = "ok"

    return MetricResult(
        score=round(score, 1),
        raw={"laplacian_var": round(lap_var, 2), "brenner": round(brenner, 2), "tenengrad": round(tenengrad, 2)},
        issues=issues, severity=severity,
    )


# ══════════════════════════════════════════════
# 2. LIGHTING
# ══════════════════════════════════════════════
def _tile_brightness(gray, tiles=4):
    h, w = gray.shape
    th, tw = h // tiles, w // tiles
    vals = []
    for r in range(tiles):
        for c in range(tiles):
            tile = gray[r*th:(r+1)*th, c*tw:(c+1)*tw]
            vals.append(float(np.mean(tile)))
    return np.array(vals)


def analyze_lighting(gray):
    mean_b = float(np.mean(gray))
    std_b  = float(np.std(gray))
    tile_means = _tile_brightness(gray, tiles=4)
    tile_cv    = float(np.std(tile_means) / (np.mean(tile_means) + 1e-6))
    sat_frac   = float(np.mean(gray > 250)) * 100

    issues, score = [], 100.0

    if mean_b < BRIGHT_LOW:
        score -= (BRIGHT_LOW - mean_b) / BRIGHT_LOW * 55
        issues.append(f"Image severely under-exposed (mean brightness {mean_b:.0f}/255)")
    elif mean_b < BRIGHT_IDEAL_LO:
        score -= (BRIGHT_IDEAL_LO - mean_b) / BRIGHT_IDEAL_LO * 25
        issues.append(f"Image slightly dark (mean brightness {mean_b:.0f}/255)")
    elif mean_b > BRIGHT_HIGH:
        score -= (mean_b - BRIGHT_HIGH) / (255 - BRIGHT_HIGH) * 55
        issues.append(f"Image over-exposed / saturated (mean brightness {mean_b:.0f}/255)")
    elif mean_b > BRIGHT_IDEAL_HI:
        score -= (mean_b - BRIGHT_IDEAL_HI) / (BRIGHT_HIGH - BRIGHT_IDEAL_HI) * 25
        issues.append(f"Image slightly bright (mean brightness {mean_b:.0f}/255)")

    if tile_cv > 0.18:
        score -= min(35, tile_cv * 100)
        issues.append(f"Uneven illumination across image (tile CV = {tile_cv:.2f})")
    elif tile_cv > 0.10:
        score -= 12
        issues.append(f"Mild illumination gradient detected (tile CV = {tile_cv:.2f})")

    if sat_frac > 5:
        score -= min(20, sat_frac)
        issues.append(f"{sat_frac:.1f}% pixels fully saturated")

    score = max(0.0, min(100.0, score))
    severity = "ok" if score >= 75 else ("warning" if score >= 45 else "critical")

    return MetricResult(
        score=round(score, 1),
        raw={"mean_brightness": round(mean_b, 1), "std_brightness": round(std_b, 1),
             "tile_cv": round(tile_cv, 3), "saturation_pct": round(sat_frac, 2)},
        issues=issues, severity=severity,
    )


# ══════════════════════════════════════════════
# 3. NOISE
# ══════════════════════════════════════════════
def _estimate_noise_sigma(gray):
    H = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], dtype=np.float32)
    filtered = cv2.filter2D(gray.astype(np.float32), -1, H)
    sigma = np.sqrt(np.pi / 2) * np.mean(np.abs(filtered)) / 6
    return float(sigma)


def _ringing_artifacts(gray):
    f = np.fft.fft2(gray.astype(np.float32))
    mag = np.abs(np.fft.fftshift(f))
    h, w = mag.shape
    cy, cx = h//2, w//2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x-cx)**2 + (y-cy)**2)
    hf = float(np.sum(mag[r > min(h,w)*0.3]))
    return hf / (float(np.sum(mag)) + 1e-9)


def analyze_noise(gray):
    sigma      = _estimate_noise_sigma(gray)
    ring_ratio = _ringing_artifacts(gray)
    snr        = float(np.mean(gray)) / (sigma + 1e-6)
    sp_frac    = float(np.mean((gray < 5) | (gray > 250))) * 100

    issues, score = [], 100.0

    if sigma > NOISE_BAD:
        score -= min(45, (sigma - NOISE_BAD) / NOISE_BAD * 45)
        issues.append(f"High noise level (σ = {sigma:.1f}, SNR = {snr:.1f})")
    elif sigma > NOISE_GOOD * 2:
        score -= 18
        issues.append(f"Moderate noise present (σ = {sigma:.1f})")

    if sp_frac > 0.5:
        score -= min(30, sp_frac * 8)
        issues.append(f"Salt-and-pepper artifacts: {sp_frac:.2f}% pixels")

    if ring_ratio > 0.45:
        score -= 15
        issues.append("Ringing/compression artifacts in frequency domain")

    score = max(0.0, min(100.0, score))
    severity = "ok" if score >= 75 else ("warning" if score >= 45 else "critical")

    return MetricResult(
        score=round(score, 1),
        raw={"noise_sigma": round(sigma, 2), "snr": round(snr, 2),
             "sp_artifact_pct": round(sp_frac, 3), "hf_ring_ratio": round(ring_ratio, 3)},
        issues=issues, severity=severity,
    )


# ══════════════════════════════════════════════
# 4. CELL DENSITY
# ══════════════════════════════════════════════
def analyze_density(gray, original_bgr):
    h, w = gray.shape

    blur   = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, blockSize=31, C=6)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask   = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  kernel, iterations=1)

    coverage = float(np.mean(mask > 0)) * 100

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (h * w) * 0.0001
    max_area = (h * w) * 0.10
    cells = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    cell_count = len(cells)

    # Density heatmap grid
    grid_r, grid_c = 6, 6
    tile_h, tile_w = h // grid_r, w // grid_c
    density_grid   = np.zeros((grid_r, grid_c), dtype=np.float32)
    for c in cells:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx_ = int(M["m10"] / M["m00"])
            cy_ = int(M["m01"] / M["m00"])
            density_grid[min(cy_//tile_h, grid_r-1), min(cx_//tile_w, grid_c-1)] += 1

    grid_mean  = float(np.mean(density_grid))
    density_cv = float(np.std(density_grid)) / (grid_mean + 1e-6)

    # Heatmap overlay
    heatmap_norm = cv2.resize(density_grid, (w,h), interpolation=cv2.INTER_LINEAR)
    if heatmap_norm.max() > 0:
        heatmap_norm = (heatmap_norm / heatmap_norm.max() * 255)
    heatmap_uint8   = np.clip(heatmap_norm, 0, 255).astype(np.uint8)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_INFERNO)
    heatmap_overlay = cv2.addWeighted(original_bgr, 0.55, heatmap_color, 0.45, 0)

    # Annotated image
    annotated = original_bgr.copy()
    for c in cells:
        (cx_, cy_), radius = cv2.minEnclosingCircle(c)
        cv2.circle(annotated, (int(cx_), int(cy_)), int(radius)+2, (0,230,180), 1, lineType=cv2.LINE_AA)

    issues, score = [], 100.0

    if coverage < DENSITY_SPARSE:
        score -= 45
        issues.append(f"Region too sparse — very few cells ({coverage:.1f}% coverage, ~{cell_count} cells)")
    elif coverage < DENSITY_IDEAL_LO:
        score -= 18
        issues.append(f"Low cell density ({coverage:.1f}% coverage, ~{cell_count} cells)")
    elif coverage > DENSITY_DENSE:
        score -= 40
        issues.append(f"Overcrowded — cells heavily overlapping ({coverage:.1f}% coverage)")
    elif coverage > DENSITY_IDEAL_HI:
        score -= 18
        issues.append(f"High cell density — segmentation may be unreliable ({coverage:.1f}% coverage)")

    if density_cv > DENSITY_CV_THRESH:
        score -= min(30, density_cv * 30)
        issues.append(f"Uneven cell distribution — clustered regions (spatial CV = {density_cv:.2f})")

    score = max(0.0, min(100.0, score))
    severity = "ok" if score >= 75 else ("warning" if score >= 45 else "critical")

    return (
        MetricResult(
            score=round(score, 1),
            raw={"coverage_pct": round(coverage, 2), "approx_cell_count": cell_count,
                 "density_cv": round(density_cv, 3)},
            issues=issues, severity=severity,
        ),
        annotated,
        heatmap_overlay,
    )


# ══════════════════════════════════════════════
# 5. MASTER ANALYSIS
# ══════════════════════════════════════════════
def analyze_image(image_bgr: np.ndarray) -> QualityReport:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_r    = analyze_blur(gray)
    light_r   = analyze_lighting(gray)
    noise_r   = analyze_noise(gray)
    dens_r, annotated_cells, heatmap = analyze_density(gray, image_bgr)

    overall = round(float(
        W_BLUR * blur_r.score +
        W_LIGHTING * light_r.score +
        W_NOISE * noise_r.score +
        W_DENSITY * dens_r.score
    ), 1)

    label   = "GOOD FOR ANALYSIS" if overall >= 60 else "NOT SUITABLE FOR ANALYSIS"
    summary = blur_r.issues + light_r.issues + noise_r.issues + dens_r.issues

    hist_data = {}
    for ch_idx, ch_name in enumerate(["Blue","Green","Red"]):
        h = cv2.calcHist([image_bgr], [ch_idx], None, [256], [0,256])
        hist_data[ch_name] = h.flatten().tolist()

    return QualityReport(
        blur=blur_r, lighting=light_r, noise=noise_r, density=dens_r,
        overall_score=overall, label=label, summary=summary,
        annotated_image=annotated_cells, heatmap_image=heatmap,
        histogram_data=hist_data,
    )


# ══════════════════════════════════════════════
# 6. CLI PRINTER
# ══════════════════════════════════════════════
def print_report(report: QualityReport, filename: str = "image"):
    SEP = "─" * 60
    W   = 30

    def bar(s):
        filled = int(s / 100 * W)
        col = "\033[92m" if s>=75 else ("\033[93m" if s>=45 else "\033[91m")
        return f"{col}{'█'*filled}{'░'*(W-filled)}\033[0m {s:.1f}/100"

    print(f"\n{SEP}\n  🔬  MicroScope QC  —  {filename}\n{SEP}")
    lc = "\033[92m" if report.label.startswith("GOOD") else "\033[91m"
    print(f"\n  Overall Score : {bar(report.overall_score)}")
    print(f"  Status        : {lc}{report.label}\033[0m\n{SEP}")

    for name, m in [("Sharpness / Blur", report.blur),
                    ("Lighting / Exposure", report.lighting),
                    ("Noise / Artifacts", report.noise),
                    ("Cell Density", report.density)]:
        icon = {"ok":"✅","warning":"⚠️ ","critical":"❌"}[m.severity]
        print(f"  {icon}  {name:<22} {bar(m.score)}")
        for k,v in m.raw.items():
            print(f"       ↳ {k}: {v}")
        for issue in m.issues:
            print(f"       • {issue}")

    print(f"\n{SEP}")
    if report.summary:
        print("  Issues found:")
        for i in report.summary:
            print(f"    • {i}")
    else:
        print("  No significant issues found.")
    print(SEP + "\n")
