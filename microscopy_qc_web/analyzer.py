"""
═══════════════════════════════════════════════════════════════════════
MicroScope QC v2.0 — Microscopy Image Quality Analyzer
═══════════════════════════════════════════════════════════════════════

Detects four classes of quality defects:
    1. Defocus / motion blur
    2. Illumination defects (exposure, vignetting, dynamic range)
    3. Noise & sensor artifacts
    4. Specimen density issues

Architecture:
    - Each metric reports SEVERITY (pass/warn/fail) independently
    - Final verdict from deterministic rules, not just the score
    - Every conclusion: measured value + threshold + rule ID
    - All thresholds are auditable constants at top of file

References:
    - Pertuz et al. 2013, "Analysis of focus measure operators"
    - Immerkær 1996, "Fast noise variance estimation"
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

VERSION = "2.0.0"

# ═══════════ THRESHOLDS ═══════════
BLUR_LV_FAIL  =  50.0
BLUR_LV_WARN  = 150.0
BLUR_LV_PASS  = 400.0
TEN_FAIL  =   500.0
TEN_WARN  =  3000.0
TEN_PASS  = 12000.0
EDGE_DENSITY_FAIL  = 0.015
EDGE_DENSITY_WARN  = 0.040

EXPOSURE_DARK_FAIL    =  35
EXPOSURE_DARK_WARN    =  70
EXPOSURE_BRIGHT_WARN  = 195
EXPOSURE_BRIGHT_FAIL  = 225
SATURATED_HIGH_FAIL = 0.10
SATURATED_HIGH_WARN = 0.03
SATURATED_LOW_FAIL  = 0.10
SATURATED_LOW_WARN  = 0.03
DYNAMIC_RANGE_FAIL =  60
DYNAMIC_RANGE_WARN = 100
TILE_CV_FAIL = 0.25
TILE_CV_WARN = 0.12

NOISE_SIGMA_FAIL = 14.0
NOISE_SIGMA_WARN =  7.0
SNR_FAIL =  8.0
SNR_WARN = 18.0
SP_FAIL = 0.020
SP_WARN = 0.005

COVERAGE_SPARSE_FAIL =   2.0
COVERAGE_SPARSE_WARN =   8.0
COVERAGE_DENSE_WARN  =  50.0
COVERAGE_DENSE_FAIL  =  68.0
DENSITY_CV_FAIL = 1.20
DENSITY_CV_WARN = 0.70
TOUCHING_FAIL = 0.65
TOUCHING_WARN = 0.35

WEIGHT_BLUR     = 0.35
WEIGHT_EXPOSURE = 0.20
WEIGHT_NOISE    = 0.20
WEIGHT_DENSITY  = 0.25

SCORE_PASS     = 75.0
SCORE_REVIEW   = 55.0


# ═══════════ DATA CLASSES ═══════════
@dataclass
class Finding:
    rule_id: str
    severity: str
    metric: str
    measured: float
    threshold: float
    operator: str
    message: str
    impact: str

@dataclass
class MetricResult:
    name: str
    score: float
    severity: str
    measurements: dict
    findings: list

@dataclass
class Verdict:
    decision: str
    confidence: float
    reasoning: list
    blockers: list

@dataclass
class QualityReport:
    version: str
    timestamp: str
    image_info: dict
    metrics: dict
    overall_score: float
    verdict: Verdict
    annotated_image: Optional[np.ndarray] = None
    heatmap_image:   Optional[np.ndarray] = None
    histogram_data:  Optional[dict] = None


# ═══════════ HELPERS ═══════════
def _band_score(value, fail, warn, pass_, higher_is_better=True):
    if higher_is_better:
        if value <= fail:  return max(0.0, (value / fail) * 30)
        if value <= warn:  return 30 + (value - fail) / (warn - fail) * 30
        if value <= pass_: return 60 + (value - warn) / (pass_ - warn) * 30
        return min(100.0, 90 + (value - pass_) / pass_ * 10)
    else:
        if value >= fail:  return max(0.0, 30 - (value - fail) / fail * 30)
        if value >= warn:  return 30 + (fail - value) / (fail - warn) * 30
        if value >= pass_: return 60 + (warn - value) / (warn - pass_) * 30
        return min(100.0, 90 + (pass_ - value) / pass_ * 10)

def _worst(severities):
    if "fail" in severities: return "fail"
    if "warn" in severities: return "warn"
    return "pass"


# ═══════════ 1. BLUR ═══════════
def analyze_blur(gray):
    g = gray.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    lap_var = float(lap.var())
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sx*sx + sy*sy)
    tenengrad = float(np.mean(grad_mag ** 2))
    grad_8u = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges = cv2.threshold(grad_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    f = np.fft.fft2(g)
    mag = np.abs(np.fft.fftshift(f))
    h, w = mag.shape
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - w//2)**2 + (Y - h//2)**2)
    cutoff = min(h, w) * 0.15
    hf_ratio = float(mag[r > cutoff].sum()) / (float(mag.sum()) + 1e-9)

    s_lv  = _band_score(lap_var,    BLUR_LV_FAIL, BLUR_LV_WARN, BLUR_LV_PASS, True)
    s_ten = _band_score(tenengrad,  TEN_FAIL,     TEN_WARN,     TEN_PASS,     True)
    s_ed  = _band_score(edge_density, EDGE_DENSITY_FAIL, EDGE_DENSITY_WARN, 0.10, True)
    score = round(0.55*s_lv + 0.25*s_ten + 0.20*s_ed, 1)

    findings = []
    if lap_var < BLUR_LV_FAIL:
        findings.append(Finding("BLUR.LV.FAIL","fail","laplacian_variance",
            round(lap_var,2),BLUR_LV_FAIL,"<",
            f"Severe defocus or motion blur (LV={lap_var:.1f})",
            "Cell boundaries cannot be reliably segmented."))
    elif lap_var < BLUR_LV_WARN:
        findings.append(Finding("BLUR.LV.WARN","warn","laplacian_variance",
            round(lap_var,2),BLUR_LV_WARN,"<",
            f"Mild blur detected (LV={lap_var:.1f})",
            "Fine subcellular structures may be lost."))
    else:
        findings.append(Finding("BLUR.LV.PASS","pass","laplacian_variance",
            round(lap_var,2),BLUR_LV_WARN,"≥",
            f"Image is in focus (LV={lap_var:.1f})",
            "Sufficient sharpness for analysis."))

    if tenengrad < TEN_FAIL and lap_var < BLUR_LV_WARN:
        findings.append(Finding("BLUR.TEN.CONFIRM","fail","tenengrad",
            round(tenengrad,1),TEN_FAIL,"<",
            "Tenengrad confirms low gradient energy",
            "Independent confirmation of blur."))

    if edge_density < EDGE_DENSITY_FAIL:
        findings.append(Finding("BLUR.EDGES.FAIL","fail","edge_density",
            round(edge_density,4),EDGE_DENSITY_FAIL,"<",
            f"Almost no detectable edges ({edge_density*100:.2f}%)",
            "Image lacks expected cell structure."))

    return MetricResult(
        name="Sharpness",
        score=max(0.0, min(100.0, score)),
        severity=_worst([f.severity for f in findings]),
        measurements={
            "laplacian_variance": round(lap_var,2),
            "tenengrad": round(tenengrad,1),
            "edge_density": round(edge_density,4),
            "hf_energy_ratio": round(hf_ratio,4),
        },
        findings=findings,
    )


# ═══════════ 2. EXPOSURE ═══════════
def analyze_exposure(gray):
    g = gray.astype(np.float32)
    findings = []

    mean_v = float(g.mean())
    p1, p99 = np.percentile(g, [1, 99])
    dyn_range = float(p99 - p1)
    sat_high = float(np.mean(gray >= 254))
    sat_low  = float(np.mean(gray <= 1))

    h, w = gray.shape
    tiles = 5
    th, tw = h // tiles, w // tiles
    tile_means = []
    for r in range(tiles):
        for c in range(tiles):
            tile = g[r*th:(r+1)*th, c*tw:(c+1)*tw]
            tile_means.append(tile.mean())
    tile_means = np.array(tile_means, dtype=np.float32)
    tile_cv = float(tile_means.std() / (tile_means.mean() + 1e-6))

    small = cv2.resize(g, (64,64), interpolation=cv2.INTER_AREA)
    Y, X = np.mgrid[0:64, 0:64].astype(np.float32)
    A = np.column_stack([X.ravel()**2, Y.ravel()**2, (X*Y).ravel(),
                         X.ravel(), Y.ravel(), np.ones(64*64)])
    coef, *_ = np.linalg.lstsq(A, small.ravel(), rcond=None)
    surface = (A @ coef).reshape(64,64)
    vignette_strength = float((surface.max() - surface.min()) / (mean_v + 1e-6))

    if mean_v < EXPOSURE_DARK_FAIL or mean_v > EXPOSURE_BRIGHT_FAIL:
        s_exp = 0
    elif mean_v < EXPOSURE_DARK_WARN:
        s_exp = 30 + (mean_v - EXPOSURE_DARK_FAIL) / (EXPOSURE_DARK_WARN - EXPOSURE_DARK_FAIL) * 30
    elif mean_v > EXPOSURE_BRIGHT_WARN:
        s_exp = 30 + (EXPOSURE_BRIGHT_FAIL - mean_v) / (EXPOSURE_BRIGHT_FAIL - EXPOSURE_BRIGHT_WARN) * 30
    else:
        s_exp = 100

    s_uni = _band_score(tile_cv, TILE_CV_FAIL, TILE_CV_WARN, 0.04, False)
    s_dr  = _band_score(dyn_range, DYNAMIC_RANGE_FAIL, DYNAMIC_RANGE_WARN, 200, True)

    sat_penalty = 0
    if sat_high > SATURATED_HIGH_FAIL: sat_penalty += 30
    elif sat_high > SATURATED_HIGH_WARN: sat_penalty += 12
    if sat_low > SATURATED_LOW_FAIL: sat_penalty += 30
    elif sat_low > SATURATED_LOW_WARN: sat_penalty += 12

    score = round(0.50*s_exp + 0.30*s_uni + 0.20*s_dr - sat_penalty, 1)
    score = max(0.0, min(100.0, score))

    if mean_v < EXPOSURE_DARK_FAIL:
        findings.append(Finding("EXPOSURE.UNDER.FAIL","fail","mean_intensity",
            round(mean_v,1),EXPOSURE_DARK_FAIL,"<",
            f"Severely underexposed (mean = {mean_v:.0f}/255)",
            "Cell features may be lost in noise floor."))
    elif mean_v < EXPOSURE_DARK_WARN:
        findings.append(Finding("EXPOSURE.UNDER.WARN","warn","mean_intensity",
            round(mean_v,1),EXPOSURE_DARK_WARN,"<",
            f"Image is dim (mean = {mean_v:.0f}/255)",
            "Reduced contrast for staining-based identification."))
    elif mean_v > EXPOSURE_BRIGHT_FAIL:
        findings.append(Finding("EXPOSURE.OVER.FAIL","fail","mean_intensity",
            round(mean_v,1),EXPOSURE_BRIGHT_FAIL,">",
            f"Severely overexposed (mean = {mean_v:.0f}/255)",
            "Highlights clipped — cell detail unrecoverable."))
    elif mean_v > EXPOSURE_BRIGHT_WARN:
        findings.append(Finding("EXPOSURE.OVER.WARN","warn","mean_intensity",
            round(mean_v,1),EXPOSURE_BRIGHT_WARN,">",
            f"Image is bright (mean = {mean_v:.0f}/255)",
            "Risk of clipping in light cell regions."))
    else:
        findings.append(Finding("EXPOSURE.OK","pass","mean_intensity",
            round(mean_v,1),EXPOSURE_BRIGHT_WARN,"in_range",
            f"Exposure within target range (mean = {mean_v:.0f})",
            "Good histogram placement."))

    if sat_high > SATURATED_HIGH_FAIL:
        findings.append(Finding("EXPOSURE.CLIP.HIGH.FAIL","fail","saturated_high_fraction",
            round(sat_high,4),SATURATED_HIGH_FAIL,">",
            f"{sat_high*100:.1f}% of pixels are blown out (=255)",
            "Loss of detail in bright regions."))
    elif sat_high > SATURATED_HIGH_WARN:
        findings.append(Finding("EXPOSURE.CLIP.HIGH.WARN","warn","saturated_high_fraction",
            round(sat_high,4),SATURATED_HIGH_WARN,">",
            f"{sat_high*100:.1f}% pixels at maximum value",
            "Some highlight clipping."))
    if sat_low > SATURATED_LOW_FAIL:
        findings.append(Finding("EXPOSURE.CLIP.LOW.FAIL","fail","saturated_low_fraction",
            round(sat_low,4),SATURATED_LOW_FAIL,">",
            f"{sat_low*100:.1f}% of pixels are crushed to black",
            "Loss of detail in shadow regions."))

    if tile_cv > TILE_CV_FAIL:
        findings.append(Finding("EXPOSURE.UNIFORM.FAIL","fail","tile_cv",
            round(tile_cv,3),TILE_CV_FAIL,">",
            f"Strong illumination gradient (CV={tile_cv:.2f})",
            "Vignette or uneven lighting; consider flat-field correction."))
    elif tile_cv > TILE_CV_WARN:
        findings.append(Finding("EXPOSURE.UNIFORM.WARN","warn","tile_cv",
            round(tile_cv,3),TILE_CV_WARN,">",
            f"Mild illumination unevenness (CV={tile_cv:.2f})",
            "Edge regions may have biased measurements."))

    if dyn_range < DYNAMIC_RANGE_FAIL:
        findings.append(Finding("EXPOSURE.DR.FAIL","fail","dynamic_range",
            round(dyn_range,1),DYNAMIC_RANGE_FAIL,"<",
            f"Compressed dynamic range ({dyn_range:.0f}/255)",
            "Low contrast — staining differentiation impaired."))
    elif dyn_range < DYNAMIC_RANGE_WARN:
        findings.append(Finding("EXPOSURE.DR.WARN","warn","dynamic_range",
            round(dyn_range,1),DYNAMIC_RANGE_WARN,"<",
            f"Limited dynamic range ({dyn_range:.0f}/255)",
            "Lower contrast than ideal."))

    return MetricResult(
        name="Lighting",
        score=score,
        severity=_worst([f.severity for f in findings]),
        measurements={
            "mean_intensity": round(mean_v,1),
            "p1": round(float(p1),1),
            "p99": round(float(p99),1),
            "dynamic_range": round(dyn_range,1),
            "saturated_high_pct": round(sat_high*100,3),
            "saturated_low_pct": round(sat_low*100,3),
            "tile_cv": round(tile_cv,3),
            "vignette_strength": round(vignette_strength,3),
        },
        findings=findings,
    )


# ═══════════ 3. NOISE ═══════════
def _immerkaer_sigma(gray):
    H = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], dtype=np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), -1, H)
    return float(np.sqrt(np.pi/2) * np.mean(np.abs(conv)) / 6.0)

def _salt_pepper_fraction(gray):
    g = gray.astype(np.int16)
    extreme = (gray <= 1) | (gray >= 254)
    if not extreme.any(): return 0.0
    kernel = np.ones((3,3), dtype=np.float32) / 8.0
    kernel[1,1] = 0
    neighbour_mean = cv2.filter2D(g.astype(np.float32), -1, kernel)
    isolation = np.abs(g - neighbour_mean) > 80
    sp = extreme & isolation
    return float(np.count_nonzero(sp)) / gray.size

def analyze_noise(gray):
    sigma = _immerkaer_sigma(gray)
    mean_v = float(np.mean(gray))
    snr = mean_v / (sigma + 1e-6)
    sp_frac = _salt_pepper_fraction(gray)
    h, w = gray.shape
    background_std = float(gray[0:h//5, 0:w//5].astype(np.float32).std())

    s_sigma = _band_score(sigma, NOISE_SIGMA_FAIL, NOISE_SIGMA_WARN, 2.0, False)
    s_snr   = _band_score(snr, SNR_FAIL, SNR_WARN, 50, True)
    s_sp    = 100 - min(100, sp_frac * 5000)
    score = round(0.45*s_sigma + 0.35*s_snr + 0.20*s_sp, 1)
    score = max(0.0, min(100.0, score))

    findings = []
    if sigma >= NOISE_SIGMA_FAIL:
        findings.append(Finding("NOISE.SIGMA.FAIL","fail","noise_sigma",
            round(sigma,2),NOISE_SIGMA_FAIL,"≥",
            f"High noise level (σ = {sigma:.1f})",
            "Cell boundaries obscured by sensor noise."))
    elif sigma >= NOISE_SIGMA_WARN:
        findings.append(Finding("NOISE.SIGMA.WARN","warn","noise_sigma",
            round(sigma,2),NOISE_SIGMA_WARN,"≥",
            f"Moderate noise (σ = {sigma:.1f})",
            "Some texture features may be obscured."))
    else:
        findings.append(Finding("NOISE.SIGMA.PASS","pass","noise_sigma",
            round(sigma,2),NOISE_SIGMA_WARN,"<",
            f"Low noise (σ = {sigma:.1f})",
            "Clean signal."))

    if snr < SNR_FAIL:
        findings.append(Finding("NOISE.SNR.FAIL","fail","snr",
            round(snr,1),SNR_FAIL,"<",
            f"Poor SNR ({snr:.1f})",
            "Signal barely above noise floor."))
    elif snr < SNR_WARN:
        findings.append(Finding("NOISE.SNR.WARN","warn","snr",
            round(snr,1),SNR_WARN,"<",
            f"Marginal SNR ({snr:.1f})",
            "Reduced confidence in pixel-level features."))

    if sp_frac >= SP_FAIL:
        findings.append(Finding("NOISE.SALTPEPPER.FAIL","fail","sp_fraction",
            round(sp_frac,4),SP_FAIL,"≥",
            f"Salt-and-pepper artifacts: {sp_frac*100:.2f}% of pixels",
            "Suggests sensor defects, transmission errors, or compression."))
    elif sp_frac >= SP_WARN:
        findings.append(Finding("NOISE.SALTPEPPER.WARN","warn","sp_fraction",
            round(sp_frac,4),SP_WARN,"≥",
            f"Some salt-and-pepper noise ({sp_frac*100:.2f}%)",
            "Median filter recommended."))

    return MetricResult(
        name="Noise",
        score=score,
        severity=_worst([f.severity for f in findings]),
        measurements={
            "sigma": round(sigma,2),
            "snr": round(snr,2),
            "salt_pepper_pct": round(sp_frac*100,3),
            "background_std": round(background_std,2),
        },
        findings=findings,
    )


# ═══════════ 4. DENSITY (RBC-aware multi-method) ═══════════
def _estimate_background(gray):
    return float(np.percentile(gray, 90))


def _deduplicate_circles(centers, radii, min_dist_factor=0.8):
    """Remove duplicate Hough detections by merging circles with overlapping centres."""
    if len(centers) == 0:
        return centers, radii
    median_r = float(np.median(radii))
    min_dist = median_r * min_dist_factor
    keep = np.ones(len(centers), dtype=bool)
    for i in range(len(centers)):
        if not keep[i]: continue
        for j in range(i + 1, len(centers)):
            if not keep[j]: continue
            d = float(np.linalg.norm(centers[i].astype(float) - centers[j].astype(float)))
            if d < min_dist:
                if radii[i] >= radii[j]: keep[j] = False
                else: keep[i] = False; break
    return centers[keep], radii[keep]


def _detect_cells_hough(gray, h, w):
    """
    Hough Circle Transform tuned for RBCs.
    Higher param2 (accumulator threshold) = fewer false positives.
    minDist = 1.8x min_radius prevents double-counting same cell.
    """
    blurred = cv2.medianBlur(gray, 5)
    min_dim = min(h, w)
    min_r   = max(6,  int(min_dim * 0.013))
    max_r   = max(20, int(min_dim * 0.050))
    min_dist = max(int(min_r * 1.8), 12)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=min_dist,
        param1=90, param2=28,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return np.array([]), np.array([])
    circles = np.around(circles[0]).astype(int)
    centers, radii = circles[:, :2], circles[:, 2]
    return _deduplicate_circles(centers, radii)


def _detect_cells_threshold(gray, h, w, bg_intensity):
    """
    Adaptive threshold + watershed — fallback for non-round / odd-shaped cells.
    Auto-detects whether cells are darker or lighter than background.
    """
    img_area = h * w
    mean_v = float(gray.mean())

    # Decide if cells are darker (typical staining) or lighter than background
    cells_darker = mean_v < bg_intensity - 10

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    if cells_darker:
        thresh = cv2.adaptiveThreshold(
            cv2.GaussianBlur(enhanced, (5, 5), 0),
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=51, C=5
        )
    else:
        thresh = cv2.adaptiveThreshold(
            cv2.GaussianBlur(enhanced, (5, 5), 0),
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=51, C=5
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Filter blobs by size: reject tiny noise and huge background regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
    keep_mask = np.zeros_like(clean)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if (img_area * 0.00005) < area < (img_area * 0.05):
            keep_mask[labels == i] = 255

    return keep_mask


def _detect_rbc_donut(gray, h, w):
    """
    Specialized RBC detector that matches the donut signature:
    a darker ring with a lighter center (the central pallor of biconcave RBCs).
    Uses morphological top-hat to enhance ring structures, then Hough.
    """
    # Top-hat highlights small dark structures (cell boundaries)
    kernel_size = max(15, int(min(h, w) * 0.03))
    if kernel_size % 2 == 0: kernel_size += 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
    # Enhance and detect circles on the cell-boundary signal
    enhanced = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    min_dim = min(h, w)
    min_r = max(6, int(min_dim * 0.012))
    max_r = max(20, int(min_dim * 0.055))
    min_dist = max(8, int(min_r * 1.6))

    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=min_dist,
        param1=70,
        param2=20,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return np.array([]), np.array([])
    circles = np.around(circles[0]).astype(int)
    centers, radii = circles[:, :2], circles[:, 2]
    return _deduplicate_circles(centers, radii)


def analyze_density(gray, original_bgr):
    """
    Multi-method cell density analysis tuned for blood smears.
    Strategy:
      1. Blackhat-enhanced Hough (primary — best for RBC donut shape)
      2. Standard Hough (secondary confirmation)
      3. Adaptive threshold mask for true pixel-level coverage
      4. Touching fraction via nearest-neighbour gap analysis (not overlap)
    """
    h, w = gray.shape
    img_area = h * w

    bg_intensity = _estimate_background(gray)

    # --- METHOD A: blackhat Hough (RBC donuts) — primary ---------
    centers_b, radii_b = _detect_rbc_donut(gray, h, w)
    count_b = len(centers_b)

    # --- METHOD B: standard Hough — secondary --------------------
    centers_a, radii_a = _detect_cells_hough(gray, h, w)
    count_a = len(centers_a)

    # --- METHOD C: threshold mask for coverage -------------------
    keep_mask = _detect_cells_threshold(gray, h, w, bg_intensity)
    coverage_thresh = float(np.mean(keep_mask > 0)) * 100

    # --- Pick best Hough result ----------------------------------
    if count_b >= count_a:
        centers, radii, hough_count, method_used = centers_b, radii_b, count_b, "blackhat_hough"
    else:
        centers, radii, hough_count, method_used = centers_a, radii_a, count_a, "standard_hough"

    if hough_count >= 5:
        cell_count = hough_count
        cell_centers = centers
        cell_radii = radii

        # Coverage: use threshold mask as ground truth for pixel coverage
        # (circle area sum overcounts because circles overlap)
        # Blend: 70% threshold mask, 30% circle estimate for robustness
        circle_area = float(np.sum(np.pi * (radii.astype(float) ** 2)))
        circle_coverage = min(75.0, (circle_area / img_area) * 100)
        coverage = round(0.7 * coverage_thresh + 0.3 * circle_coverage, 2)
        # If threshold failed badly (< 2%), trust circles more
        if coverage_thresh < 2.0 and circle_coverage > 5.0:
            coverage = circle_coverage
    else:
        # Fallback to threshold watershed
        dist = cv2.distanceTransform(keep_mask, cv2.DIST_L2, 5)
        if dist.max() > 0:
            _, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
            n_ws, _ = cv2.connectedComponents(sure_fg.astype(np.uint8))
            cell_count = max(0, n_ws - 1)
        else:
            cell_count = 0
        coverage = coverage_thresh
        method_used = "threshold"
        contours, _ = cv2.findContours(keep_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cell_centers, cell_radii = [], []
        for c in contours:
            area = cv2.contourArea(c)
            if area < img_area * 0.0001: continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            cell_centers.append([int(cx), int(cy)])
            cell_radii.append(int(r))
        cell_centers = np.array(cell_centers) if cell_centers else np.zeros((0,2), int)
        cell_radii   = np.array(cell_radii)   if cell_radii   else np.zeros((0,),  int)

    # --- Touching fraction: centre-distance based -------------------
    # For RBCs, Hough radii are ~10-15% larger than actual cell boundary.
    # Two cells are "truly touching" only if their centres are closer than
    # 1.5 × median_radius (i.e. they genuinely overlap, not just adjacent).
    # Normal well-spread RBCs have centre distances of 1.8-2.5 × radius.
    if len(cell_centers) > 1 and len(cell_radii) > 0:
        median_r = float(np.median(cell_radii))
        # Threshold: centre distance < 1.5 * median_r = genuine overlap
        touch_dist = median_r * 1.5
        touching_count = 0
        n = min(len(cell_centers), 400)
        for i in range(n):
            closest = 9999.0
            for j in range(n):
                if i == j: continue
                d = float(np.linalg.norm(
                    cell_centers[i].astype(float) - cell_centers[j].astype(float)
                ))
                if d < closest:
                    closest = d
            if closest < touch_dist:
                touching_count += 1
        touching_frac = min(1.0, touching_count / n)
    else:
        touching_frac = 0.0

    # --- Spatial CV across 6×6 grid -----------------------------
    grid_r, grid_c = 6, 6
    th, tw = h // grid_r, w // grid_c
    density_grid = np.zeros((grid_r, grid_c), dtype=np.float32)
    for cx, cy in cell_centers:
        density_grid[min(cy // th, grid_r - 1), min(cx // tw, grid_c - 1)] += 1
    g_mean = float(density_grid.mean())
    density_cv = float(density_grid.std() / (g_mean + 1e-6)) if g_mean > 0 else 0.0

    # --- Heatmap visualization ----------------------------------
    if density_grid.max() > 0:
        norm = density_grid / density_grid.max() * 255
    else:
        norm = density_grid
    norm_resized = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.clip(norm_resized, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_INFERNO)
    heatmap_overlay = cv2.addWeighted(original_bgr, 0.55, heatmap_color, 0.45, 0)

    # --- Annotated image ----------------------------------------
    annotated = original_bgr.copy()
    for (cx, cy), r in zip(cell_centers, cell_radii):
        cv2.circle(annotated, (int(cx), int(cy)), int(r), (0, 230, 180), 1, lineType=cv2.LINE_AA)

    if coverage < COVERAGE_SPARSE_FAIL:
        s_cov = 5
    elif coverage < COVERAGE_SPARSE_WARN:
        s_cov = 30 + (coverage - COVERAGE_SPARSE_FAIL) / (COVERAGE_SPARSE_WARN - COVERAGE_SPARSE_FAIL) * 30
    elif coverage <= COVERAGE_DENSE_WARN:
        s_cov = 100
    elif coverage <= COVERAGE_DENSE_FAIL:
        s_cov = 30 + (COVERAGE_DENSE_FAIL - coverage) / (COVERAGE_DENSE_FAIL - COVERAGE_DENSE_WARN) * 30
    else:
        s_cov = 5

    s_cv = _band_score(density_cv, DENSITY_CV_FAIL, DENSITY_CV_WARN, 0.25, False)
    s_touch = _band_score(touching_frac, TOUCHING_FAIL, TOUCHING_WARN, 0.10, False)

    score = round(0.55*s_cov + 0.25*s_cv + 0.20*s_touch, 1)
    score = max(0.0, min(100.0, score))

    findings = []
    if coverage < COVERAGE_SPARSE_FAIL:
        findings.append(Finding("DENSITY.SPARSE.FAIL","fail","coverage_pct",
            round(coverage,2),COVERAGE_SPARSE_FAIL,"<",
            f"Field is essentially empty ({coverage:.1f}% coverage, ~{cell_count} cells)",
            "Insufficient sample for analysis."))
    elif coverage < COVERAGE_SPARSE_WARN:
        findings.append(Finding("DENSITY.SPARSE.WARN","warn","coverage_pct",
            round(coverage,2),COVERAGE_SPARSE_WARN,"<",
            f"Sparse field ({coverage:.1f}% coverage, ~{cell_count} cells)",
            "Larger field of view recommended."))
    elif coverage > COVERAGE_DENSE_FAIL:
        findings.append(Finding("DENSITY.DENSE.FAIL","fail","coverage_pct",
            round(coverage,2),COVERAGE_DENSE_FAIL,">",
            f"Severely overcrowded ({coverage:.1f}% coverage)",
            "Heavy overlap prevents reliable single-cell segmentation."))
    elif coverage > COVERAGE_DENSE_WARN:
        findings.append(Finding("DENSITY.DENSE.WARN","warn","coverage_pct",
            round(coverage,2),COVERAGE_DENSE_WARN,">",
            f"High cell density ({coverage:.1f}% coverage)",
            "Some cell overlap; counting may be less accurate."))
    else:
        findings.append(Finding("DENSITY.OK","pass","coverage_pct",
            round(coverage,2),COVERAGE_DENSE_WARN,"in_range",
            f"Optimal cell density ({coverage:.1f}% coverage, ~{cell_count} cells)",
            "Ideal monolayer for analysis."))

    if density_cv > DENSITY_CV_FAIL:
        findings.append(Finding("DENSITY.UNIFORM.FAIL","fail","spatial_cv",
            round(density_cv,3),DENSITY_CV_FAIL,">",
            f"Cells highly clustered (spatial CV = {density_cv:.2f})",
            "Some grid regions empty, others packed; non-representative."))
    elif density_cv > DENSITY_CV_WARN:
        findings.append(Finding("DENSITY.UNIFORM.WARN","warn","spatial_cv",
            round(density_cv,3),DENSITY_CV_WARN,">",
            f"Uneven cell distribution (CV = {density_cv:.2f})",
            "Avoid sampling local clusters."))

    if touching_frac > TOUCHING_FAIL:
        findings.append(Finding("DENSITY.TOUCHING.FAIL","fail","touching_fraction",
            round(touching_frac,3),TOUCHING_FAIL,">",
            f"{touching_frac*100:.0f}% of cells appear touching/overlapping",
            "Watershed splitting required; counts may be unreliable."))

    metric = MetricResult(
        name="Density",
        score=score,
        severity=_worst([f.severity for f in findings]),
        measurements={
            "coverage_pct": round(coverage,2),
            "cell_count_estimate": int(cell_count),
            "spatial_cv": round(density_cv,3),
            "touching_fraction": round(touching_frac,3),
            "detection_method": method_used,
        },
        findings=findings,
    )
    return metric, annotated, heatmap_overlay


# ═══════════ DECISION ENGINE ═══════════
def make_verdict(metrics, overall_score):
    reasoning = []
    blockers = []
    critical_findings = []
    warn_findings = []
    for key, m in metrics.items():
        for f in m.findings:
            if f.severity == "fail": critical_findings.append((key, f))
            elif f.severity == "warn": warn_findings.append((key, f))

    reasoning.append({
        "step": "1. Aggregate score",
        "detail": f"Weighted overall score = {overall_score:.1f}/100",
        "outcome": ("strong" if overall_score >= SCORE_PASS
                    else "marginal" if overall_score >= SCORE_REVIEW else "weak"),
    })
    reasoning.append({
        "step": "2. Findings audit",
        "detail": f"{len(critical_findings)} critical, {len(warn_findings)} warning",
        "outcome": ("blocking" if critical_findings else "clear"),
    })

    blur_critical = any(f.severity=="fail" for f in metrics["blur"].findings)
    noise_critical = any(f.severity=="fail" for f in metrics["noise"].findings)

    if blur_critical:
        for f in metrics["blur"].findings:
            if f.severity=="fail": blockers.append(f.rule_id)
        reasoning.append({"step":"3. Blur veto",
            "detail":"Critical blur — focus is not recoverable","outcome":"REJECT"})
        return Verdict("REJECT", 0.95, reasoning, blockers)

    if noise_critical and metrics["noise"].score < 30:
        for f in metrics["noise"].findings:
            if f.severity=="fail": blockers.append(f.rule_id)
        reasoning.append({"step":"3. Noise veto",
            "detail":"Critical noise — signal not recoverable","outcome":"REJECT"})
        return Verdict("REJECT", 0.92, reasoning, blockers)

    if len(critical_findings) >= 2:
        for cat, f in critical_findings: blockers.append(f.rule_id)
        reasoning.append({"step":"3. Multiple criticals",
            "detail":f"{len(critical_findings)} critical issues; not safely correctable","outcome":"REJECT"})
        return Verdict("REJECT", 0.90, reasoning, blockers)

    if overall_score < SCORE_REVIEW:
        reasoning.append({"step":"3. Score threshold",
            "detail":f"Score {overall_score:.1f} below reject threshold ({SCORE_REVIEW})","outcome":"REJECT"})
        return Verdict("REJECT", 0.85, reasoning,
            [f.rule_id for _,f in critical_findings])

    if critical_findings or overall_score < SCORE_PASS:
        for cat, f in critical_findings: blockers.append(f.rule_id)
        reasoning.append({"step":"3. Borderline",
            "detail":("One critical issue present" if critical_findings
                      else f"Score in review band ({SCORE_REVIEW} ≤ {overall_score:.1f} < {SCORE_PASS})"),
            "outcome":"REVIEW"})
        confidence = 0.6 + (overall_score - SCORE_REVIEW) / (SCORE_PASS - SCORE_REVIEW) * 0.2
        return Verdict("REVIEW", round(confidence,2), reasoning, blockers)

    reasoning.append({"step":"3. All checks passed",
        "detail":f"No critical issues; score {overall_score:.1f} ≥ {SCORE_PASS}","outcome":"PASS"})
    return Verdict("PASS", round(0.85 + min(0.15, (overall_score-SCORE_PASS)/100), 2), reasoning, [])


# ═══════════ MASTER ═══════════
def analyze_image(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image")
    if len(image_bgr.shape) == 2:
        gray = image_bgr
        image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_m = analyze_blur(gray)
    light_m = analyze_exposure(gray)
    noise_m = analyze_noise(gray)
    dens_m, annotated, heatmap = analyze_density(gray, image_bgr)

    overall = round(
        WEIGHT_BLUR*blur_m.score + WEIGHT_EXPOSURE*light_m.score +
        WEIGHT_NOISE*noise_m.score + WEIGHT_DENSITY*dens_m.score, 1
    )
    metrics = {"blur":blur_m, "lighting":light_m, "noise":noise_m, "density":dens_m}
    verdict = make_verdict(metrics, overall)

    hist = {}
    for ch_idx, ch_name in enumerate(["Blue","Green","Red"]):
        hist[ch_name] = cv2.calcHist([image_bgr],[ch_idx],None,[256],[0,256]).flatten().tolist()

    return QualityReport(
        version=VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        image_info={"width":int(image_bgr.shape[1]),"height":int(image_bgr.shape[0]),"channels":int(image_bgr.shape[2])},
        metrics=metrics,
        overall_score=overall,
        verdict=verdict,
        annotated_image=annotated,
        heatmap_image=heatmap,
        histogram_data=hist,
    )
