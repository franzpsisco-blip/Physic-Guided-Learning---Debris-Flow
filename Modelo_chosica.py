# -*- coding: utf-8 -*-
"""
Modelo_chosica_PGL_corrected_v2.py

Physics-Guided Learning (Debris Flow) 

soil / hydro covariates available as rasters:
  - bulk density
  - sand
  - clay
  - silt
  - rainfall (or wetness proxy)

Everything else (DEM, slope, aspect, NDVI, TWI, relief) is still read from DEM_STACK
(as in your original code). If you do NOT have some of those in DEM_STACK, adjust BAND_MAP.

"""

from __future__ import annotations
import os
import warnings
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

import shap

# Prefer gaussian smoothing (scipy)
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_GAUSS = True
except Exception:
    SCIPY_GAUSS = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ====================== PATHS ======================
# Terrain/covariate stack (multi-band GeoTIFF)
DEM_STACK = r"D:\02_grid-20250906T025757Z-1-001\02_grid\out_rf\Chosica2020_ED_AllParams_on_vU_RF_trim_grid.tif"

# 3D InSAR velocities
VU_PATH   = r"D:\02_grid-20250906T025757Z-1-001\02_grid\out_rf\vU_RF_trim.tif"
VE_PATH   = r"D:\02_grid-20250906T025757Z-1-001\02_grid\out_rf\vE_RF_trim.tif"
VN_PATH   = r"D:\02_grid-20250906T025757Z-1-001\02_grid\out_rf\vN_RF_trim.tif"

# If your soil/rain covariates are NOT inside DEM_STACK, set these paths (GeoTIFFs).
# If you keep them inside DEM_STACK, leave these as None and set BAND_MAP below.
BULK_PATH = None  # e.g. r"D:\covariates\bulk_density.tif"
SAND_PATH = None  # e.g. r"D:\covariates\sand.tif"
CLAY_PATH = None  # e.g. r"D:\covariates\clay.tif"
SILT_PATH = None  # e.g. r"D:\covariates\silt.tif"
RAIN_PATH = None  # e.g. r"D:\covariates\rain_smooth.tif"

OUT_BASE  = r"D:\out_debris_ML_multiclass_PGL_corrected_v2"
RASTERS   = os.path.join(OUT_BASE, "rasters")
FIGURES   = os.path.join(OUT_BASE, "figures")
TRAINING  = os.path.join(OUT_BASE, "training_rasters")
os.makedirs(RASTERS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(TRAINING, exist_ok=True)

# ====================== CLASSES ======================
CLASS_NAMES = ["stable", "slope_movement", "debris_deposition"]
STABLE, SLOPE_MOV, DEP = range(3)
N_CLASSES = 3

# ====================== HYPER / THRESHOLDS ======================
VU_MIN_MOV      = 10.0     # mm/yr
SLOPE_MIN_LS    = 10.0     # deg (paper uses ~10° domain threshold; tune if needed)

Q_SRC      = 82
Q_NORISK   = 20

PINN_HIDDEN_LAYERS = (64, 32)
PINN_LR            = 1e-3
PINN_EPOCHS        = 40
PINN_BATCH_SIZE    = 512

LAMBDA_LOW_MOV      = 6.0
LAMBDA_LS           = 1.2
LAMBDA_DEP_DOMAIN   = 2.2
LAMBDA_FS_SOURCE    = 3.0
LAMBDA_FS_STABLE    = 2.0
LAMBDA_MRVBF_DEP    = 2.5
LAMBDA_TWI_DEP      = 2.5

FS_INSTAB_HIGH = 0.6
FS_INSTAB_LOW  = 0.3

GENTLE_SLOPE_MAX = 18.0
MRVBF_DEP_THR    = 0.55
TWI_DEP_THR      = 0.60

PRIOR_LS_LOGIT_BOOST     = 0.45
PRIOR_DEP_LOGIT_BOOST    = 0.55
PRIOR_STABLE_LOGIT_BOOST = 0.60
DISALLOW_PENALTY         = 3.0

KFOLDS       = 5
RANDOM_STATE = 42

MAX_PER_CLASS = 20_000
MIN_PER_CLASS = 300

MAX_SHAP_SAMPLES = 1500

PROB_SHRINK   = 0.85
SMOOTH_SIGMA  = 1.2
UNCERT_THRESH = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[i] Using device: {DEVICE}")

# ====================== SF CONFIG ======================
# Constant soil depth (since you do not have depth raster)
Z_SOIL_M = 2.0

# water unit weight
GAMMA_W_KN = 9.81

# logistic sharpness for SF_instab
A_SF_LOGIT = 4.0

# ====================== BAND MAP (DEM_STACK) ======================
# Update to match your stack. Set to None if not present.
BAND_MAP = {
    # terrain / vegetation (required for the model)
    "dem": 7,
    "slope_deg": 8,
    "aspect_deg": 10,
    "ndvi": 17,
    "relief_250m": 29,

    # wetness proxies (at least one of rain_or_wetness or twi_raw should exist)
    "rain_or_wetness": 21,
    "twi_raw": 27,

    # soil covariates (set bands if inside DEM_STACK; otherwise keep None and set *_PATH above)
    "bulk_density": None,
    "sand": None,
    "clay": None,
    "silt": None,

    "profile": None,
}

# ====================== IO / UTILS ======================

def read_band(path, band=1):
    with rasterio.open(path) as ds:
        arr = ds.read(band).astype("float32")
        prof = ds.profile
        nod = ds.nodata
    if nod is not None:
        arr = np.where(arr == nod, np.nan, arr)
    arr = np.where(np.abs(arr) > 1e7, np.nan, arr)
    return arr, prof

def read_stack(path, band_map):
    out = {}
    with rasterio.open(path) as ds:
        profile = ds.profile
        nod = ds.nodata
        for k, b in band_map.items():
            if k == "profile":
                continue
            if b is None:
                out[k] = None
                continue
            arr = ds.read(int(b)).astype("float32")
            if nod is not None:
                arr = np.where(arr == nod, np.nan, arr)
            out[k] = arr
    out["profile"] = profile
    return out

def write_gtiff(path, arr, profile, nodata=np.nan):
    prof = profile.copy()
    prof.update(count=1, dtype="float32", nodata=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)
    print(f"[+] WRITE GTiff: {path}")

def robust_norm01(x, qlo=5, qhi=95, mask=None):
    if x is None:
        return None
    if mask is None:
        v = x[np.isfinite(x)]
    else:
        v = x[mask & np.isfinite(x)]
    if v.size == 0:
        out = np.zeros_like(x, dtype="float32")
        out[~np.isfinite(x)] = np.nan
        return out
    lo, hi = np.percentile(v, [qlo, qhi])
    hi = hi + 1e-6
    xx = np.clip(x, lo, hi)
    out = (xx - lo) / (hi - lo)
    out[~np.isfinite(x)] = np.nan
    return out.astype("float32")

def smooth_prob_map(prob_map, sigma=1.2):
    if prob_map is None:
        return prob_map
    finite = np.isfinite(prob_map)
    if finite.sum() == 0:
        return prob_map.astype("float32")
    arr = np.where(finite, prob_map, 0.0).astype("float32")
    if SCIPY_GAUSS:
        a = gaussian_filter(arr, sigma=sigma, mode="nearest")
        m = gaussian_filter(finite.astype("float32"), sigma=sigma, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(m > 0, a / m, np.nan).astype("float32")
        return out
    return prob_map.astype("float32")

def shrink_probs(p, alpha=0.85):
    return (0.5 + (p - 0.5) * alpha).astype("float32")

def softmax_np(z, axis=-1):
    z = z - np.nanmax(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    s = np.nansum(ez, axis=axis, keepdims=True) + 1e-12
    return (ez / s).astype("float32")

def probs_to_logits(p):
    p = np.clip(p, 1e-6, 1.0)
    return np.log(p).astype("float32")

# ====================== MRVBF (APPROX) ======================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def mrvbf_approx(dem, slope_deg, mask, sigmas=(2,4,8,16,32,64), slope_thr=8.0):
    if dem is None or slope_deg is None:
        return None
    if not SCIPY_GAUSS:
        m = mask & np.isfinite(slope_deg)
        raw = np.zeros_like(slope_deg, dtype="float32")
        raw[m] = np.clip(1.0 - slope_deg[m] / 30.0, 0.0, 1.0)
        raw[~mask] = np.nan
        return raw.astype("float32")

    dem_f = dem.astype("float32").copy()
    slp_f = slope_deg.astype("float32").copy()
    dem_f[~mask] = np.nan
    slp_f[~mask] = np.nan

    med_dem = np.nanmedian(dem_f[mask])
    med_slp = np.nanmedian(slp_f[mask])
    dem_fill = np.where(mask & np.isfinite(dem_f), dem_f, med_dem).astype("float32")
    slp_fill = np.where(mask & np.isfinite(slp_f), slp_f, med_slp).astype("float32")

    acc = np.zeros_like(dem_fill, dtype="float32")
    wsum = 0.0

    for s in sigmas:
        slp_s = gaussian_filter(slp_fill, sigma=float(s), mode="nearest")
        dem_s = gaussian_filter(dem_fill, sigma=float(s), mode="nearest")
        dem2_s = gaussian_filter(dem_fill * dem_fill, sigma=float(s), mode="nearest")
        var = np.clip(dem2_s - dem_s * dem_s, 0.0, None)
        std = np.sqrt(var + 1e-6).astype("float32")

        flat = sigmoid((slope_thr - slp_s) / (0.75 + 0.02 * slope_thr)).astype("float32")
        low  = sigmoid((dem_s - dem_fill) / (std + 1e-6)).astype("float32")

        score = (flat * low).astype("float32")
        w = float(np.log2(s + 1.0))
        acc += w * score
        wsum += w

    raw = acc / (wsum + 1e-6)
    raw[~mask] = np.nan
    return robust_norm01(raw, 5, 95, mask=mask)

# ====================== SOIL COVARIATES HELPERS ======================

def _to_fraction(x, mask):
    """Detect 0–100 vs 0–1."""
    if x is None:
        return None
    v = x[mask & np.isfinite(x)]
    if v.size == 0:
        return None
    med = float(np.nanmedian(v))
    if med > 1.5:
        return (x / 100.0).astype("float32")
    return x.astype("float32")

def _to_bulk_density_kgm3(bd, mask):
    """Detect g/cm3 vs kg/m3."""
    if bd is None:
        return None
    v = bd[mask & np.isfinite(bd)]
    if v.size == 0:
        return None
    med = float(np.nanmedian(v))
    if med < 10.0:  # likely g/cm3
        return (bd * 1000.0).astype("float32")
    return bd.astype("float32")

def load_soil_covariates_from_paths(profile_ref, mask, bulk_path, sand_path, clay_path, silt_path, rain_path):
    """
    Reads soil/rain rasters from external paths and checks they match the reference grid.
    """
    def _read_and_check(p):
        if p is None:
            return None
        arr, prof = read_band(p, 1)
        # basic grid check
        if (prof["crs"] != profile_ref["crs"] or
            prof["transform"] != profile_ref["transform"] or
            prof["width"] != profile_ref["width"] or
            prof["height"] != profile_ref["height"]):
            raise ValueError(f"[!] Raster grid mismatch vs DEM_STACK: {p}")
        arr[~mask] = np.nan
        return arr.astype("float32")

    bulk = _read_and_check(bulk_path)
    sand = _read_and_check(sand_path)
    clay = _read_and_check(clay_path)
    silt = _read_and_check(silt_path)
    rain = _read_and_check(rain_path)
    return bulk, sand, clay, silt, rain

def derive_geotech_from_bulk_texture(bulk, sand, clay, silt, mask):
    """
    Derive spatial gamma_s (kN/m3), phi (deg), c (kPa) using ONLY bulk + sand/clay/silt.

    This is a smooth, bounded proxy meant to avoid SF artifacts:
    - gamma_s from bulk density
    - phi increases with sand, decreases with clay
    - c increases with clay

    If you have calibrated relationships, replace this function.
    """
    rho = _to_bulk_density_kgm3(bulk, mask)
    if rho is None:
        # fallback: typical range
        gamma_s = np.full_like(clay if clay is not None else sand, 18.0, dtype="float32")
    else:
        gamma_s = (rho * 9.81 / 1000.0).astype("float32")
        gamma_s = np.clip(gamma_s, 12.0, 22.0).astype("float32")

    sand_f = _to_fraction(sand, mask)
    clay_f = _to_fraction(clay, mask)
    silt_f = _to_fraction(silt, mask)

    if sand_f is None:
        sand_f = np.zeros_like(gamma_s, dtype="float32") + 0.40
    if clay_f is None:
        clay_f = np.zeros_like(gamma_s, dtype="float32") + 0.25
    if silt_f is None:
        silt_f = np.clip(1.0 - (sand_f + clay_f), 0.0, 1.0).astype("float32")

    # friction angle proxy (bounded)
    phi = 20.0 + 18.0 * sand_f - 8.0 * clay_f
    phi = np.clip(phi, 15.0, 40.0).astype("float32")

    # cohesion proxy (bounded)
    c = 2.0 + 12.0 * clay_f
    c = np.clip(c, 0.5, 20.0).astype("float32")

    for arr in (gamma_s, phi, c):
        arr[~mask] = np.nan
    return gamma_s, phi, c

def compute_sf_and_instab(slope_deg, m_sat01, gamma_s_knm3, phi_deg, c_kpa, z_soil_m, mask):
    """
    Infinite-slope Mohr–Coulomb:
      sigma_n = gamma_s z cos^2(theta)
      u       = gamma_w m z cos^2(theta)
      tau_d   = gamma_s z sin(theta) cos(theta)
      tau_r   = c + (sigma_n - u) tan(phi)
      SF      = tau_r / tau_d
      SF_instab = 1/(1+exp(a*(SF-1)))
    """
    theta = np.deg2rad(slope_deg.astype("float32"))
    cos2 = (np.cos(theta) ** 2).astype("float32")
    sincos = (np.sin(theta) * np.cos(theta)).astype("float32")

    z = np.full_like(slope_deg, float(z_soil_m), dtype="float32")

    sigma_n = (gamma_s_knm3 * z * cos2).astype("float32")
    u       = (GAMMA_W_KN * m_sat01 * z * cos2).astype("float32")
    tau_d   = (gamma_s_knm3 * z * sincos).astype("float32")

    phi = np.deg2rad(phi_deg.astype("float32"))
    tau_r = (c_kpa + (sigma_n - u) * np.tan(phi)).astype("float32")

    with np.errstate(invalid="ignore", divide="ignore"):
        sf = tau_r / (tau_d + 1e-6)
    sf[~mask] = np.nan

    sf_clip = np.clip(sf, 0.1, 5.0).astype("float32")
    with np.errstate(invalid="ignore"):
        instab = 1.0 / (1.0 + np.exp(A_SF_LOGIT * (sf_clip - 1.0)))
    instab[~mask] = np.nan
    return sf.astype("float32"), instab.astype("float32")

# ====================== SHAP PLOTS ======================

def beeswarm_axis_shap(ax, shap_vals, X, feat_names, max_display=15, cmap_name="coolwarm"):
    if shap_vals.ndim != 2 or X.ndim != 2:
        raise ValueError("shap_vals and X must be 2D arrays.")
    if shap_vals.shape[0] != X.shape[0]:
        if shap_vals.shape[1] == X.shape[0] and shap_vals.shape[0] == X.shape[1]:
            shap_vals = shap_vals.T
        else:
            raise ValueError(f"shap_vals shape {shap_vals.shape} incompatible with X shape {X.shape}.")

    shap_abs_mean = np.mean(np.abs(shap_vals), axis=0)

    vel_names = ["vU","vE","vN","vU_abs","vabs","v_par","vh"]
    vel_idx = [feat_names.index(n) for n in vel_names if n in feat_names]
    all_idx = np.argsort(shap_abs_mean)[::-1].tolist()
    other_idx = [i for i in all_idx if i not in vel_idx]
    chosen_idx = (vel_idx + other_idx)[:max_display]
    chosen_names = [feat_names[i] for i in chosen_idx]

    cmap = plt.get_cmap(cmap_name)
    rng = np.random.default_rng(0)
    x_min, x_max = 0.0, 0.0

    for row, j in enumerate(chosen_idx):
        sv = shap_vals[:, j]
        fv = X[:, j]
        y = np.full_like(sv, row, dtype="float32") + rng.uniform(-0.35, 0.35, size=sv.size)
        vmin, vmax = np.nanpercentile(fv, [5, 95])
        if vmax <= vmin:
            vmax = vmin + 1e-6
        normed = np.clip((fv - vmin) / (vmax - vmin), 0, 1)
        ax.scatter(sv, y, c=cmap(normed), s=6, alpha=0.65, linewidths=0, rasterized=True)
        x_min = min(x_min, float(np.nanmin(sv)))
        x_max = max(x_max, float(np.nanmax(sv)))

    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(range(len(chosen_idx)))
    ax.set_yticklabels(chosen_names)
    ax.invert_yaxis()
    ax.set_xlabel(f"SHAP value (impact) / class = {CLASS_NAMES[SLOPE_MOV]}")
    ax.set_xlim(x_min * 1.1, x_max * 1.1)

def plot_global_roc_pr_shap_multiclass(info, feat_names, out_path,
                                       X_shap_for_plot, shap_vals_for_plot,
                                       use_ambiguous_only=True,
                                       ROC_PR_LOWER=0.2, ROC_PR_UPPER=0.8, ROC_PR_MIN_PIX=500, ROC_JITTER_STD=0.02):
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    colors = {"stable": "#8fa1cb", "slope_movement": "#c48788", "debris_deposition": "#c287b5"}

    y_oof = info["y_oof"]
    proba_oof = info["proba_oof"]
    maxp = proba_oof.max(axis=1)

    if use_ambiguous_only:
        mask_eval = (maxp > ROC_PR_LOWER) & (maxp < ROC_PR_UPPER)
        if mask_eval.sum() < ROC_PR_MIN_PIX:
            mask_eval = np.ones_like(maxp, dtype=bool)
    else:
        mask_eval = np.ones_like(maxp, dtype=bool)

    y_eval = y_oof[mask_eval]
    proba_eval = proba_oof[mask_eval]
    y_bin = label_binarize(y_eval, classes=np.arange(N_CLASSES))
    rng_local = np.random.RandomState(0)

    # ROC
    ax_roc = axes[0]
    for cls in range(N_CLASSES):
        name = CLASS_NAMES[cls]
        y_true = y_bin[:, cls]
        if int(y_true.sum()) < 5:
            continue
        scores = proba_eval[:, cls].copy()
        if ROC_JITTER_STD > 0:
            scores += rng_local.normal(0.0, ROC_JITTER_STD, size=scores.shape)
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_cls = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2, color=colors[name], label=f"{name} (AUC = {auc_cls:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("False positive rate"); ax_roc.set_ylabel("True positive rate")
    ax_roc.legend(frameon=False, loc="lower right")

    # PR
    ax_pr = axes[1]
    for cls in range(N_CLASSES):
        name = CLASS_NAMES[cls]
        y_true = y_bin[:, cls]
        if int(y_true.sum()) < 5:
            continue
        scores = proba_eval[:, cls].copy()
        if ROC_JITTER_STD > 0:
            scores += rng_local.normal(0.0, ROC_JITTER_STD, size=scores.shape)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap_cls = average_precision_score(y_true, scores)
        ax_pr.plot(rec, prec, lw=2, color=colors[name], label=f"{name} (AP = {ap_cls:.3f})")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1)
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.legend(frameon=False, loc="lower left")

    # SHAP
    ax_shap = axes[2]
    beeswarm_axis_shap(ax_shap, shap_vals_for_plot, X_shap_for_plot, feat_names, max_display=15)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] FIGURE saved: {out_path}")

# ====================== MODEL ======================

class PhysicsGuidedMLP(nn.Module):
    def __init__(self, in_features, hidden_layers, n_classes):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def predict_proba(model, X_np, device=DEVICE, batch_size=4096):
    model.eval()
    X_np = X_np.astype("float32")
    out = []
    with torch.no_grad():
        for i in range(0, X_np.shape[0], batch_size):
            xb = torch.from_numpy(X_np[i:i+batch_size]).to(device)
            logits = model(xb)
            pb = torch.softmax(logits, dim=1)
            out.append(pb.cpu().numpy())
    if len(out) == 0:
        return np.empty((0, N_CLASSES), dtype="float32")
    return np.vstack(out).astype("float32")

def train_single_model(X, y, feat_names, n_epochs=PINN_EPOCHS, batch_size=PINN_BATCH_SIZE, lr=PINN_LR, device=DEVICE):
    n_samples, n_features = X.shape
    model = PhysicsGuidedMLP(n_features, PINN_HIDDEN_LAYERS, N_CLASSES).to(device)

    X_tensor = torch.from_numpy(X.astype("float32")).to(device)
    y_tensor = torch.from_numpy(y.astype("int64")).to(device)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    idx_vu_abs = feat_names.index("vU_abs")
    idx_slope  = feat_names.index("slope")
    idx_fs     = feat_names.index("SF_instab")
    idx_twi    = feat_names.index("TWI_n")
    idx_mrvbf  = feat_names.index("MRVBF_n")

    model.train()
    for epoch in range(1, n_epochs + 1):
        tot = tot_data = tot_phys = 0.0

        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)

            loss_data = ce(logits, yb)

            vU_abs_b = xb[:, idx_vu_abs]
            slope_b  = xb[:, idx_slope]
            FS_b     = xb[:, idx_fs]
            TWI_b    = xb[:, idx_twi]
            MRVBF_b  = xb[:, idx_mrvbf]

            P_st = probs[:, STABLE]
            P_ls = probs[:, SLOPE_MOV]
            P_dp = probs[:, DEP]

            loss_phys = torch.tensor(0.0, device=device)

            low_mov = (vU_abs_b < VU_MIN_MOV)
            if low_mov.any():
                loss_phys = loss_phys + LAMBDA_LOW_MOV * (P_ls[low_mov] + P_dp[low_mov]).mean()

            moving = (vU_abs_b >= VU_MIN_MOV)
            ls_dom = moving & (slope_b >= SLOPE_MIN_LS)
            if ls_dom.any():
                loss_phys = loss_phys + LAMBDA_LS * torch.relu(P_dp[ls_dom] - P_ls[ls_dom]).mean()

            gentle = (slope_b <= GENTLE_SLOPE_MAX)
            fs_low = (FS_b <= FS_INSTAB_LOW)
            mrvbf_hi = (MRVBF_b >= MRVBF_DEP_THR)
            twi_hi   = (TWI_b   >= TWI_DEP_THR)

            dep_dom = moving & gentle & fs_low & (mrvbf_hi | twi_hi)
            if dep_dom.any():
                dep_push = (torch.relu(0.55 - P_dp[dep_dom]) + torch.relu(P_ls[dep_dom] - P_dp[dep_dom])).mean()
                loss_phys = loss_phys + LAMBDA_DEP_DOMAIN * dep_push
                loss_phys = loss_phys + LAMBDA_MRVBF_DEP * torch.relu(0.55 - P_dp[dep_dom]).mean()
                loss_phys = loss_phys + LAMBDA_TWI_DEP   * torch.relu(0.55 - P_dp[dep_dom]).mean()

            fs_hi = (FS_b >= FS_INSTAB_HIGH)
            if fs_hi.any():
                loss_fs = (torch.relu(0.60 - P_ls[fs_hi]) + torch.relu(P_st[fs_hi] - P_ls[fs_hi])).mean()
                loss_phys = loss_phys + LAMBDA_FS_SOURCE * loss_fs

            fs_lo = (FS_b <= FS_INSTAB_LOW)
            if fs_lo.any():
                loss_fs = (torch.relu(0.70 - P_st[fs_lo]) + (P_ls[fs_lo] + P_dp[fs_lo])).mean()
                loss_phys = loss_phys + LAMBDA_FS_STABLE * loss_fs

            loss = loss_data + loss_phys
            loss.backward()
            opt.step()

            bs = xb.size(0)
            tot += loss.item() * bs
            tot_data += loss_data.item() * bs
            tot_phys += loss_phys.item() * bs

        print(f"[epoch {epoch:03d}] loss={tot/n_samples:.4f}  data={tot_data/n_samples:.4f}  phys={tot_phys/n_samples:.4f}")

    return model

def train_with_cv_and_shap(X, y, feat_names, n_splits=KFOLDS, n_shap=MAX_SHAP_SAMPLES):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    n = X.shape[0]
    proba_oof = np.zeros((n, N_CLASSES), dtype="float32")

    for k, (tr, te) in enumerate(skf.split(X, y), start=1):
        print(f"[i] Fold {k}/{n_splits} ...")
        m = train_single_model(X[tr], y[tr], feat_names)
        proba_oof[te, :] = predict_proba(m, X[te])

    print("[i] Training final model on full dataset...")
    clf_final = train_single_model(X, y, feat_names)

    n_shap = min(n_shap, X.shape[0])
    rng = np.random.RandomState(RANDOM_STATE)
    n_bg = max(50, min(200, X.shape[0] // 10))
    idx_bg = rng.choice(X.shape[0], size=n_bg, replace=False)
    idx_sh = rng.choice(X.shape[0], size=n_shap, replace=False)
    X_bg = X[idx_bg]
    X_shap = X[idx_sh]

    def f_scalar(z):
        return predict_proba(clf_final, z)[:, SLOPE_MOV]

    print("[i] Computing Kernel SHAP for slope_movement ...")
    explainer = shap.KernelExplainer(f_scalar, X_bg)
    shap_vals_raw = np.array(explainer.shap_values(X_shap, nsamples=200))

    if shap_vals_raw.ndim != 2:
        raise ValueError(f"[!] Unexpected SHAP array shape: {shap_vals_raw.shape}")
    shap_vals = shap_vals_raw.astype("float32")
    if shap_vals.shape != X_shap.shape and shap_vals.T.shape == X_shap.shape:
        shap_vals = shap_vals.T.astype("float32")

    info = dict(proba_oof=proba_oof, y_oof=y, X_shap=X_shap.astype("float32"), shap_vals=shap_vals)
    return clf_final, info

# ====================== MAIN PIPELINE ======================

print("[+] === LOAD & ALIGN ===")

stack = read_stack(DEM_STACK, BAND_MAP)
profile = stack["profile"]

dem        = stack["dem"]
slope      = stack["slope_deg"]
aspect_deg = stack["aspect_deg"]
ndvi       = stack["ndvi"]
relief250  = stack["relief_250m"]
rain_stack = stack["rain_or_wetness"]
twi_raw    = stack["twi_raw"]

if dem is None or slope is None or aspect_deg is None or ndvi is None:
    raise ValueError("[!] DEM_STACK is missing required terrain/vegetation bands. Update BAND_MAP.")

vU, _ = read_band(VU_PATH)
vE, _ = read_band(VE_PATH)
vN, _ = read_band(VN_PATH)

mask_insar = np.isfinite(vU) & np.isfinite(vE) & np.isfinite(vN)
mask_morph = np.isfinite(dem) & np.isfinite(slope)
mask_basic = mask_insar & mask_morph
print(f"[i] InSAR valid (3 comps): {mask_insar.sum()} / {mask_insar.size} ({100*mask_insar.mean():.2f}%)")

# Load soil/rain covariates
bulk = stack["bulk_density"]
sand = stack["sand"]
clay = stack["clay"]
silt = stack["silt"]
rain = rain_stack

if BULK_PATH or SAND_PATH or CLAY_PATH or SILT_PATH or RAIN_PATH:
    bulk2, sand2, clay2, silt2, rain2 = load_soil_covariates_from_paths(profile, mask_morph, BULK_PATH, SAND_PATH, CLAY_PATH, SILT_PATH, RAIN_PATH)
    bulk = bulk2 if bulk2 is not None else bulk
    sand = sand2 if sand2 is not None else sand
    clay = clay2 if clay2 is not None else clay
    silt = silt2 if silt2 is not None else silt
    rain = rain2 if rain2 is not None else rain

# Safety: require rain + at least one soil property (bulk or texture)
if rain is None:
    raise ValueError("[!] Missing rainfall/wetness raster (rain_or_wetness band or RAIN_PATH).")
if bulk is None and sand is None and clay is None and silt is None:
    raise ValueError("[!] Missing soil rasters (bulk/sand/clay/silt). Provide via BAND_MAP or *_PATH.")

# Kinematics
vU = np.where(mask_insar, vU, np.nan).astype("float32")
vE = np.where(mask_insar, vE, np.nan).astype("float32")
vN = np.where(mask_insar, vN, np.nan).astype("float32")
vh   = np.hypot(vE, vN).astype("float32")
vabs = np.hypot(vh, vU).astype("float32")
vU_abs = np.abs(vU).astype("float32")

aspect_rad = np.deg2rad(aspect_deg.astype("float32"))
ux = np.sin(aspect_rad).astype("float32")
uy = np.cos(aspect_rad).astype("float32")
v_par = (vE * ux + vN * uy).astype("float32")
cosAsp = np.cos(aspect_rad).astype("float32")
sinAsp = np.sin(aspect_rad).astype("float32")

# Percentiles for seeds
vU_abs_q = np.percentile(vU_abs[np.isfinite(vU_abs)], [Q_NORISK, Q_SRC, 95])
vh_q = np.percentile(vh[np.isfinite(vh)], [30, 80])
p_norisk, p_src_th, p95_abs = vU_abs_q
vh_p30, vh_p80 = vh_q

# Saturation proxy m in [0,1]
print("[+] m_sat (rain/wetness -> [0,1]) ...")
m_sat = robust_norm01(rain, 5, 95, mask=mask_morph)
m_sat[~mask_morph] = np.nan
write_gtiff(os.path.join(RASTERS, "m_sat_norm01.tif"), m_sat, profile)

# Geotech (ONLY bulk + texture)
print("[+] Geotech from bulk + sand/clay/silt ...")
gamma_s, phi_deg, c_kpa = derive_geotech_from_bulk_texture(bulk, sand, clay, silt, mask_morph)
write_gtiff(os.path.join(RASTERS, "gamma_s_knm3.tif"), gamma_s, profile)
write_gtiff(os.path.join(RASTERS, "phi_eff_deg.tif"), phi_deg, profile)
write_gtiff(os.path.join(RASTERS, "c_eff_kpa.tif"), c_kpa, profile)

# SF + SF_instab
print("[+] SF + SF_instab (infinite slope) ...")
SF_raw, SF_instab = compute_sf_and_instab(slope, m_sat, gamma_s, phi_deg, c_kpa, Z_SOIL_M, mask_morph)
write_gtiff(os.path.join(RASTERS, "SF_raw.tif"), SF_raw, profile)
write_gtiff(os.path.join(RASTERS, "SF_instab.tif"), SF_instab, profile)

# TWI
print("[+] TWI (normalized) ...")
TWI_n = robust_norm01(twi_raw, 5, 95, mask=mask_morph) if twi_raw is not None else None
if TWI_n is None:
    # fallback: use m_sat as wetness proxy if TWI not available
    TWI_n = m_sat.copy()
TWI_n[~mask_morph] = np.nan
write_gtiff(os.path.join(RASTERS, "TWI_norm01.tif"), TWI_n, profile)

# MRVBF
print("[+] MRVBF (approx) ...")
MRVBF_n = mrvbf_approx(dem, slope, mask=mask_morph)
MRVBF_n[~mask_morph] = np.nan
write_gtiff(os.path.join(RASTERS, "MRVBF_norm01.tif"), MRVBF_n, profile)

# ====================== PHYSICS SEEDS ======================
mov_mask = mask_basic & (vU_abs >= VU_MIN_MOV)
strong_mov = mov_mask & ((vU_abs >= p_src_th) | (vh >= vh_p80))

slope_movement_seed = strong_mov & (slope >= SLOPE_MIN_LS)

dep_seed = strong_mov & (slope < SLOPE_MIN_LS)
dep_seed = dep_seed & (slope <= (GENTLE_SLOPE_MAX + 5.0)) & (SF_instab <= FS_INSTAB_LOW)
dep_seed = dep_seed & ((TWI_n >= 0.5) | (MRVBF_n >= 0.5))

stable_seed = mask_basic & (vU_abs <= p_norisk) & (vh <= vh_p30)
stable_seed &= ~(slope_movement_seed | dep_seed)

write_gtiff(os.path.join(TRAINING, "seed_slope_movement.tif"), slope_movement_seed.astype("float32"), profile, nodata=0)
write_gtiff(os.path.join(TRAINING, "seed_debris_deposition.tif"), dep_seed.astype("float32"), profile, nodata=0)
write_gtiff(os.path.join(TRAINING, "seed_stable.tif"), stable_seed.astype("float32"), profile, nodata=0)

# ====================== FEATURE STACK ======================
features_list = [
    vU, vE, vN, vU_abs, vabs, v_par, vh,
    slope, dem,
    ndvi, m_sat, relief250,
    SF_instab, TWI_n, MRVBF_n,
    cosAsp, sinAsp
]
feat_names = [
    "vU","vE","vN","vU_abs","vabs","v_par","vh",
    "slope","dem",
    "NDVI","m_sat","Relief250m",
    "SF_instab","TWI_n","MRVBF_n",
    "cosAsp","sinAsp"
]

F_raw = np.stack(features_list, axis=-1).astype("float32")
H, W, C = F_raw.shape

# Impute NaN inside mask_insar with band median
F = F_raw.copy()
for c in range(C):
    band = F[..., c]
    m = np.isfinite(band) & mask_insar
    if m.any():
        med = np.nanmedian(band[m])
        band[(~np.isfinite(band)) & mask_insar] = med
        F[..., c] = band

valid_mask = mask_insar.copy()
flatF = F.reshape(-1, C)

# ====================== TRAIN SET FROM SEEDS ======================
y_full = np.full(H * W, -1, dtype="int16")
y_full[stable_seed.ravel()] = STABLE
y_full[slope_movement_seed.ravel()] = SLOPE_MOV
y_full[dep_seed.ravel()] = DEP

label_mask = (y_full >= 0) & valid_mask.ravel()
idx_all = np.where(label_mask)[0]
y_all = y_full[idx_all]
print(f"[i] Labeled valid pixels: {idx_all.size}")

rng = np.random.RandomState(RANDOM_STATE)
train_idx = []
for cls in range(N_CLASSES):
    cls_idx = idx_all[y_all == cls]
    n_cls = cls_idx.size
    if n_cls == 0:
        print(f"[!] WARNING: class {CLASS_NAMES[cls]} has 0 seeds.")
        continue
    if cls == STABLE:
        sel = cls_idx if n_cls >= MIN_PER_CLASS else rng.choice(cls_idx, size=MIN_PER_CLASS, replace=True)
    else:
        target = max(MIN_PER_CLASS, min(MAX_PER_CLASS, int(0.35 * n_cls)))
        sel = rng.choice(cls_idx, size=target, replace=(target > n_cls))
    train_idx.append(sel)
    print(f"[i] Class {CLASS_NAMES[cls]} -> {len(sel)} training pixels (seeds={n_cls})")
train_idx = np.concatenate(train_idx)
rng.shuffle(train_idx)

X_train = flatF[train_idx]
y_train = y_full[train_idx]
print(f"[i] Dataset -> X={X_train.shape}, y={y_train.shape}")

# ====================== TRAIN + SHAP ======================
print("[+] Training PG-MLP + SHAP ...")
clf, info = train_with_cv_and_shap(X_train, y_train, feat_names, n_splits=KFOLDS, n_shap=MAX_SHAP_SAMPLES)

# ====================== INFERENCE ======================
print("[+] Inference to raster ...")
flat_all = F.reshape(-1, C)
probs = np.full((H * W, N_CLASSES), np.nan, dtype="float32")

idx_valid = np.where(valid_mask.ravel())[0]
p = predict_proba(clf, flat_all[idx_valid])
p = shrink_probs(p, alpha=PROB_SHRINK)
probs[idx_valid, :] = p
probs = probs.reshape(H, W, N_CLASSES)

# ====================== DOMAIN PRIORS (LOGIT SPACE) + RE-SOFTMAX ======================
low_mov = (vU_abs < VU_MIN_MOV) & valid_mask
ls_domain = (valid_mask & (vU_abs >= VU_MIN_MOV) & (slope >= SLOPE_MIN_LS))
dep_domain = (valid_mask & (vU_abs >= VU_MIN_MOV) &
              (slope <= (GENTLE_SLOPE_MAX + 5.0)) &
              (SF_instab <= FS_INSTAB_LOW) &
              ((TWI_n >= TWI_DEP_THR) | (MRVBF_n >= MRVBF_DEP_THR)))

logits = probs_to_logits(np.where(np.isfinite(probs), probs, np.nan))
mask_logits = np.isfinite(logits).all(axis=2)
uniform_logits = np.log(np.full(N_CLASSES, 1.0 / N_CLASSES, dtype="float32"))
logits[~mask_logits] = uniform_logits

logits[low_mov, STABLE] += PRIOR_STABLE_LOGIT_BOOST
logits[low_mov, SLOPE_MOV] -= DISALLOW_PENALTY
logits[low_mov, DEP] -= DISALLOW_PENALTY

logits[ls_domain, SLOPE_MOV] += PRIOR_LS_LOGIT_BOOST
logits[ls_domain, DEP] -= 0.5 * DISALLOW_PENALTY

logits[dep_domain, DEP] += PRIOR_DEP_LOGIT_BOOST
logits[dep_domain, SLOPE_MOV] -= 0.5 * DISALLOW_PENALTY

probs = softmax_np(logits, axis=2)

for k in range(N_CLASSES):
    probs[:, :, k] = smooth_prob_map(probs[:, :, k], sigma=SMOOTH_SIGMA)
    probs[:, :, k][~valid_mask] = np.nan

# enforce low motion stable
probs[low_mov, STABLE] = 1.0
probs[low_mov, SLOPE_MOV] = 0.0
probs[low_mov, DEP] = 0.0

prob_stable = probs[:, :, STABLE]
prob_slope  = probs[:, :, SLOPE_MOV]
prob_dep    = probs[:, :, DEP]

write_gtiff(os.path.join(RASTERS, "prob_stable.tif"), prob_stable, profile)
write_gtiff(os.path.join(RASTERS, "prob_slope_movement.tif"), prob_slope, profile)
write_gtiff(os.path.join(RASTERS, "prob_debris_deposition.tif"), prob_dep, profile)

# ====================== CLASS MAP + UNCERTAINTY ======================
probs_filled = np.where(np.isfinite(probs), probs, -1.0)
max_prob = np.max(probs_filled, axis=2)
class_idx = np.argmax(probs_filled, axis=2).astype("float32")
uncertainty = np.clip(1.0 - max_prob, 0, 1).astype("float32")

class_idx[~valid_mask] = np.nan
uncertainty[~valid_mask] = np.nan

uncertainty_mask = (max_prob < UNCERT_THRESH).astype("float32")
uncertainty_mask[~valid_mask] = np.nan
class_idx[max_prob < UNCERT_THRESH] = np.nan

write_gtiff(os.path.join(RASTERS, "class_map.tif"), class_idx, profile)
write_gtiff(os.path.join(RASTERS, "uncertainty_1minusmax.tif"), uncertainty, profile)
write_gtiff(os.path.join(RASTERS, "uncertainty_mask.tif"), uncertainty_mask, profile)

# Entropy
probs_norm = probs / (np.nansum(probs, axis=2, keepdims=True) + 1e-6)
with np.errstate(invalid="ignore"):
    entropy = -np.nansum(probs_norm * np.log(probs_norm + 1e-6), axis=2)
entropy[~valid_mask] = np.nan
write_gtiff(os.path.join(RASTERS, "uncertainty_entropy.tif"), entropy.astype("float32"), profile)

# ====================== ROC/PR/SHAP FIGURES ======================
plot_global_roc_pr_shap_multiclass(
    info, feat_names,
    os.path.join(FIGURES, "global_ROC_PR_SHAP_multiclass_ALL.png"),
    X_shap_for_plot=info["X_shap"],
    shap_vals_for_plot=info["shap_vals"],
    use_ambiguous_only=False
)

plot_global_roc_pr_shap_multiclass(
    info, feat_names,
    os.path.join(FIGURES, "global_ROC_PR_SHAP_multiclass_AMBIG.png"),
    X_shap_for_plot=info["X_shap"],
    shap_vals_for_plot=info["shap_vals"],
    use_ambiguous_only=True
)

# ====================== SUMMARY ======================
res_x = abs(profile["transform"][0])
res_y = abs(profile["transform"][4])
px_to_km2 = (res_x * res_y) / 1e6

area_ls  = np.nansum(prob_slope >= 0.5) * px_to_km2
area_dep = np.nansum(prob_dep   >= 0.5) * px_to_km2
area_st  = np.nansum(prob_stable>= 0.5) * px_to_km2

print("\n=== SUMMARY (p>=0.5) ===")
print(f"Seeds -> slope_movement={int(slope_movement_seed.sum())}  deposition={int(dep_seed.sum())}  stable={int(stable_seed.sum())}")
print(f"AREAS >=0.5 (km²) -> Slope-movement={area_ls:.2f}  Deposition={area_dep:.2f}  Stable={area_st:.2f}")
print("Rasters  ->", RASTERS)
print("Training ->", TRAINING)
print("Figures  ->", FIGURES)
print("[✓] Finished.")
