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

# ====================== PATHS ======================

DEM_STACK = r"D:\02_grid-20250906T025757Z-1-001\02_grid\out_rf\Chosica2020_ED_AllParams_on_vU_RF_trim_grid.tif"
VU_PATH   = r"D:\02_grid-20250906T025757Z-1-001\02_grid\out_rf\vU_RF_trim.tif"
VE_PATH   = r"D:\02_grid-20250906T025757Z-1-001\02_grid\out_rf\vE_RF_trim.tif"
VN_PATH   = r"D:\02_grid-20250906T025757Z-1-001\02_grid\out_rf\vN_RF_trim.tif"

OUT_BASE  = r"D:\out_debris_ML_multiclass_vAll_PINN_shap_gauss_MRVBF_TWI_FSfull_FINAL"
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

# Motion thresholds
VU_MIN_MOV      = 10.0    # mm/yr, minimum |vU|
SLOPE_MIN_LS    = 20.0    # deg, landslide domain min slope

# Seed percentiles
Q_SRC      = 82
Q_NORISK   = 20
Q_VU_NEG   = 10
Q_VU_POS   = 90

# Network
PINN_HIDDEN_LAYERS = (96, 48)  # slightly stronger than before
PINN_LR            = 1e-3
PINN_EPOCHS        = 40
PINN_BATCH_SIZE    = 512

# Physics weights (tune here)
LAMBDA_LOW_MOV      = 6.0
LAMBDA_LS           = 1.2
LAMBDA_DEP_DOMAIN   = 2.2
LAMBDA_FS_SOURCE    = 3.0
LAMBDA_FS_STABLE    = 2.0
LAMBDA_MRVBF_DEP    = 2.5
LAMBDA_TWI_DEP      = 2.5

# FS_instab thresholds
FS_INSTAB_HIGH = 0.6
FS_INSTAB_LOW  = 0.3

# Deposition terrain thresholds
GENTLE_SLOPE_MAX = 18.0   # deg
MRVBF_DEP_THR    = 0.55   # in [0,1]
TWI_DEP_THR      = 0.60   # in [0,1]

# Logit priors strength (IMPORTANT for "dominant class" behavior)
# These are added to logits then re-softmax => probabilities remain valid.
PRIOR_LS_LOGIT_BOOST    = 0.45   # encourage slope class in LS domain
PRIOR_DEP_LOGIT_BOOST   = 0.55   # encourage deposition in deposition domain
PRIOR_STABLE_LOGIT_BOOST= 0.60   # encourage stable when low motion

# Optional hard disallow (in logit space) - keep mild, not -inf
DISALLOW_PENALTY = 3.0  # larger => stronger suppression of disallowed classes

KFOLDS       = 5
RANDOM_STATE = 42

# Class balance
MAX_PER_CLASS = 20_000
MIN_PER_CLASS = 300

# SHAP
MAX_SHAP_SAMPLES = 1500

# Prob smoothing
PROB_SHRINK   = 0.85  # less shrink than 0.6; too much shrink can kill deposition
SMOOTH_SIGMA  = 1.2
UNCERT_THRESH = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[i] Using device: {DEVICE}")

# ====================== GEOTECH FOR FS ======================

C_EFF_KPA   = 5.0
PHI_EFF_DEG = 30.0
GAMMA_S_KN  = 18.0
GAMMA_W_KN  = 9.81
Z_SOIL_M    = 2.0

# ====================== IO / UTILS ======================

def read_band(path, band=1):
    with rasterio.open(path) as ds:
        arr = ds.read(band).astype("float32")
        prof = ds.profile
        nod = ds.nodata
    if nod is not None:
        arr = np.where(arr == nod, np.nan, arr)
    arr = np.where(np.abs(arr) > 1e4, np.nan, arr)
    return arr, prof

def read_dem_stack_features(path):
    """
    Adjust band indices if needed (based on your stack).
    """
    with rasterio.open(path) as ds:
        dem         = ds.read(7).astype("float32")
        slope_deg   = ds.read(8).astype("float32")
        aspect_deg  = ds.read(10).astype("float32")
        NDVI        = ds.read(17).astype("float32")
        R_mm_smooth = ds.read(21).astype("float32")
        TWI_raw     = ds.read(27).astype("float32")
        Relief250m  = ds.read(29).astype("float32")
        profile     = ds.profile

    rad = np.deg2rad(aspect_deg)
    cosAsp = np.cos(rad).astype("float32")
    sinAsp = np.sin(rad).astype("float32")
    return (dem, slope_deg, aspect_deg, cosAsp, sinAsp,
            NDVI, R_mm_smooth, TWI_raw, Relief250m, profile)

def write_gtiff(path, arr, profile, nodata=np.nan):
    prof = profile.copy()
    prof.update(count=1, dtype="float32", nodata=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)
    print(f"[+] WRITE GTiff: {path}")

def robust_norm01(x, qlo=5, qhi=95, mask=None):
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
    else:
        return prob_map.astype("float32")

def shrink_probs(p, alpha=0.85):
    if alpha is None:
        return p.astype("float32")
    return (0.5 + (p - 0.5) * alpha).astype("float32")

def softmax_np(z, axis=-1):
    z = z - np.nanmax(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    s = np.nansum(ez, axis=axis, keepdims=True) + 1e-12
    return (ez / s).astype("float32")

def probs_to_logits(p):
    p = np.clip(p, 1e-6, 1.0)
    return np.log(p).astype("float32")

# ====================== TERRAIN DERIVATIVES ======================

def terrain_plan_profile_curv(dem, res_x, res_y):
    gy, gx = np.gradient(dem.astype("float64"), res_y, res_x)
    gyy, gyx = np.gradient(gy, res_y, res_x)
    gxy, gxx = np.gradient(gx, res_y, res_x)
    plan_curv = (gxx + gyy).astype("float32")
    prof_curv = (gyy).astype("float32")
    return plan_curv, prof_curv

# ====================== MRVBF (MEMORY SAFE APPROX) ======================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def mrvbf_approx(dem, slope_deg, mask, sigmas=(2, 4, 8, 16), slope_thr=5.0):
    """
    Memory-safe multi-scale approximation of MRVBF:
    - Flatness from smoothed slope at multiple scales
    - Lowness from relative elevation (local mean - elevation) / local std
    Combines across scales and normalizes to [0,1] robustly.

    Requires scipy for best results; if not available, it still runs but will be weaker.
    """
    if not SCIPY_GAUSS:
        # Fallback: simple proxy using slope only (still valid, but less good)
        # valley bottoms ~ low slope
        m = mask & np.isfinite(slope_deg)
        raw = np.zeros_like(slope_deg, dtype="float32")
        raw[m] = np.clip(1.0 - slope_deg[m] / 30.0, 0.0, 1.0)
        raw[~mask] = np.nan
        return raw.astype("float32")

    dem_f = dem.astype("float32").copy()
    slp_f = slope_deg.astype("float32").copy()

    dem_f[~mask] = np.nan
    slp_f[~mask] = np.nan

    # fill NaN for filtering stability
    med_dem = np.nanmedian(dem_f[mask])
    med_slp = np.nanmedian(slp_f[mask])
    dem_fill = np.where(mask & np.isfinite(dem_f), dem_f, med_dem).astype("float32")
    slp_fill = np.where(mask & np.isfinite(slp_f), slp_f, med_slp).astype("float32")

    acc = np.zeros_like(dem_fill, dtype="float32")
    wsum = 0.0

    for s in sigmas:
        # smoothed slope and dem
        slp_s = gaussian_filter(slp_fill, sigma=float(s), mode="nearest")
        dem_s = gaussian_filter(dem_fill, sigma=float(s), mode="nearest")
        dem2_s = gaussian_filter(dem_fill * dem_fill, sigma=float(s), mode="nearest")
        var = np.clip(dem2_s - dem_s * dem_s, 0.0, None)
        std = np.sqrt(var + 1e-6).astype("float32")

        # flatness: high when slope below threshold
        flat = sigmoid((slope_thr - slp_s) / (0.75 + 0.02 * slope_thr)).astype("float32")

        # lowness: high when elevation is lower than local mean
        low = sigmoid((dem_s - dem_fill) / (std + 1e-6)).astype("float32")

        score = (flat * low).astype("float32")

        # give more weight to broader scales
        w = float(np.log2(s + 1.0))
        acc += w * score
        wsum += w

    raw = acc / (wsum + 1e-6)
    raw[~mask] = np.nan

    # robust normalize to [0,1]
    mrvbf_n = robust_norm01(raw, 5, 95, mask=mask)
    return mrvbf_n.astype("float32")

# ====================== SHAP BEESWARM (same as yours) ======================

def beeswarm_axis_shap(ax, shap_vals, X, feat_names, max_display=15, cmap_name="coolwarm"):
    if shap_vals.ndim != 2 or X.ndim != 2:
        raise ValueError("shap_vals and X must be 2D arrays.")
    if shap_vals.shape[0] != X.shape[0]:
        if shap_vals.shape[1] == X.shape[0] and shap_vals.shape[0] == X.shape[1]:
            shap_vals = shap_vals.T
            print("[i] Warning: shap_vals transposed inside plot function to match X.")
        else:
            raise ValueError(f"shap_vals shape {shap_vals.shape} incompatible with X shape {X.shape}.")

    ns, nf = shap_vals.shape
    shap_abs_mean = np.mean(np.abs(shap_vals), axis=0)

    vel_names = ["vU", "vE", "vN", "vabs", "v_par", "vh"]
    vel_idx = [feat_names.index(n) for n in vel_names if n in feat_names]

    all_idx = np.argsort(shap_abs_mean)[::-1].tolist()
    other_idx = [i for i in all_idx if i not in vel_idx]
    chosen_idx = (vel_idx + other_idx)[:max_display]

    chosen_names = [feat_names[i] for i in chosen_idx]
    print(f"[i] SHAP beeswarm shows: {', '.join(chosen_names)}")

    cmap = plt.get_cmap(cmap_name)
    rng = np.random.default_rng(0)
    x_min, x_max = 0.0, 0.0

    for row, j in enumerate(chosen_idx):
        sv = shap_vals[:, j]
        fv = X[:, j]
        y = np.full_like(sv, row, dtype="float32") + rng.uniform(-0.35, 0.35, size=ns)
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
    base_dir = os.path.dirname(out_path)

    y_oof = info["y_oof"]
    proba_oof = info["proba_oof"]
    maxp = proba_oof.max(axis=1)

    if use_ambiguous_only:
        mask_eval = (maxp > ROC_PR_LOWER) & (maxp < ROC_PR_UPPER)
        if mask_eval.sum() < ROC_PR_MIN_PIX:
            print("[i] ROC/PR: too few ambiguous pixels; using all labeled pixels.")
            mask_eval = np.ones_like(maxp, dtype=bool)
    else:
        print("[i] ROC/PR: using ALL labeled pixels.")
        mask_eval = np.ones_like(maxp, dtype=bool)

    y_eval = y_oof[mask_eval]
    proba_eval = proba_oof[mask_eval]
    y_bin = label_binarize(y_eval, classes=np.arange(N_CLASSES))
    rng_local = np.random.RandomState(0)

    # ROC
    ax_roc = axes[0]
    auc_list = []
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
        auc_list.append(auc_cls)
        ax_roc.plot(fpr, tpr, lw=2, color=colors[name], label=f"{name} (AUC = {auc_cls:.3f})")
        np.savetxt(os.path.join(base_dir, f"roc_{name}.csv"),
                   np.column_stack([fpr, tpr]), delimiter=",", header="fpr,tpr", comments="")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("False positive rate"); ax_roc.set_ylabel("True positive rate")
    ax_roc.legend(frameon=False, loc="lower right")
    ax_roc.text(-0.18, 1.05, "(a)", transform=ax_roc.transAxes, fontsize=12)
    if auc_list:
        print(f"[i] ROC macro-AUC = {np.mean(auc_list):.3f}")

    # PR
    ax_pr = axes[1]
    ap_list = []
    for cls in range(N_CLASSES):
        name = CLASS_NAMES[cls]
        y_true = y_bin[:, cls]
        if int(y_true.sum()) < 5:
            continue
        scores = proba_eval[:, cls].copy()
        if ROC_JITTER_STD > 0:
            scores += rng_local.normal(0.0, ROC_JITTER_STD, size=scores.shape)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        order = np.argsort(rec)
        rec_s = rec[order]
        pre_s = prec[order]
        for i in range(len(pre_s) - 2, -1, -1):
            pre_s[i] = max(pre_s[i], pre_s[i + 1])
        ap_cls = average_precision_score(y_true, scores)
        ap_list.append(ap_cls)
        ax_pr.plot(rec_s, pre_s, lw=2, color=colors[name], label=f"{name} (AP = {ap_cls:.3f})")
        np.savetxt(os.path.join(base_dir, f"pr_{name}.csv"),
                   np.column_stack([rec_s, pre_s]), delimiter=",", header="recall,precision", comments="")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1)
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.legend(frameon=False, loc="lower left")
    ax_pr.text(-0.18, 1.05, "(b)", transform=ax_pr.transAxes, fontsize=12)
    if ap_list:
        print(f"[i] PR macro-AP = {np.mean(ap_list):.3f}")

    # SHAP
    ax_shap = axes[2]
    beeswarm_axis_shap(ax_shap, shap_vals_for_plot, X_shap_for_plot, feat_names, max_display=15)
    ax_shap.text(-0.18, 1.05, "(c)", transform=ax_shap.transAxes, fontsize=12)

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

    # feature indices used in soft constraints
    idx_vu_abs = feat_names.index("vU_abs")
    idx_vh     = feat_names.index("vh")
    idx_slope  = feat_names.index("slope")
    idx_fs     = feat_names.index("FS_instab")
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
            vh_b     = xb[:, idx_vh]
            slope_b  = xb[:, idx_slope]
            FS_b     = xb[:, idx_fs]
            TWI_b    = xb[:, idx_twi]
            MRVBF_b  = xb[:, idx_mrvbf]

            P_st = probs[:, STABLE]
            P_ls = probs[:, SLOPE_MOV]
            P_dp = probs[:, DEP]

            loss_phys = torch.tensor(0.0, device=device)

            # 1) low motion => stable
            low_mov = (vU_abs_b < VU_MIN_MOV)
            if low_mov.any():
                loss_phys = loss_phys + LAMBDA_LOW_MOV * (P_ls[low_mov] + P_dp[low_mov]).mean()

            # 2) landslide domain preference (slope >= SLOPE_MIN_LS and moving)
            moving = (vU_abs_b >= VU_MIN_MOV)
            ls_dom = moving & (slope_b >= SLOPE_MIN_LS)
            if ls_dom.any():
                # push P_ls above P_dp
                loss_phys = loss_phys + LAMBDA_LS * torch.relu(P_dp[ls_dom] - P_ls[ls_dom]).mean()

            # 3) deposition domain preference (gentle, low FS_instab, MRVBF or TWI high)
            gentle = (slope_b <= GENTLE_SLOPE_MAX)
            fs_low = (FS_b <= FS_INSTAB_LOW)
            mrvbf_hi = (MRVBF_b >= MRVBF_DEP_THR)
            twi_hi   = (TWI_b   >= TWI_DEP_THR)

            dep_dom = moving & gentle & fs_low & (mrvbf_hi | twi_hi)
            if dep_dom.any():
                # ensure deposition is high and > slope
                dep_push = (torch.relu(0.55 - P_dp[dep_dom]) + torch.relu(P_ls[dep_dom] - P_dp[dep_dom])).mean()
                loss_phys = loss_phys + LAMBDA_DEP_DOMAIN * dep_push

                # also explicitly tie MRVBF/TWI to deposition
                loss_phys = loss_phys + LAMBDA_MRVBF_DEP * torch.relu(0.55 - P_dp[dep_dom]).mean()
                loss_phys = loss_phys + LAMBDA_TWI_DEP   * torch.relu(0.55 - P_dp[dep_dom]).mean()

            # 4) FS guidance: high instability => slope, low instability => stable
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
    print("[i] ROC/PR are against physical seeds (rule consistency), not independent inventory validation.")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    n = X.shape[0]
    proba_oof = np.zeros((n, N_CLASSES), dtype="float32")

    for k, (tr, te) in enumerate(skf.split(X, y), start=1):
        print(f"[i] Fold {k}/{n_splits} ...")
        m = train_single_model(X[tr], y[tr], feat_names)
        proba_oof[te, :] = predict_proba(m, X[te])

    print("[i] Training final model on full dataset...")
    clf_final = train_single_model(X, y, feat_names)

    # SHAP for slope class
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

    if shap_vals_raw.shape == X_shap.shape:
        shap_vals = shap_vals_raw.astype("float32")
    elif shap_vals_raw.T.shape == X_shap.shape:
        shap_vals = shap_vals_raw.T.astype("float32")
        print("[i] Note: SHAP array transposed and corrected (T).")
    else:
        raise ValueError(f"[!] SHAP values shape {shap_vals_raw.shape} incompatible with X_shap {X_shap.shape}.")

    info = dict(proba_oof=proba_oof, y_oof=y, X_shap=X_shap.astype("float32"), shap_vals=shap_vals)
    return clf_final, info

# ====================== MAIN PIPELINE ======================

print("[+] === LOAD & ALIGN ===")

(dem, slope, aspect_deg, cosAsp, sinAsp,
 NDVI, R_mm_smooth, TWI_raw, Relief250m, profile) = read_dem_stack_features(DEM_STACK)

vU, _ = read_band(VU_PATH)
vE, _ = read_band(VE_PATH)
vN, _ = read_band(VN_PATH)

mask_insar = np.isfinite(vU) & np.isfinite(vE) & np.isfinite(vN)
print(f"[i] InSAR valid (3 comps): {mask_insar.sum()} / {mask_insar.size} ({100*mask_insar.mean():.2f}%)")

# mask basic geom
mask_morph = np.isfinite(dem) & np.isfinite(slope)

vU = np.where(mask_insar, vU, np.nan).astype("float32")
vE = np.where(mask_insar, vE, np.nan).astype("float32")
vN = np.where(mask_insar, vN, np.nan).astype("float32")

vh   = np.hypot(vE, vN).astype("float32")
vabs = np.hypot(vh, vU).astype("float32")
vU_abs = np.abs(vU).astype("float32")

aspect_rad = np.deg2rad(aspect_deg)
ux = np.sin(aspect_rad)
uy = np.cos(aspect_rad)
v_par = (vE * ux + vN * uy).astype("float32")

# percentiles for seeds
vU_q = np.percentile(vU[np.isfinite(vU)], [Q_VU_NEG, Q_VU_POS])
vU_abs_q = np.percentile(vU_abs[np.isfinite(vU_abs)], [Q_NORISK, Q_SRC, 95])
vh_q = np.percentile(vh[np.isfinite(vh)], [30, 80])

p_norisk = vU_abs_q[0]
p_src_th = vU_abs_q[1]
p95_abs  = vU_abs_q[2]
vh_p30   = vh_q[0]
vh_p80   = vh_q[1]

print(f"[i] |vU| percentiles: p{Q_NORISK}={p_norisk:.2f}, p{Q_SRC}={p_src_th:.2f}, p95={p95_abs:.2f}")
print(f"[i] vh percentiles: p30={vh_p30:.2f}, p80={vh_p80:.2f}")

res_x = abs(profile["transform"][0])
res_y = abs(profile["transform"][4])

# ===== FS (infinite-slope) + FS_instab =====
print("[+] Computing FS (infinite-slope) & FS_instab ...")
slope_rad = np.deg2rad(slope)

valid_R = R_mm_smooth[np.isfinite(R_mm_smooth)]
if valid_R.size > 0:
    R_p20, R_p80 = np.percentile(valid_R, [20, 80])
    with np.errstate(invalid="ignore", divide="ignore"):
        m_sat = (R_mm_smooth - R_p20) / (R_p80 - R_p20 + 1e-6)
        m_sat = np.clip(m_sat, 0.0, 1.0)
else:
    m_sat = np.ones_like(slope, dtype="float32")

mask_fs = np.isfinite(slope) & np.isfinite(dem) & np.isfinite(R_mm_smooth)

sigma_n = GAMMA_S_KN * Z_SOIL_M * (np.cos(slope_rad) ** 2)
u       = GAMMA_W_KN * m_sat * Z_SOIL_M * (np.cos(slope_rad) ** 2)
phi_rad = np.deg2rad(PHI_EFF_DEG)

tau_resist = C_EFF_KPA + (sigma_n - u) * np.tan(phi_rad)
tau_drive  = GAMMA_S_KN * Z_SOIL_M * np.sin(slope_rad) * np.cos(slope_rad)

with np.errstate(invalid="ignore", divide="ignore"):
    FS_raw = tau_resist / (tau_drive + 1e-6)
FS_raw[~mask_fs] = np.nan
write_gtiff(os.path.join(RASTERS, "FS_raw.tif"), FS_raw, profile)

FS_clip = np.clip(FS_raw, 0.1, 5.0)
a_logit = 4.0
with np.errstate(invalid="ignore"):
    FS_instab = 1.0 / (1.0 + np.exp(a_logit * (FS_clip - 1.0)))
FS_instab[~mask_fs] = np.nan
write_gtiff(os.path.join(RASTERS, "FS_instab.tif"), FS_instab, profile)

# ===== TWI normalized =====
print("[+] Normalizing TWI to 0–1 ...")
TWI_n = robust_norm01(TWI_raw, 5, 95, mask=mask_morph)
write_gtiff(os.path.join(RASTERS, "TWI_norm01.tif"), TWI_n, profile)

# ===== MRVBF (normalized 0–1) =====
print("[+] Computing MRVBF (multi-scale valley-bottom flatness) ...")
MRVBF_n = mrvbf_approx(dem, slope, mask=mask_morph, sigmas=(2,4,8,16,32,64), slope_thr=8.0)
write_gtiff(os.path.join(RASTERS, "MRVBF_norm01.tif"), MRVBF_n, profile)

# ====================== PHYSICS SEEDS ======================

mask_basic = mask_insar & np.isfinite(slope) & np.isfinite(dem)

mov_mask = mask_basic & (vU_abs >= VU_MIN_MOV)
strong_mov = mov_mask & ((vU_abs >= p_src_th) | (vh >= vh_p80))

slope_movement_seed = strong_mov & (slope >= SLOPE_MIN_LS)

# deposition seeds: moving but gentle and valley-bottom-ish + hydrologic accumulation + low FS_instab
dep_seed = strong_mov & (slope < SLOPE_MIN_LS)
dep_seed = dep_seed & (slope <= (GENTLE_SLOPE_MAX + 5.0)) & (FS_instab <= FS_INSTAB_LOW)
dep_seed = dep_seed & ((TWI_n >= 0.5) | (MRVBF_n >= 0.5))

stable_seed = mask_basic & (vU_abs <= p_norisk) & (vh <= vh_p30)
stable_seed &= ~(slope_movement_seed | dep_seed)

n_ls = int(slope_movement_seed.sum())
n_dp = int(dep_seed.sum())
n_st = int(stable_seed.sum())
print(f"[i] Seeds -> slope_movement={n_ls}  debris_deposition={n_dp}  stable={n_st}")

write_gtiff(os.path.join(TRAINING, "seed_slope_movement.tif"), slope_movement_seed.astype("float32"), profile, nodata=0)
write_gtiff(os.path.join(TRAINING, "seed_debris_deposition.tif"), dep_seed.astype("float32"), profile, nodata=0)
write_gtiff(os.path.join(TRAINING, "seed_stable.tif"), stable_seed.astype("float32"), profile, nodata=0)

# ====================== FEATURE STACK ======================

features_list = [
    vU, vE, vN, vU_abs, vabs, v_par, vh,
    slope,
    dem,
    NDVI, R_mm_smooth, Relief250m,
    FS_instab, TWI_n, MRVBF_n,
    cosAsp, sinAsp
]
feat_names = [
    "vU","vE","vN","vU_abs","vabs","v_par","vh",
    "slope",
    "dem",
    "NDVI","R_mm_smooth","Relief250m",
    "FS_instab","TWI_n","MRVBF_n",
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

# ====================== BUILD TRAIN SET FROM SEEDS ======================

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
        # keep stable full (or minimally downsample if too huge)
        if n_cls < MIN_PER_CLASS:
            sel = rng.choice(cls_idx, size=MIN_PER_CLASS, replace=True)
        else:
            sel = cls_idx
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

print("[+] Training physics-guided MLP + SHAP ...")
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
# This is the key fix: all outputs remain valid probabilities.

# Domains
low_mov = (vU_abs < VU_MIN_MOV) & valid_mask

ls_domain = (valid_mask &
             (vU_abs >= VU_MIN_MOV) &
             (slope >= SLOPE_MIN_LS))

dep_domain = (valid_mask &
              (vU_abs >= VU_MIN_MOV) &
              (slope <= (GENTLE_SLOPE_MAX + 5.0)) &
              (FS_instab <= FS_INSTAB_LOW) &
              ((TWI_n >= TWI_DEP_THR) | (MRVBF_n >= MRVBF_DEP_THR)))

# Convert prob -> logits, add priors, re-softmax
logits = probs_to_logits(np.where(np.isfinite(probs), probs, np.nan))

# init missing logits
mask_logits = np.isfinite(logits).all(axis=2)
# if any pixel has NaNs, set uniform logits there
uniform_logits = np.log(np.full(N_CLASSES, 1.0 / N_CLASSES, dtype="float32"))
logits[~mask_logits] = uniform_logits

# priors:
# stable boost for low motion
logits[low_mov, STABLE] += PRIOR_STABLE_LOGIT_BOOST
logits[low_mov, SLOPE_MOV] -= DISALLOW_PENALTY
logits[low_mov, DEP] -= DISALLOW_PENALTY

# landslide domain: boost slope, mildly suppress deposition
logits[ls_domain, SLOPE_MOV] += PRIOR_LS_LOGIT_BOOST
logits[ls_domain, DEP] -= 0.5 * DISALLOW_PENALTY

# deposition domain: boost deposition, mildly suppress slope
logits[dep_domain, DEP] += PRIOR_DEP_LOGIT_BOOST
logits[dep_domain, SLOPE_MOV] -= 0.5 * DISALLOW_PENALTY

# Re-softmax => sums to 1 and in [0,1]
probs = softmax_np(logits, axis=2)

# Smooth each probability map (optional but recommended)
for k in range(N_CLASSES):
    probs[:, :, k] = smooth_prob_map(probs[:, :, k], sigma=SMOOTH_SIGMA)
    probs[:, :, k][~valid_mask] = np.nan

# Re-enforce low motion as stable (final)
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

print("[+] Figure ROC/PR/SHAP (ALL) ...")
plot_global_roc_pr_shap_multiclass(
    info, feat_names,
    os.path.join(FIGURES, "global_ROC_PR_SHAP_multiclass_ALL.png"),
    X_shap_for_plot=info["X_shap"],
    shap_vals_for_plot=info["shap_vals"],
    use_ambiguous_only=False
)

print("[+] Figure ROC/PR/SHAP (AMBIG) ...")
plot_global_roc_pr_shap_multiclass(
    info, feat_names,
    os.path.join(FIGURES, "global_ROC_PR_SHAP_multiclass_AMBIG.png"),
    X_shap_for_plot=info["X_shap"],
    shap_vals_for_plot=info["shap_vals"],
    use_ambiguous_only=True
)

# ====================== RISK INDEX ======================

print("[+] Risk index ...")
prob_moving = np.clip(1.0 - prob_stable, 0, 1)

with np.errstate(invalid="ignore"):
    norm_vu = np.clip((vU_abs - p_norisk) / (p95_abs - p_norisk + 1e-6), 0, 1)
    norm_slope = np.clip((slope - 5.0) / (40.0 - 5.0 + 1e-6), 0, 1)

risk_base = prob_moving * norm_vu * norm_slope
valid_rb = risk_base[(risk_base > 0) & np.isfinite(risk_base)]
if valid_rb.size > 0:
    ref = np.percentile(valid_rb, 95)
    risk_index = np.clip(risk_base / (ref + 1e-6), 0, 1)
else:
    risk_index = risk_base

risk_index = risk_index.astype("float32")
risk_index[~valid_mask] = np.nan
risk_index[uncertainty_mask == 1] = np.nan
write_gtiff(os.path.join(RASTERS, "risk_index.tif"), risk_index, profile)

valid_vals = risk_index[np.isfinite(risk_index) & (risk_index > 0)]
if valid_vals.size > 0:
    q20, q40, q60, q80 = np.percentile(valid_vals, [30, 50, 70, 90])
else:
    q20 = q40 = q60 = q80 = 0.0

bins = [0, q20, q40, q60, q80, 1.01]
risk_class = np.digitize(risk_index, bins, right=False).astype("float32")
risk_class[~np.isfinite(risk_index)] = np.nan
write_gtiff(os.path.join(RASTERS, "risk_class.tif"), risk_class, profile)

# Risk map (same palette)
print("[+] Risk map figure ...")
risk_cmap = mcolors.ListedColormap(["#1a9850","#66bd63","#ffffbf","#fdae61","#d73027"])
risk_plot = np.ma.masked_where((~np.isfinite(risk_class)) | (risk_class < 1), risk_class)
bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
norm = mcolors.BoundaryNorm(bounds, risk_cmap.N)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300})
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(risk_plot, cmap=risk_cmap, norm=norm, origin="upper")
ax.set_xticks([]); ax.set_yticks([])
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[1,2,3,4,5])
cbar.ax.set_yticklabels(["Very low","Low","Moderate","High","Very high"])
cbar.set_label("Relative susceptibility (dimensionless)")
fig.savefig(os.path.join(FIGURES, "risk_map_GnYlRd.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("[+] FIGURE saved: risk_map_GnYlRd.png")

# ====================== SUMMARY ======================

px_valid = int(mask_insar.sum())
px_to_km2 = (res_x * res_y) / 1e6

area_ls  = np.nansum(prob_slope >= 0.5) * px_to_km2
area_dep = np.nansum(prob_dep   >= 0.5) * px_to_km2
area_st  = np.nansum(prob_stable>= 0.5) * px_to_km2

print("\n=== SUMMARY (p>=0.5) ===")
print(f"InSAR valid pixels (3 comps): {px_valid} ({100*px_valid/(H*W):.2f}%)")
print(f"Seeds -> slope_movement={n_ls}  deposition={n_dp}  stable={n_st}")
print(f"AREAS >=0.5 (km²) -> Slope-movement={area_ls:.2f}  Deposition={area_dep:.2f}  Stable={area_st:.2f}")
print("Rasters  ->", RASTERS)
print("Training ->", TRAINING)
print("Figures  ->", FIGURES)
print("[✓] Finished.")
