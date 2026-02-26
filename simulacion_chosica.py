# -*- coding: utf-8 -*-
"""
Chosica (Peru) – Debris-flow / landslide hybrid runout with SynxFlow,
plus erosion–deposition proxy.
"""

import os
import time
import csv
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform as rio_transform

from synxflow import IO
from synxflow import landslide

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LightSource
from matplotlib.ticker import FixedLocator
import imageio.v2 as imageio
from multiprocessing import Process
from scipy import ndimage 

# ---------------------------------------------------------------------
# 0. PATHS & GLOBAL PARAMETERS
# ---------------------------------------------------------------------

BASE_DIR = r"C:\Users\Zhou laoshi\OneDrive\Documentos\CHOSICA_PERU"

MODELO_DIR          = os.path.join(BASE_DIR, "Modelo")
RASTER_OUT_DIR      = os.path.join(BASE_DIR, "Rasters_salida")
RUNOUT_CASE_FOLDER  = os.path.join(BASE_DIR, "SynxFlow_Runout")
ED_FOLDER           = os.path.join(BASE_DIR, "ErosionDeposition")
FRAMES_RUNOUT_DIR   = os.path.join(BASE_DIR, "GIF_frames_runout")
FRAMES_ED_DIR       = os.path.join(BASE_DIR, "GIF_frames_ED")
GIF_DIR             = os.path.join(BASE_DIR, "GIFs")
RUNOUT_TIF_DIR      = os.path.join(BASE_DIR, "Runout_TIF")
CHART_DIR           = os.path.join(BASE_DIR, "Charts")

for folder in [
    BASE_DIR, MODELO_DIR, RASTER_OUT_DIR,
    RUNOUT_CASE_FOLDER, ED_FOLDER,
    FRAMES_RUNOUT_DIR, FRAMES_ED_DIR,
    GIF_DIR, RUNOUT_TIF_DIR, CHART_DIR
]:
    os.makedirs(folder, exist_ok=True)

print("[INFO] Working base directory:", BASE_DIR)

# Input rasters
DEM_STACK = (
    r"D:\02_grid-20250906T025757Z-1-001\02_grid\out_rf\Chosica2020_ED_AllParams_on_vU_RF_trim_grid.tif"
)

VU_PATH   = r"D:\02_grid-20250906T025757Z-1-001\02_grid\out_rf\vU_RF_trim.tif"
VE_PATH   = r"D:\02_grid-20250906T025757Z-1-001\02_grid\out_rf\vE_RF_trim.tif"
VN_PATH   = r"D:\02_grid-20250906T025757Z-1-001\02_grid\out_rf\vN_RF_trim.tif"

# Probabilistic landslide raster
PROB_LANDSLIDE_PATH = (
    r"D:\out_debris_ML_multiclass_vAll_LR\test\rasters\prob_landslide.tif"
)

# Velocity → thickness parameters
VEL_THRESH_MM_YR   = 2.0       # below this velocity treated as zero
MAX_DEPTH_M        = 30.0      # upper part of thickness range (increased)
MIN_DEPTH_M        = 2.0       # minimum thickness in any source cell (increased)
TARGET_VOLUME_M3   = 5e8     # slightly larger target volume for more mobility
GAMMA_EXP          = 0.5      # exponent in velocity→thickness mapping

# Source area filters
SLOPE_MIN_DEG      = 10.0
PROB_LANDSLIDE_MIN = 0.6
MIN_SOURCE_PIX     = 50        # remove tiny patches

# Rheology (Mohr–Coulomb) – slightly more mobile than previous version
RHEO_TYPE   = 1
RHEO_PARAMS = [0.2, 3.0, 2200]
# [friction coeff (~tan(phi)), cohesion kPa, density kg/m³]
# Lower friction & cohesion than before, density unchanged.

# Runtime [t0, t_end, dt_out, dt_backup]
RUNTIME = [0, 900, 1, 900]     # 0–900 s, output every 1 s

# Stop-loss: 1.5 days wall-clock (36 h)
MAX_WALL_TIME_SEC = int(1.5 * 24 * 3000)  # 129,600 s

# GIF + hazard thresholds
GIF_RUNOUT_PATH = os.path.join(GIF_DIR, "chosica_runout_2D_plasma.gif")
GIF_ED_PATH     = os.path.join(GIF_DIR, "chosica_erosion_deposition_2D.gif")

HZ_THRESH = [0.05, 0.30, 1.0, 3.0]  # thickness in m for hazard classes

# Hillshade parameters
HILLSHADE_AZIMUTH  = 315.0
HILLSHADE_ALTITUDE = 45.0


# ---------------------------------------------------------------------
# 1. DEM & FEATURES
# ---------------------------------------------------------------------

def read_dem_stack_features(path):
    with rasterio.open(path) as ds:
        print("\nDEM_STACK info:")
        print("  CRS:", ds.crs)
        print("  Transform:", ds.transform)
        print("  Size (WxH):", ds.width, "x", ds.height)
        print("  Band count:", ds.count)

        dem         = ds.read(7).astype("float32")
        slope_deg   = ds.read(8).astype("float32")
        aspect_deg  = ds.read(10).astype("float32")
        LS          = ds.read(14).astype("float32")
        NDVI        = ds.read(17).astype("float32")
        R_mm_smooth = ds.read(21).astype("float32")
        SPI         = ds.read(24).astype("float32")
        TWI         = ds.read(27).astype("float32")
        Relief250m  = ds.read(29).astype("float32")
        profile     = ds.profile

    print("\n[DEBUG] DEM stats:")
    print("  DEM shape:", dem.shape)
    print("  DEM min/max:", float(np.nanmin(dem)), float(np.nanmax(dem)))
    print("[DEBUG] Slope min/max (deg):", float(np.nanmin(slope_deg)), "/", float(np.nanmax(slope_deg)))
    return dem, slope_deg, aspect_deg, LS, NDVI, R_mm_smooth, SPI, TWI, Relief250m, profile


# ---------------------------------------------------------------------
# 2. RASTER UTILITIES
# ---------------------------------------------------------------------

def clean_nodata(arr, name="raster"):
    arr = arr.astype("float32")
    mask_extreme = np.abs(arr) > 1e20
    if np.any(mask_extreme):
        print(f"[DEBUG] {name}: {mask_extreme.sum()} values |v|>1e20 set to NaN.")
        arr[mask_extreme] = np.nan

    mask_outlier = np.abs(arr) > 1e5
    if np.any(mask_outlier):
        print(f"[DEBUG] {name}: {mask_outlier.sum()} values |v|>1e5 set to NaN.")
        arr[mask_outlier] = np.nan

    return arr


def read_and_align_raster(path, target_profile, name="raster", treat_nodata_as_nan=True):
    with rasterio.open(path) as src:
        print(f"\n[INFO] Reading {name} from: {path}")
        print("  Original CRS:", src.crs)
        print("  Original transform:", src.transform)
        print("  Original size (WxH):", src.width, "x", src.height)

        data = src.read(1).astype("float32")

        if treat_nodata_as_nan and src.nodata is not None:
            nd = src.nodata
            print(f"  nodata value in file: {nd}")
            data[data == nd] = np.nan

        if treat_nodata_as_nan:
            data = clean_nodata(data, name=name + " (raw)")

        same_grid = (
            src.crs == target_profile["crs"] and
            src.transform == target_profile["transform"] and
            src.width == target_profile["width"] and
            src.height == target_profile["height"]
        )

        if same_grid:
            print(f"  {name} already aligned to DEM grid.")
            aligned = data
        else:
            print(f"  {name} not aligned. Reprojecting to DEM grid...")
            aligned = np.full(
                (target_profile["height"], target_profile["width"]),
                np.nan if treat_nodata_as_nan else 0.0,
                dtype="float32"
            )
            reproject(
                source=data,
                destination=aligned,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_profile["transform"],
                dst_crs=target_profile["crs"],
                resampling=Resampling.bilinear,
            )

    if treat_nodata_as_nan:
        aligned = clean_nodata(aligned, name=name + " (aligned)")

    print(f"[DEBUG] {name} stats after alignment:")
    print("  Shape:", aligned.shape)
    finite_vals = aligned[np.isfinite(aligned)]
    if finite_vals.size > 0:
        print("  Min / Max (ign. NaN):", float(finite_vals.min()), "/", float(finite_vals.max()))
    else:
        print("  All values NaN or zero.")
    return aligned


# ---------------------------------------------------------------------
# 3. RUN SYNXFLOW WITH STOP-LOSS
# ---------------------------------------------------------------------

def _run_synxflow_process(case_folder):
    from synxflow import landslide as ls_mod
    ls_mod.run(case_folder)


def run_synxflow_with_timeout(case_folder, max_time_sec):
    print(f"\n[INFO] Starting SynxFlow with stop-loss of {max_time_sec} s (wall-clock).")
    p = Process(target=_run_synxflow_process, args=(case_folder,))
    p.start()
    start = time.time()
    p.join(timeout=max_time_sec)
    elapsed = time.time() - start

    if p.is_alive():
        print(f"[WARN] SynxFlow exceeded wall-clock limit ({elapsed:.1f} s). Terminating.")
        p.terminate()
        p.join()
        finished = False
    else:
        print(f"[INFO] SynxFlow finished in {elapsed:.1f} s within time limit.")
        finished = True

    return finished


# ---------------------------------------------------------------------
# 4. DMS TICKS
# ---------------------------------------------------------------------

def dms_str(value, is_lat=True):
    hemi = 'N' if (is_lat and value >= 0) else \
           'S' if (is_lat and value < 0) else \
           'E' if (not is_lat and value >= 0) else 'W'
    v = abs(value)
    deg = int(v)
    minf = (v - deg) * 60.0
    mins = int(minf)
    secs = (minf - mins) * 60.0
    return f"{deg}°{mins:02d}'{secs:02.0f}\"{hemi}"


def apply_dms_ticks(ax, crs):
    try:
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        if len(xticks) == 0 or len(yticks) == 0:
            return

        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.yaxis.set_major_locator(FixedLocator(yticks))

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        mid_y = 0.5 * (ylim[0] + ylim[1])
        mid_x = 0.5 * (xlim[0] + xlim[1])

        lon_x, lat_x = rio_transform(
            crs,
            "EPSG:4326",
            list(xticks),
            [mid_y] * len(xticks)
        )
        lon_y, lat_y = rio_transform(
            crs,
            "EPSG:4326",
            [mid_x] * len(yticks),
            list(yticks)
        )

        ax.set_xticklabels([dms_str(lon, is_lat=False) for lon in lon_x],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels([dms_str(lat, is_lat=True) for lat in lat_y],
                           fontsize=7)

        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)

    except Exception as e:
        print("[WARN] Could not apply DMS ticks:", e)


# ---------------------------------------------------------------------
# 5. METRICS
# ---------------------------------------------------------------------

def compute_runout_length(h0, h, transform, h_thresh=0.1):
    src_mask = h0 > 0
    runout_mask = h >= h_thresh
    if not np.any(src_mask) or not np.any(runout_mask):
        return np.nan

    rows_src, cols_src = np.where(src_mask)
    xs_src, ys_src = rasterio.transform.xy(transform, rows_src, cols_src)
    x0 = np.mean(xs_src)
    y0 = np.mean(ys_src)

    rows_r, cols_r = np.where(runout_mask)
    xs_r, ys_r = rasterio.transform.xy(transform, rows_r, cols_r)
    xs_r = np.array(xs_r)
    ys_r = np.array(ys_r)
    dist = np.sqrt((xs_r - x0)**2 + (ys_r - y0)**2)
    return float(dist.max())


def classify_hazard(h_final, cell_area_m2, thresholds):
    hazard_class = np.zeros_like(h_final, dtype="int16")
    h = h_final

    mask1 = (h >= thresholds[0]) & (h < thresholds[1])
    mask2 = (h >= thresholds[1]) & (h < thresholds[2])
    mask3 = (h >= thresholds[2]) & (h < thresholds[3])
    mask4 = (h >= thresholds[3])

    hazard_class[mask1] = 1
    hazard_class[mask2] = 2
    hazard_class[mask3] = 3
    hazard_class[mask4] = 4

    stats = {}
    for cls in range(5):
        n_pix = int((hazard_class == cls).sum())
        area_km2 = n_pix * cell_area_m2 / 1e6
        stats[cls] = area_km2

    return hazard_class, stats


# ---------------------------------------------------------------------
# 6. MAIN PIPELINE
# ---------------------------------------------------------------------

def main():
    # DEM
    dem, slope_deg, aspect_deg, LS, NDVI, R_mm_smooth, SPI, TWI, Relief250m, dem_profile = \
        read_dem_stack_features(DEM_STACK)

    height, width = dem.shape
    print("\n[INFO] DEM grid:", width, "x", height)

    target_profile = dem_profile.copy()
    target_profile.update(count=1, dtype="float32", width=width, height=height)

    tr = dem_profile["transform"]
    pixel_w = tr.a
    pixel_h = -tr.e
    cell_area = pixel_w * pixel_h
    print(f"[INFO] DEM resolution: {pixel_w} x {pixel_h} m -> cell area {cell_area} m^2")

    # velocities
    vU = read_and_align_raster(VU_PATH, target_profile, name="vU (mm/yr)", treat_nodata_as_nan=True)
    vE = read_and_align_raster(VE_PATH, target_profile, name="vE (mm/yr)", treat_nodata_as_nan=True)
    vN = read_and_align_raster(VN_PATH, target_profile, name="vN (mm/yr)", treat_nodata_as_nan=True)

    # probabilistic landslide map
    prob_ls = read_and_align_raster(
        PROB_LANDSLIDE_PATH,
        target_profile,
        name="prob_landslide",
        treat_nodata_as_nan=True
    )

    print("\n[DEBUG] prob_landslide stats (ign. NaN):")
    finite_prob = prob_ls[np.isfinite(prob_ls)]
    print("  min / max:", float(np.nanmin(finite_prob)), "/", float(np.nanmax(finite_prob)))

    # Source mask: prob>=0.6, slope>=10°
    prob_mask = np.isfinite(prob_ls) & (prob_ls >= PROB_LANDSLIDE_MIN)
    slope_mask = np.isfinite(slope_deg) & (slope_deg >= SLOPE_MIN_DEG)
    source_mask = prob_mask & slope_mask

    print("[DEBUG] prob>=%.2f pix:" % PROB_LANDSLIDE_MIN, int(prob_mask.sum()))
    print("[DEBUG] slope>=%.1f° pix:" % SLOPE_MIN_DEG, int(slope_mask.sum()))
    print("[DEBUG] combined source pix BEFORE area filter:", int(source_mask.sum()))

    # connected-component filtering
    labeled, num = ndimage.label(source_mask)
    print("[DEBUG] Connected components in source mask:", num)
    if num > 0:
        counts = np.bincount(labeled.ravel())
        keep_labels = np.where(counts >= MIN_SOURCE_PIX)[0]
        keep_labels = keep_labels[keep_labels != 0]
        filtered_mask = np.isin(labeled, keep_labels)
        print("[DEBUG] components kept (>= %d pix):" % MIN_SOURCE_PIX, len(keep_labels))
        print("[DEBUG] combined source pix AFTER area filter:", int(filtered_mask.sum()))
        source_mask = filtered_mask

    if source_mask.sum() == 0:
        raise RuntimeError(
            "No source pixels after probability + slope + area filter."
        )

    seed_binary = source_mask.astype("float32")
    print("[INFO] FINAL SOURCE CELLS:", int(seed_binary.sum()))

    # velocity magnitude & h0
    print("\n[INFO] Computing velocity magnitude and initial thickness h0...")

    vU_safe = np.where(np.isfinite(vU), vU, 0.0)
    vE_safe = np.where(np.isfinite(vE), vE, 0.0)
    vN_safe = np.where(np.isfinite(vN), vN, 0.0)

    with np.errstate(over="ignore", invalid="ignore"):
        vel_mag_mm_yr = np.sqrt(vU_safe**2 + vE_safe**2 + vN_safe**2)

    bad_mask = (~np.isfinite(vU)) | (~np.isfinite(vE)) | (~np.isfinite(vN))
    vel_mag_mm_yr[bad_mask] = np.nan
    vel_mag_mm_yr[vel_mag_mm_yr < VEL_THRESH_MM_YR] = 0.0

    vel_positive = vel_mag_mm_yr[vel_mag_mm_yr > 0]
    if vel_positive.size == 0:
        raise RuntimeError("All velocities below threshold; no moving mass.")

    vel_clip = np.nanpercentile(vel_positive, 99.0)
    print("\n[DEBUG] Velocity mag (mm/yr) after thresholding:")
    print("  min >0:", float(np.nanmin(vel_positive)))
    print("  99th percentile:", float(vel_clip))

    vel_eff = np.clip(vel_mag_mm_yr, VEL_THRESH_MM_YR, vel_clip)
    vel_norm = (vel_eff - VEL_THRESH_MM_YR) / (vel_clip - VEL_THRESH_MM_YR)
    vel_norm[vel_mag_mm_yr <= VEL_THRESH_MM_YR] = 0.0
    vel_norm = np.nan_to_num(vel_norm, nan=0.0, posinf=0.0, neginf=0.0)

    h_shape = vel_norm ** GAMMA_EXP

    h0 = np.zeros_like(dem, dtype="float32")
    src_mask = seed_binary > 0.5
    h0[src_mask] = MIN_DEPTH_M + h_shape[src_mask] * MAX_DEPTH_M

    current_volume = float(np.nansum(h0 * cell_area))
    print("\n[DEBUG] Base h0 stats BEFORE scaling:")
    print("  min/max (source):", float(np.nanmin(h0[src_mask])), "/", float(np.nanmax(h0[src_mask])))
    print("  preliminary volume:", current_volume, "m^3")

    if TARGET_VOLUME_M3 is not None and current_volume > 0:
        scale = TARGET_VOLUME_M3 / current_volume
        h0 *= scale
        current_volume = float(np.nansum(h0 * cell_area))
        print(f"[INFO] Rescaled h0 to target volume {TARGET_VOLUME_M3:.3g} m^3")
        print("       New volume:", current_volume, "m^3")
        print("       New max thickness:", float(np.nanmax(h0)), "m")

    # save h0 & DEM
    h0_profile = target_profile.copy()
    h0_profile.update(dtype="float32", count=1)
    h0_path = os.path.join(RASTER_OUT_DIR, "h0_depth_from_velocity.tif")
    with rasterio.open(h0_path, "w", **h0_profile) as dst:
        dst.write(h0, 1)
    print(f"\n[INFO] h0 saved to: {h0_path}")

    dem_out_path = os.path.join(RASTER_OUT_DIR, "dem_synxflow.tif")
    dem_profile_out = dem_profile.copy()
    dem_profile_out.update(count=1, dtype="float32")
    with rasterio.open(dem_out_path, "w", **dem_profile_out) as dst:
        dst.write(dem, 1)
    print(f"[INFO] DEM for SynxFlow saved to: {dem_out_path}")

    # build SynxFlow case
    print("\n[INFO] Building SynxFlow Runout InputModel...")
    DEM_raster = IO.Raster(dem_out_path)
    case_input = IO.InputModel(DEM_raster, num_of_sections=1, case_folder=RUNOUT_CASE_FOLDER)

    print("[INFO] Setting initial condition h0 for runout...")
    case_input.set_initial_condition("h0", h0)

    print("[INFO] Writing landslide configuration (Mohr–Coulomb)...")
    case_input.write_landslide_config(
        rheology_type=RHEO_TYPE,
        rheology_params=RHEO_PARAMS,
        gravity_correction_type=1,
        curvature_on=True
    )
    print("[INFO] rheology_type   =", RHEO_TYPE)
    print("[INFO] rheology_params =", RHEO_PARAMS)

    print("[INFO] Runtime settings [t0, t_end, dt_out, dt_backup]:", RUNTIME)
    case_input.set_runtime(RUNTIME)
    case_input.write_input_files()

    # run SynxFlow with stop-loss
    finished = run_synxflow_with_timeout(RUNOUT_CASE_FOLDER, MAX_WALL_TIME_SEC)
    if not finished:
        print("[WARN] SynxFlow runout terminated by stop-loss; using partial outputs if any.")

    # read outputs
    print("\n[INFO] Reading runout outputs...")
    case_output = IO.OutputModel(input_obj=case_input)
    output_folder = os.path.join(RUNOUT_CASE_FOLDER, "output")
    if not os.path.isdir(output_folder):
        raise RuntimeError(f"Runout output folder not found: {output_folder}")

    files = sorted(f for f in os.listdir(output_folder) if f.startswith("h_") and f.endswith(".asc"))
    if not files:
        raise RuntimeError("No h_*.asc files in output folder.")

    print("[INFO] Runout thickness files found:")
    for f in files:
        print("   ", f)

    thickness_list = []
    time_list = []

    tif_profile = target_profile.copy()
    tif_profile.update(dtype="float32", count=1)

    for f in files:
        tag = os.path.splitext(f)[0]
        depth_raster = case_output.read_grid_file(file_tag=tag)
        arr = depth_raster.array.astype("float32")
        thickness_list.append(arr)
        try:
            t_val = float(tag.split("_", 1)[1])
        except Exception:
            t_val = np.nan
        time_list.append(t_val)

        tif_name = f"thickness_{tag}.tif"
        tif_path = os.path.join(RUNOUT_TIF_DIR, tif_name)
        with rasterio.open(tif_path, "w", **tif_profile) as dst:
            dst.write(arr, 1)

    stacked = np.stack(thickness_list)
    global_max_thick = float(np.nanpercentile(stacked, 99.0))
    print("\n[INFO] 99th percentile thickness over all outputs:", global_max_thick, "m")

    # -----------------------------------------------------------------
    # Professional hillshade (your method)
    # -----------------------------------------------------------------

    tr = dem_profile["transform"]
    x0 = tr.c
    y0 = tr.f
    dx = tr.a
    dy = -tr.e
    x1 = x0 + dx * width
    y1 = y0 - dy * height
    extent = [x0, x1, y1, y0]

    # Fill NaNs before computing hillshade
    dem_filled = np.where(np.isnan(dem), np.nanmean(dem), dem)
    ls = LightSource(azdeg=HILLSHADE_AZIMUTH, altdeg=HILLSHADE_ALTITUDE)
    hillshade = ls.hillshade(dem_filled, vert_exag=1.0, dx=dx, dy=dy)

    grey_cmap = mpl.colormaps["Greys"]
    hillshade_rgb = grey_cmap(hillshade)
    hillshade_rgb[..., 3] = 1.0

    # -----------------------------------------------------------------
    # Time-series metrics
    # -----------------------------------------------------------------

    print("\n[INFO] Computing time-series metrics for CSV and plots...")
    time_thick_pairs = [(t, h) for t, h in zip(time_list, thickness_list) if np.isfinite(t)]
    time_thick_pairs.sort(key=lambda x: x[0])

    times_sorted = []
    volumes = []
    volumes_exceed = []
    max_thickness_ts = []
    runout_length_ts = []

    for t, h_t in time_thick_pairs:
        times_sorted.append(t)
        V_all = float(np.nansum(h_t * cell_area))
        volumes.append(V_all)

        mask_exceed = h_t >= 0.1
        V_exc = float(np.nansum(h_t[mask_exceed] * cell_area))
        volumes_exceed.append(V_exc)

        max_thickness_ts.append(float(np.nanmax(h_t)))
        runout_length_ts.append(
            compute_runout_length(h0, h_t, dem_profile["transform"], h_thresh=0.1)
        )

    csv_path = os.path.join(CHART_DIR, "chosica_runout_timeseries.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["time_s", "volume_total_m3", "volume_h_ge_0.1_m3",
                         "max_thickness_m", "runout_length_m"])
        for t, v_all, v_exc, hmax, L in zip(
            times_sorted, volumes, volumes_exceed, max_thickness_ts, runout_length_ts
        ):
            writer.writerow([t, v_all, v_exc, hmax, L])
    print(f"[INFO] Time-series CSV saved to: {csv_path}")

    # Dimensionless metrics + boxplots (exactly as before)
    plt.ioff()

    t_arr = np.array(times_sorted, dtype="float64")
    t_final = t_arr[-1]
    t_star = t_arr / t_final

    V0 = volumes[0]
    V_tot = np.array(volumes) / V0
    V_exc = np.array(volumes_exceed) / V0

    def smooth(y):
        if len(y) < 3:
            return y
        y_pad = np.pad(y, (1, 1), mode="edge")
        return (y_pad[:-2] + y_pad[1:-1] + y_pad[2:]) / 3.0

    V_tot_star = smooth(V_tot)
    V_exc_star = smooth(V_exc)

    L_arr = np.array(runout_length_ts)
    L_cummax = np.maximum.accumulate(np.nan_to_num(L_arr, nan=0.0))
    Le_max = np.nanmax(L_cummax)
    L_star = L_cummax / Le_max

    cmap_seq = mpl.colormaps["viridis"]
    c1 = cmap_seq(0.15)
    c2 = cmap_seq(0.75)
    c3 = cmap_seq(0.45)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(10, 3.5), dpi=300
    )

    # (a) Volume ratios
    ax1.plot(t_star, V_tot_star, color=c1, lw=2, label="V(t)/V₀ (total)")
    ax1.plot(t_star, V_exc_star, color=c2, lw=1.5, ls="--",
             label="V(t)/V₀ (h ≥ 0.1 m)")
    ax1.set_ylabel("Relative volume", fontsize=9)
    ax1.set_xlabel("Dimensionless time  t / tₑ", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, loc="best", frameon=False)
    ax1.set_title("(a)", loc="left", fontsize=10)
    ax1.set_xlim(0, 1)

    # (b) Runout length ratio
    ax2.plot(t_star, L_star, color=c3, lw=2)
    ax2.set_ylabel("Relative runout length", fontsize=9)
    ax2.set_xlabel("Dimensionless time  t / tₑ", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("(b)", loc="left", fontsize=10)
    ax2.set_xlim(0, 1)

    # (c) Boxplots (final thickness, erosion, deposition)
    h_final = time_thick_pairs[-1][1]
    delta_h = h_final - h0
    erosion_map = np.clip(-delta_h, 0.0, None)
    deposition_map = np.clip(delta_h, 0.0, None)

    thick_vals = h_final[h_final > 0.05]
    ero_vals = erosion_map[erosion_map > 0.01]
    dep_vals = deposition_map[deposition_map > 0.01]

    data_log = [np.log10(thick_vals), np.log10(ero_vals), np.log10(dep_vals)]
    labels_bp = ["Final thickness", "Erosion depth", "Deposition depth"]

    bp = ax3.boxplot(
        data_log,
        labels=labels_bp,
        patch_artist=True,
        showfliers=False
    )
    cols_bp = [c1, c2, c3]
    for patch, col in zip(bp["boxes"], cols_bp):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)

    ax3.set_ylabel("log₁₀(depth [m])", fontsize=9)
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.set_title("(c)", loc="left", fontsize=10)
    for label in ax3.get_xticklabels():
        label.set_rotation(20)
        label.set_ha("right")

    for ax in (ax1, ax2, ax3):
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("Chosica debris-flow dynamics (dimensionless metrics & depth distributions)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    ts_png = os.path.join(CHART_DIR, "chosica_timeseries_boxplots.png")
    fig.savefig(ts_png, dpi=300)
    plt.close(fig)
    print(f"[INFO] Combined time-series + boxplot figure saved to: {ts_png}")

    # -----------------------------------------------------------------
    # Runout GIF
    # -----------------------------------------------------------------

    frame_paths_runout = []
    plt.ioff()
    print("\n[INFO] Creating runout GIF frames (hillshade, plasma thickness)...")

    for idx, (t, thick) in enumerate(time_thick_pairs):
        print(f"[INFO] Runout frame {idx+1}/{len(time_thick_pairs)} (t={t} s)...")

        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

        ax.imshow(hillshade_rgb, extent=extent, origin="upper", alpha=1.0)

        thick_masked = np.ma.masked_where(thick <= 1e-4, thick)
        im = ax.imshow(
            thick_masked,
            extent=extent,
            origin="upper",
            cmap="plasma",
            vmin=0,
            vmax=global_max_thick,
            alpha=0.9
        )

        ax.set_title(f"Chosica debris-flow runout – t = {t:.0f} s", fontsize=10)

        # Colourbar with half the map length
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.5)
        cbar.set_label("Flow thickness (m)", fontsize=9)

        ax.set_aspect("equal")
        apply_dms_ticks(ax, dem_profile["crs"])

        frame_path = os.path.join(FRAMES_RUNOUT_DIR, f"runout_{int(t):04d}.png")
        fig.tight_layout()
        fig.savefig(frame_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        frame_paths_runout.append(frame_path)

    print("\n[INFO] Assembling runout GIF...")
    images_runout = [imageio.imread(fp) for fp in sorted(frame_paths_runout)]
    imageio.mimsave(GIF_RUNOUT_PATH, images_runout, duration=0.08, loop=0)
    print(f"[INFO] Runout GIF written to: {GIF_RUNOUT_PATH}")

    # -----------------------------------------------------------------
    # Erosion / deposition maps & metrics
    # -----------------------------------------------------------------

    print("\n[INFO] Computing erosion/deposition proxy from runout results...")

    ed_profile = target_profile.copy()
    ed_profile.update(dtype="float32", count=1)

    erosion_path = os.path.join(ED_FOLDER, "erosion_depth_proxy.tif")
    deposition_path = os.path.join(ED_FOLDER, "deposition_depth_proxy.tif")

    with rasterio.open(erosion_path, "w", **ed_profile) as dst:
        dst.write(erosion_map.astype("float32"), 1)
    with rasterio.open(deposition_path, "w", **ed_profile) as dst:
        dst.write(deposition_map.astype("float32"), 1)

    print(f"[INFO] Erosion map saved to: {erosion_path}")
    print(f"[INFO] Deposition map saved to: {deposition_path}")

    V0_total = float(np.nansum(h0 * cell_area))
    Vf = float(np.nansum(h_final * cell_area))
    Vero = float(np.nansum(erosion_map * cell_area))
    Vdep = float(np.nansum(deposition_map * cell_area))

    print("\n=== VOLUME METRICS (for publication) ===")
    print(f"  Initial volume V0:         {V0_total: .3e} m³")
    print(f"  Final volume Vf:           {Vf: .3e} m³")
    print(f"  Erosion volume (proxy):    {Vero: .3e} m³")
    print(f"  Deposition volume (proxy): {Vdep: .3e} m³")
    print(f"  Mass balance Vf - V0:      {Vf - V0_total: .3e} m³")

    runout_length_m_final = float(np.nanmax(L_cummax))
    print("\n=== RUNOUT METRIC (FINAL) ===")
    print(f"  Runout length envelope (h>=0.1 m): {runout_length_m_final: .1f} m")

    # Hazard classification
    hazard_class, hz_stats = classify_hazard(h_final, cell_area, HZ_THRESH)

    hz_profile = target_profile.copy()
    hz_profile.update(dtype="int16", count=1, nodata=-1)

    hazard_path = os.path.join(ED_FOLDER, "hazard_class_thickness.tif")
    with rasterio.open(hazard_path, "w", **hz_profile) as dst:
        dst.write(hazard_class.astype("int16"), 1)
    print(f"\n[INFO] Hazard class map saved to: {hazard_path}")

    print("\n=== HAZARD AREA METRICS (thickness-based) ===")
    print("  Class 0 (h < {:.2f} m):      {:.3f} km²".format(HZ_THRESH[0], hz_stats[0]))
    print("  Class 1 ({:.2f}–{:.2f} m):   {:.3f} km²".format(HZ_THRESH[0], HZ_THRESH[1], hz_stats[1]))
    print("  Class 2 ({:.2f}–{:.2f} m):   {:.3f} km²".format(HZ_THRESH[1], HZ_THRESH[2], hz_stats[2]))
    print("  Class 3 ({:.2f}–{:.2f} m):   {:.3f} km²".format(HZ_THRESH[2], HZ_THRESH[3], hz_stats[3]))
    print("  Class 4 (>= {:.2f} m):       {:.3f} km²".format(HZ_THRESH[3], hz_stats[4]))

    # -----------------------------------------------------------------
    # Erosion/deposition GIF (static map)
    # -----------------------------------------------------------------

    print("\n[INFO] Creating erosion/deposition GIF (static map)...")

    max_ed = max(
        float(np.nanmax(erosion_map)) if np.nanmax(erosion_map) > 0 else 0.0,
        float(np.nanmax(deposition_map)) if np.nanmax(deposition_map) > 0 else 0.0
    )
    if max_ed == 0:
        max_ed = 1.0

    ed_field = deposition_map - erosion_map
    ed_masked = np.ma.masked_where(np.abs(ed_field) <= 1e-3, ed_field)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    ax.imshow(hillshade_rgb, extent=extent, origin="upper", alpha=1.0)

    div_cmap = mpl.colormaps["RdBu_r"]
    im_ed = ax.imshow(
        ed_masked,
        extent=extent,
        origin="upper",
        cmap=div_cmap,
        vmin=-max_ed,
        vmax=max_ed,
        alpha=0.8
    )

    ax.set_title("Chosica debris flow – Erosion (−) / Deposition (+) depth proxy", fontsize=10)

    cbar = fig.colorbar(im_ed, ax=ax, fraction=0.046, pad=0.02, shrink=0.5)
    cbar.set_label("Depth change (m, final − initial)", fontsize=9)

    ax.set_aspect("equal")
    apply_dms_ticks(ax, dem_profile["crs"])

    frame_static_ed = os.path.join(FRAMES_ED_DIR, "ED_000.png")
    fig.tight_layout()
    fig.savefig(frame_static_ed, bbox_inches="tight", dpi=300)
    plt.close(fig)

    image_ed = imageio.imread(frame_static_ed)
    imageio.mimsave(GIF_ED_PATH, [image_ed], duration=1.0, loop=0)

    print(f"[INFO] Erosion/Deposition GIF written to: {GIF_ED_PATH}")
    print("[INFO] Pipeline complete.")


if __name__ == "__main__":
    main()

