import os
import numpy as np
import rasterio

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\02_grid-20250906T025757Z-1-001\02_grid"

DATASETS = [
    {
        "name": "S1_asc",
        "folder": os.path.join(BASE_DIR, "S1_asc"),
        "vlos": "S1_asc_vlos.tif",
        "inc": "incidence.tif",
        "head": "heading.tif",
    },
    {
        "name": "S1_desc",
        "folder": os.path.join(BASE_DIR, "S1_desc"),
        "vlos": "S1_sec_vlos.tif",
        "inc": "incidence.tif",
        "head": "heading.tif",
    },
    {
        "name": "TSX_asc",
        "folder": os.path.join(BASE_DIR, "TSX_asc"),
        "vlos": "TSX_asc_vlos.tif",
        "inc": "incidence.tif",
        "head": "heading.tif",
    },
    {
        "name": "TSX_desc",
        "folder": os.path.join(BASE_DIR, "TSX_desc"),
        "vlos": "TSX_desc_vlos.tif",
        "inc": "incidence.tif",
        "head": "heading.tif",
    },
]

OUT_EAST   = os.path.join(BASE_DIR, "vel_east_4LOS_common_noscale.tif")
OUT_NORTH  = os.path.join(BASE_DIR, "vel_north_4LOS_common_noscale.tif")
OUT_UP     = os.path.join(BASE_DIR, "vel_up_4LOS_common_noscale.tif")
OUT_RESID  = os.path.join(BASE_DIR, "residual_rms_4LOS_common_noscale.tif")
OUT_MASK   = os.path.join(BASE_DIR, "common_pixels_mask_4LOS.tif")

# How is incidence defined in your rasters?
#   "vertical"  : 0° = up, 90° = horizontal
#   "horizontal": 0° = horizontal, 90° = up
INC_FROM = "vertical"   # if geometry looks weird, try "horizontal"

# Condition number threshold to reject very bad geometry
COND_THRESH = 1000.0

# ============================================================
# GEOMETRY
# ============================================================

def los_unit_vector(inc_deg, head_deg, inc_from="vertical"):
    """
    Compute LOS unit vector (East, North, Up) from incidence & heading.

    LOS direction is from ground to satellite, in ENU components.
    """
    inc = np.deg2rad(inc_deg)
    az  = np.deg2rad(head_deg - 90.0)  # right-looking SAR

    if inc_from == "vertical":
        # incidence from vertical (0° = up, 90° = horizontal)
        r_h = np.sin(inc)  # horizontal component magnitude
        u   = np.cos(inc)  # vertical component
    elif inc_from == "horizontal":
        # incidence from horizontal (0° = horizontal, 90° = up)
        r_h = np.cos(inc)
        u   = np.sin(inc)
    else:
        raise ValueError("inc_from must be 'vertical' or 'horizontal'")

    e = r_h * np.sin(az)
    n = r_h * np.cos(az)
    return e, n, u

# ============================================================
# I/O HELPERS
# ============================================================

def load_raster(path):
    print(f"\n[INFO] Loading raster: {path}")
    with rasterio.open(path) as src:
        data = src.read(1).astype("float32")
        nodata = src.nodata
        print(f"       Shape: {data.shape}  Nodata: {nodata}")
        if nodata is not None:
            mask = data == nodata
            data[mask] = np.nan
            print(f"       Pixels equal to nodata: {mask.sum()}")
        print(f"       Total NaNs: {np.isnan(data).sum()}")
        meta = src.meta.copy()
    return data, meta

def write_raster_float(template_meta, out_path, data, description=None):
    print(f"\n[INFO] Writing raster: {out_path}")
    meta = template_meta.copy()
    meta.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="lzw",
    )
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data.astype("float32"), 1)
        if description is not None:
            dst.set_band_description(1, description)
    print("       Done.")

def write_raster_byte(template_meta, out_path, data, description=None):
    print(f"\n[INFO] Writing raster: {out_path}")
    meta = template_meta.copy()
    meta.update(
        dtype="uint8",
        count=1,
        nodata=0,
        compress="lzw",
    )
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data.astype("uint8"), 1)
        if description is not None:
            dst.set_band_description(1, description)
    print("       Done.")

# ============================================================
# MAIN
# ============================================================

def main():
    print("========== 3D VELOCITY FROM 4 LOS (COMMON PIXELS, NO SCALE) ==========")
    print(f"[CONFIG] INC_FROM    = {INC_FROM}")
    print(f"[CONFIG] COND_THRESH = {COND_THRESH}")
    print("=====================================================================")

    # 1) Load vlos for all 4 sensors (NO scaling)
    for ds in DATASETS:
        v_path = os.path.join(ds["folder"], ds["vlos"])
        v_data, meta = load_raster(v_path)
        ds["vlos_data"] = v_data

        total = v_data.size
        nan_cnt = np.isnan(v_data).sum()
        print(f"[STATS] {ds['name']}: NaN ratio in vlos = {nan_cnt / total:.4f}")

    shapes = [ds["vlos_data"].shape for ds in DATASETS]
    if len(set(shapes)) != 1:
        print("[ERROR] vlos rasters have different shapes:")
        for ds, sh in zip(DATASETS, shapes):
            print(f"       {ds['name']}: {sh}")
        return
    height, width = shapes[0]
    print(f"\n[INFO] All vlos rasters have shape: {height} x {width}")

    # 2) Load incidence & heading for all 4
    for ds in DATASETS:
        inc_path  = os.path.join(ds["folder"], ds["inc"])
        head_path = os.path.join(ds["folder"], ds["head"])

        inc_data, _  = load_raster(inc_path)
        head_data, _ = load_raster(head_path)

        if inc_data.shape != (height, width) or head_data.shape != (height, width):
            print("[ERROR] Incidence/heading shape mismatch in", ds["name"])
            print("       inc shape: ", inc_data.shape,
                  " head shape:", head_data.shape,
                  " expected:", (height, width))
            return

        ds["inc_data"]  = inc_data
        ds["head_data"] = head_data

    # 3) Precompute LOS unit vectors
    print("\n[INFO] Precomputing LOS unit vectors (ENU) for all 4 datasets...")
    for ds in DATASETS:
        e, n, u = los_unit_vector(ds["inc_data"], ds["head_data"], inc_from=INC_FROM)
        ds["los_e"] = e.astype("float32")
        ds["los_n"] = n.astype("float32")
        ds["los_u"] = u.astype("float32")
        print("   -", ds["name"], "LOS components ready")

    # 4) Build common-pixel mask = valid in ALL 4 (vlos, inc, head)
    print("\n[INFO] Computing common pixel mask (all 4 LOS & geometry valid)...")
    common_mask = np.ones((height, width), dtype=bool)
    for ds in DATASETS:
        valid = (
            np.isfinite(ds["vlos_data"]) &
            np.isfinite(ds["inc_data"]) &
            np.isfinite(ds["head_data"])
        )
        common_mask &= valid

    n_common = common_mask.sum()
    print(f"       Common pixels across all 4 datasets: {n_common} "
          f"({n_common / (height * width):.4f} of total)")

    # 5) Allocate outputs
    print("\n[INFO] Allocating output arrays...")
    vel_e   = np.full((height, width), np.nan, dtype="float32")
    vel_n   = np.full((height, width), np.nan, dtype="float32")
    vel_u   = np.full((height, width), np.nan, dtype="float32")
    resid_r = np.full((height, width), np.nan, dtype="float32")

    # 6) Per-pixel inversion using ALL 4 LOS at common pixels (NO scaling)
    print("\n[INFO] Solving 3D velocities (4 LOS, common pixels only, no scaling)...")
    n_solved = 0

    for i in range(height):
        if i % 100 == 0:
            print(f"   Row {i}/{height} ({100.0 * i / height:.1f}%)")
        for j in range(width):
            if not common_mask[i, j]:
                continue

            rows   = []
            b_vals = []

            for ds in DATASETS:
                v_ij  = ds["vlos_data"][i, j]
                e_ij  = ds["los_e"][i, j]
                n_ij  = ds["los_n"][i, j]
                u_ij  = ds["los_u"][i, j]

                # all finite here because of common_mask
                rows.append([e_ij, n_ij, u_ij])
                b_vals.append(v_ij)

            A = np.array(rows, dtype="float64")  # 4 x 3
            b = np.array(b_vals, dtype="float64")  # 4

            # geometry quality
            try:
                condA = np.linalg.cond(A)
            except np.linalg.LinAlgError:
                continue
            if condA > COND_THRESH:
                # geometry too ill-conditioned -> leave NaN
                continue

            # least squares solution: best-fit 3D model for 4 LOS
            try:
                v_sol, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)
                if rank < 3:
                    continue
            except np.linalg.LinAlgError:
                continue

            vel_e[i, j] = v_sol[0]
            vel_n[i, j] = v_sol[1]
            vel_u[i, j] = v_sol[2]
            n_solved += 1

            # residual RMS in same units as input vlos
            b_fit = A @ v_sol
            res   = b - b_fit
            rms   = np.sqrt(np.mean(res**2))
            resid_r[i, j] = rms

    print("\n[INFO] Inversion finished.")
    print("       Common pixels (input):     ", n_common)
    print("       Pixels successfully solved:", n_solved)
    if n_common > 0:
        print("       Fraction of common solved: ", n_solved / n_common)

    # 7) Write outputs
    write_raster_float(meta, OUT_EAST,  vel_e,   "East_velocity_4LOS_common_noscale")
    write_raster_float(meta, OUT_NORTH, vel_n,   "North_velocity_4LOS_common_noscale")
    write_raster_float(meta, OUT_UP,    vel_u,   "Up_velocity_4LOS_common_noscale")
    write_raster_float(meta, OUT_RESID, resid_r, "Residual_RMS_4LOS_common_noscale")
    write_raster_byte (meta, OUT_MASK,  common_mask.astype("uint8"),
                       "Common_pixel_mask_4LOS")

    print("\n[INFO] All done!")
    print("       Output files:")
    print("          ", OUT_EAST)
    print("          ", OUT_NORTH)
    print("          ", OUT_UP)
    print("          ", OUT_RESID)
    print("          ", OUT_MASK)
    print("=====================================================================")


if __name__ == "__main__":
    main()
