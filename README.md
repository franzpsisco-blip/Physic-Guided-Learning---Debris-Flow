# Physics-Guided Learning (Debris Flow)

Code for the paper: *Debris Flow Hazard Assessment based on Multi-Sensor 3D displacements with InSAR and Physics-Guided Learning*.

## What each script does
- **`3D vel_v2.py`** → Decomposes **LOS velocities** into **3D velocities**.  
  **Input:** 4 LOS velocity maps (S1 asc/desc + TSX asc/desc) + geometry rasters (**incidence**, **heading**).  
  **Output:** `vE.tif`, `vN.tif`, `vU.tif` (+ residual/RMS and common mask).

- **`Modelo_chosica.py`** → **Physics‑Guided Learning model** (training + inference).  
  **Input:** covariate stack (terrain + hydroclimatic + geotechnical) + 3D velocities / kinematic indicators.  
  **Output:** class probabilities + susceptibility map + uncertainty + figures.

- **`simulacion_chosica.py`** → *(optional)* **runout simulation** (SynxFlow) driven by detected source areas and model outputs.

---

## Expected folder structure (example)

```
project/
  code/
    3D vel_v2.py
    Modelo_chosica.py
    simulacion_chosica.py

  data/
    insar/
      S1_asc/   vlos.tif  incidence.tif  heading.tif
      S1_desc/  vlos.tif  incidence.tif  heading.tif
      TSX_asc/  vlos.tif  incidence.tif  heading.tif
      TSX_desc/ vlos.tif  incidence.tif  heading.tif

    stacks/
      COVARIABLE_STACK.tif      # multi-band covariate stack (terrain + hydroclimatic + geotechnical)

  outputs/
    3d_vel/
      vE.tif
      vN.tif
      vU.tif
    model/
      rasters/
      figures/
    synxflow/   # optional
```

> All rasters must be co-registered (same CRS, resolution, extent, and grid alignment).

---

## Install (minimal)
Python 3.10+.

```bash
pip install numpy scipy rasterio matplotlib scikit-learn torch shap
# Optional for simulation:
pip install synxflow imageio
```

---

## Run (3 steps)

### 1) 3D velocity decomposition (TerraSAR‑X + Sentinel‑1)
Edit paths in **`3D vel_v2.py`**, then:
```bash
python "3D vel_v2.py"
```

### 2) Physics‑Guided Learning model (`Modelo_chosica.py`)

#### Required inputs (minimum)
1) **3D velocity rasters** (GeoTIFF):
- `vE.tif` (east velocity)
- `vN.tif` (north velocity)
- `vU.tif` (up velocity)

2) **Covariate stack** (`DEM_STACK.tif`, GeoTIFF, multi-band)  
At minimum, the stack should include the following covariate groups (band order is case‑study dependent; update band indices inside the script if needed):

**Terrain / hydro-geomorphic metrics**
- DEM (elevation)
- slope, aspect
- topographic position index (TPI)
- contributing drainage area (flow accumulation / specific contributing area)
- extracted drainage / hydrological network (if encoded as a raster layer)

**Hydroclimatic forcing/wetness proxies**
- normalized rainfall
- Topographic Wetness Index (**TWI**)

**Deposition-domain metrics**
- Multi‑Resolution Valley Bottom Flatness Index (**MRVBF**)

**Geotechnical stability covariates (for the Safety Factor, SF)**
- soil bulk density and texture fractions (clay/silt/sand) to derive spatially distributed geotechnical parameters
- (internally) soil depth and unit weights; local slope is also used in SF

**Kinematic indicators derived from 3D InSAR (from vE/vN/vU)**
- total velocity magnitude
- downslope direction/motion direction
- vertical-to-horizontal ratio (or equivalent vertical movement ratio)

> Practical requirement: `DEM_STACK.tif` and `vE/vN/vU` must share the same grid (CRS/resolution/extent/alignment) and consistent velocity units (e.g., mm/yr).

Run:
```bash
python Modelo_chosica.py
```

#### Outputs (typical)
- `outputs/model/rasters/` → `prob_*.tif`, `class_map.tif`, uncertainty maps
- `outputs/model/figures/` → summary figures (ROC/PR/SHAP if enabled)

---

### 3) Optional SynxFlow simulation (`simulacion_chosica.py`)

This step turns detected unstable areas into debris‑flow source zones and simulates runout/deposition.

#### Required inputs (minimum)
- **Topography:** a DEM (either from `DEM_STACK.tif` or a standalone `DEM.tif`, depending on the script config)
- **Source zones:** polygons/rasters derived from the **initial risk database** combined with the **high‑risk unstable areas detected by the PG model**
- **(Optional but recommended) 3D velocities / intensity proxy:** `vE.tif`, `vN.tif`, `vU.tif` if the script uses deformation to scale source strength

Run:
```bash
python simulacion_chosica.py
```

---

## Notes
- The scripts were originally configured with absolute Windows paths. For reproduction, update the path variables and (if needed) the `DEM_STACK` band indices used in `Modelo_chosica.py`.
