# Physics-Guided Learning (Debris Flow)

Repo for th paper: *Debris Flow Hazard Assessment based on Multi-Sensor 3D displacements with InSAR and Physics-Guided Learning*.

## What each script does
- **`3D vel_v2.py`** → **YES: decomposes LOS velocities into 3D velocities**.  
  Input: 4 LOS velocity maps (S1 asc/desc + TSX asc/desc) + their geometry (incidence + heading).  
  Output: **`vE.tif`, `vN.tif`, `vU.tif`** (+ residual/RMS and common mask).

- **`Modelo_chosica.py`** → **THIS is the Physics‑Guided Learning model** (training + inference).  
  Input: terrain/conditioning stack + `vE/vN/vU`.  
  Output: class probabilities + final susceptibility map + uncertainty + figures.

- **`simulacion_chosica.py`** → optional **runout simulation** (SynxFlow) using the ML output (and DEM/velocities).

## Expected folder structure (example)
Put your data anywhere; just edit the path variables at the top of each script. A simple structure is:

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
      DEM_STACK.tif   # multi-band conditioning stack (DEM/slope/TWI/NDVI/rain proxy/etc.)

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

> All rasters must be co-registered (same CRS/resolution/extent).

## Install (minimal)
Python 3.10+.

```bash
pip install numpy scipy rasterio matplotlib scikit-learn torch shap
# Optional for simulation:
pip install synxflow imageio
```

## Run (3 steps)
### 1) 3D velocity decomposition using 4 sensors TERRASAR-X + SENTINEL-1
Edit paths in **`3D vel_v2.py`** (folders + file names), then:
```bash
python "3D vel_v2.py"
```

### 2) Physics-guided learning model 
Edit paths in **`Modelo_chosica.py`** (`DEM_STACK`, `vE/vN/vU`, `OUT_DIR`), then:
```bash
python Modelo_chosica.py
```

### 3) Optional SynxFlow simulation
Edit paths in **`simulacion_chosica.py`**, then:
```bash
python simulacion_chosica.py
```

## Notes
- The scripts were originally configured with absolute Windows paths (case study: Chosica). For reproduction, only update the path variables and (if needed) the DEM_STACK band indices inside `Modelo_chosica.py`.
