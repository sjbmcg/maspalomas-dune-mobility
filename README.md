# Sand Dune Migration Analysis - Maspalomas

This repository combines exploratory remote-sensing analysis in a notebook with a simplified Werner-style dune simulation. The current focus is on 2020-2021 climate data, annual Sentinel-2 composites for 2020 and 2021, and Maspalomas DEM products used to contextualise dune morphology.

## Data Structure

The data folder contains:

**DTM_MASPOLOMAS** - DEM elevation data  
**maspalomas_simulation_config** - Werner model initial conditions  
**maspalomas_climate_data** - Open-Meteo hourly climate data for Maspalomas (2020-2021)

## Code

**sand_dune_simulation.ipynb** - Complete analysis notebook  
**brad_werner_sand_simulation.py** - Werner-style cellular automaton implementation with configurable CLI options  
**utils.json** - Configuration parameters for climate download and DEM metadata  
**requirements.txt** - Python package dependencies

## Installation

```
pip install -r requirements.txt
```

For the notebook, authenticate Google Earth Engine in your environment first. If your setup requires an explicit project, set `EE_PROJECT_ID` before launching Jupyter.

## Execution

Run `sand_dune_simulation.ipynb` sequentially. The notebook includes data download, quality checks, contextual figures, clustering-based movement estimates, phase-correlation diagnostics, and initial-condition generation for the Werner simulation.

The simulation script uses meteorological wind bearings, so `45` means wind blowing from the northeast toward the southwest. A reproducible non-interactive example is:

```
python brad_werner_sand_simulation.py --max-frames 100 --seed 42 --output-dir outputs
```

The script reads `data/maspalomas_simulation_config/2020_continuous_topoIC.csv` by default, renders the evolving height field with Pygame, and saves PNG snapshots for the configured frames.

## Reproducibility Notes

- The weather CSV is derived from Open-Meteo using the coordinates in `utils.json`. If you re-run the download cells, the notebook should regenerate the same file path.
- The notebook is still an exploratory research document, so stored outputs may lag behind source edits until the notebook is re-executed.
- `requirements.txt` is intentionally lightweight and currently unpinned, so exact package versions are not frozen in this repository.
