# Sand Dune Migration Analysis - Maspalomas

This repository combines exploratory remote-sensing analysis in a notebook with a simplified Werner-style dune simulation. The current focus is on 2020-2021 climate data, annual Sentinel-2 composites for 2020 and 2021, and Maspalomas DEM products used to contextualise dune morphology.

## Data Structure

The data folder contains:

**DTM_MASPOLOMAS** - DEM elevation rasters for terrain context  
**maspalomas_simulation_config** - Werner model initial conditions  
**maspalomas_climate_data** - raw weather data, cleaned weather data, and provenance files

## Code

**sand_dune_simulation.ipynb** - Complete analysis notebook  
**brad_werner_sand_simulation.py** - Werner-style cellular automaton implementation with configurable CLI options  
**analysis_config.json** - Central configuration for weather inputs, DEM paths, and simulation artifacts  
**requirements.txt** - Minimal runtime dependencies

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

- `analysis_config.json` is the single source of truth for weather paths, DEM paths, and the default Werner initial-condition file.
- The weather folder contains four distinct artifacts: the raw Open-Meteo pull, the cleaned analysis file, request metadata, and a quality-summary JSON. Those files are intentionally kept together.
- `requirements.txt` is the maintained install surface for the project. Local cache files and ad hoc environment dumps are intentionally not tracked.
- The notebook is still an exploratory research document, so stored outputs may lag behind source edits until the notebook is re-executed.
