```markdown
# Sand Dune Migration Analysis - Maspalomas

## Data Structure

The data folder contains:

**DTM_MASPOLOMAS** - DEM elevation data  
**maspalomas_simulation_config** - Werner model initial conditions  
**climate** - Weather station data (2020-2021)

## Code

**sand_dune_simulation.ipynb** - Complete analysis notebook  
**brad_werner_sand_simulation.py** - Cellular automaton implementation  
**utils.json** - Configuration parameters (API credentials, coordinates, file paths)  
**requirements.txt** - Python package dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Execution

Run `sand_dune_simulation.ipynb` sequentially. Werner simulation requires initial conditions from `data/maspalomas_simulation_config/`. To run the simulation, open PowerShell or Command Prompt and execute:

```batch
python brad_werner_sand_simulation.py
```
```