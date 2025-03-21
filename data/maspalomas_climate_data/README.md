## Maspalomas Climate Files

- `weather_data_2020_2021.csv`: raw hourly Open-Meteo download used as the base climate input.
- `weather_data_2020_2021_cleaned.csv`: analysis-ready weather file after the notebook's basic data-quality adjustments.
- `weather_data_2020_2021_metadata.json`: request parameters and resolved Open-Meteo grid metadata for the raw pull.
- `weather_data_2020_2021_quality_summary.json`: compact summary of the cleaning checks applied to the hourly series.

The notebook reads the file locations from [analysis_config.json](../../analysis_config.json).
