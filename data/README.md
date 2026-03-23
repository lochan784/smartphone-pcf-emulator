# Data Description

## verified_devices.csv
- **Source**: Manufacturer Environmental Product Declarations (Apple, Samsung, Google) – publicly available documents.
- **Content**: 13 devices with hardware specifications and verified Product Carbon Footprint (PCF) in kg CO₂e.
- **Columns**: `device`, `brand`, `year`, `battery_mAh`, `display_in`, `mass_g`, `chip_node`, `PCF`, and other derived features used in the model.
- **Usage**: This is the gold‑standard holdout set used for evaluation and conformal calibration.

## catalog_devices.csv
- **Source**: Kaggle dataset "Smartphone Specifications and Prices" (link: [https://www.kaggle.com/datasets/...)](https://www.kaggle.com/datasets/devgondaliya007/smartphone-specifications-dataset?utm_source=chatgpt.com). Please replace with the actual Kaggle URL or citation.
- **Content**: 968 devices with detailed hardware specifications (no PCF values).
- **Columns**: Similar to `verified_devices.csv` but without the `PCF` column. Includes brand, model, release year, battery capacity, display size, mass, chip details, etc.
- **Usage**: Used for feature engineering, generating synthetic training data, and large‑scale scenario simulations.

Both datasets are provided as‑is for reproducibility. Please refer to the original sources for any licensing or attribution requirements.
