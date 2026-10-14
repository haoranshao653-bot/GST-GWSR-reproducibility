# Data-processing workflow

This repository includes the derived model table and the source products needed to inspect the published data path.

1. `data/raw/source_products/south_china_sea_heat_flow.csv` contains the clipped marine heat-flow source table used for the station filtering stage.
2. The table is restricted to 105–122°E and 0–25°N, retains offshore records, keeps 20–300 mW m⁻², removes values below 30 mW m⁻², and averages duplicate coordinates. The recorded counts are 1,177 source rows, 1,163 rows after the 30 mW m⁻² threshold, and 1,024 station locations after duplicate averaging.
3. The eight predictor grids in `data/raw/predictor_grids/` are the harmonized 0.25-degree tables used by the final model. Continuous variables are bilinearly interpolated at station coordinates. Volcano distance is computed from the GVP Holocene volcano catalogue with a spherical Earth radius of 6,371 km.
4. Run:

   ```powershell
   python scripts/prepare_dataset.py `
     --raw-heat-flow data/raw/source_products/south_china_sea_heat_flow.csv `
     --grid-root data/raw/predictor_grids `
     --output data/processed/heat_flow_stations_rebuilt.csv `
     --audit run_outputs/prepare_dataset_audit.json
   ```

5. Compare the rebuilt table with `data/processed/heat_flow_stations.csv`. The latter is the exact table used for the recorded experiments. The source grids, interpolation libraries, and floating-point versions can produce small last-decimal differences; the recorded table remains the authoritative experiment input.

The source products and metadata are kept under `data/raw/source_products/`. Their provider attributions and download records should be preserved when the repository is transferred or later made public.
