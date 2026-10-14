# Data and redistribution boundary

The experiment uses 1,024 deduplicated marine heat-flow stations from the IHFC source and eight derived predictors:

`topo`, `bouguer`, `sediment`, `moho`, `lab`, `cpd`, `lithos_km`, and `volcano`.

The public package includes `raw/IHFC.json`, a source metadata snapshot with DOI `10.5880/fidgeo.2024.014`, and `raw/china_provinces.geojson` for map context. Check the current terms of those providers before redistributing them in a public mirror. The raw IHFC metadata does not contain the final eight-predictor station table used by the model.

The repository includes `processed/heat_flow_stations.csv`, the exact 1,024-row table used by the recorded experiments. The study used WGS 84 coordinates and predictors extracted from GEBCO, WGM2012, GlobSed v3, CRUST1.0, CAM2016Litho, the Curie-point-depth supplementary product, LITHO1.0, and the Smithsonian Global Volcanism Program. Their exact names, units, and extraction description are recorded in the manuscript's Section 3.1 and `docs/data_sources.md`.

Required columns:

```text
lon, lat, heat_flow, topo, bouguer, sediment, moho, lab,
cpd, lithos_km, volcano
```

Rows must be complete and `heat_flow` must be positive because the final protocol uses a log response transform. The validation script reports problems without changing the data.
