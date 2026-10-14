# Data-source record

The response values originate from the International Heat Flow Commission / Global Heat Flow Database source represented in `data/raw/IHFC.json` (DOI: `10.5880/fidgeo.2024.014`). The study-area table was clipped to 105–122°E and 0–25°N, restricted to marine records, filtered by the stated heat-flow quality rules, and averaged at duplicate coordinates.

Predictors were extracted from the following source products and then joined to the stations in WGS 84:

| Predictor | Code | Source recorded by the study |
|---|---|---|
| Seafloor topographic elevation | `topo` | GEBCO grid |
| Bouguer gravity anomaly | `bouguer` | WGM2012 / BGI |
| Sediment thickness | `sediment` | GlobSed v3 |
| Moho depth | `moho` | CRUST1.0 |
| Lithosphere–asthenosphere boundary depth | `lab` | CAM2016Litho |
| Curie-point depth | `cpd` | supplementary product documented in the manuscript |
| Lithospheric thickness | `lithos_km` | LITHO1.0 |
| Distance to nearest Holocene volcano | `volcano` | Smithsonian Global Volcanism Program |

This table is a provenance record, not a grant of redistribution rights. A public release should retain each provider's attribution and license statement and should distribute only products permitted by those terms. If a source cannot be redistributed, publish the download URL/version and the local extraction script instead.
