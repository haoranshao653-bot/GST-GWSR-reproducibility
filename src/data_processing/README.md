# Data-processing code

`prepare_dataset.py` is the deterministic final-table builder. It takes the clipped IHFC heat-flow CSV and the eight harmonized 0.25-degree predictor grids, applies the study-area and quality filters, averages duplicate coordinates, interpolates the continuous grids, computes volcano distance from the GVP catalogue, and writes the 1,024-row model-ready table.

The other scripts in this directory are the source-data extraction, screening, and quality-audit programs used during feature construction. They are retained so that the exploratory decisions are reviewable. The final model uses the eight columns listed in `configs/final_outer_test.json`; exploratory EMAG2, active-fault, and 3D-SCS products are not silently added to the final model.

All paths in the original desktop scripts are historical provenance. For a clean rerun, use repository-relative inputs and the commands documented in `docs/data_processing_workflow.md`.
