# Private handoff to the advisor

This package is intended to be shared before submission through a private GitHub repository or the complete ZIP archive.

## Private GitHub route

1. Create an empty private repository on GitHub.
2. From this project directory, add the remote and push the committed history:

   ```powershell
   git remote add origin https://github.com/<account>/<repository>.git
   git push -u origin main
   git lfs push --all origin main
   ```

3. In GitHub repository settings, invite the advisor as a collaborator.
4. Ask the advisor to clone with Git LFS enabled:

   ```powershell
   git clone https://github.com/<account>/<repository>.git
   cd <repository>
   git lfs pull
   ```

5. The advisor can first run `python scripts/check_environment.py`, `python scripts/validate_station_table.py --input data/processed/heat_flow_stations.csv`, and `python scripts/run_smoke.py` before reviewing the full experiments.

The repository has two commits: the initial reproducibility package and the complete private data/processing addition. The current head contains the exact final table and all source files used for data preparation.

## ZIP route

Use `GST-GWSR-private-advisor-v2.zip` when the advisor prefers a direct file transfer. The archive contains the actual large raster files, not LFS pointer files. It excludes the local virtual environment and temporary run outputs.

## Later public release

After the manuscript and data-release decision are settled, create a tagged GitHub Release and archive that exact tag in Zenodo. Review all Git history before changing the repository visibility from private to public.
