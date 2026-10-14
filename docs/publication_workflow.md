# Recommended public-release workflow

For a paper, keep the live project on GitHub and make a versioned archive on Zenodo or an institutional repository.

1. Create a public GitHub repository from the public-safe ZIP. Include source code, locked configuration, environment files, README instructions, small smoke data, figures, and compact result summaries.
2. Keep large or license-sensitive data outside GitHub. If redistribution is allowed, place the exact derived table and permitted source snapshots in a Zenodo release. If redistribution is not allowed, publish the source version/DOI and the extraction/validation scripts and ask readers to obtain the data directly.
3. Create a GitHub release after the manuscript version is frozen. Link that release to Zenodo so the archive receives a DOI. Cite the DOI in the manuscript's Code availability and Data availability statements.
4. Preserve the commit hash, configuration JSON, package versions, input SHA-256, and hardware/runtime notes in the archive. Runtime should be reported as implementation-dependent.

Suggested manuscript wording:

> The source code, locked experiment configurations, figures, and recorded results are available in the accompanying GitHub repository. The archived release is deposited in Zenodo and identified by its DOI. The derived station table is redistributed only where permitted by the source-data terms; otherwise, the repository provides the source metadata, version information, validation schema, and extraction protocol.

Do not publish API keys, private desktop paths, unpublished manuscript drafts, or raw products whose licenses do not permit redistribution.
