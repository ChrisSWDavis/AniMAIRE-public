# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

AniMAIRE is a Python 3.12 scientific library for calculating atmospheric ionizing radiation dose rates. It is a single-package project (not a monorepo) with no web server, Docker, or database dependencies.

### System dependencies

The following system packages must be installed (via `apt`) before pip install:

```
libgeos-dev libproj-dev gfortran libnetcdf-dev libhdf5-dev
```

These are required by Cartopy, SpacePy, NetCDF4, and related scientific C/Fortran extensions.

### Running commands

- **Lint**: `flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics` (matches CI; exit-zero so warnings don't fail the build)
- **Tests**: `GITHUB_ACTIONS=true MPLBACKEND=Agg pytest -v` (set `GITHUB_ACTIONS=true` to skip long-running integration tests that take 30+ minutes each; omit it to run the full suite)
- **Install (dev)**: `pip install -e .` after installing requirements

### Key gotchas

- **Python 3.12 only**: OTSOpy (core dependency) only supports Python 3.12. Do not use another version.
- **Headless matplotlib**: Set `MPLBACKEND=Agg` when running tests or scripts in headless environments to avoid display errors.
- **Magnetocosmics warning**: You will see an `ERROR:` line about Magnetocosmics not being installed on first import. This is benign — Magnetocosmics is optional; the default OTSO mode works without it.
- **PATH**: pip installs scripts to `~/.local/bin`. Ensure `export PATH="$HOME/.local/bin:$PATH"` is active.
- **Long tests**: Tests marked with `@pytest.mark.skipif(IN_GITHUB_ACTIONS, ...)` are CPU-intensive integration tests (30+ minutes each). Only run them when specifically needed.
- **Caching directories**: Running AniMAIRE creates `cachedOTSOData/` directories in the working directory for caching asymptotic direction results. These are safe to delete.
