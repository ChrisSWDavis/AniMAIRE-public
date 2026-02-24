# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

AniMAIRE is a Python scientific library for calculating atmospheric ionizing radiation dose rates. It is a pure Python package with no web services, databases, or containers. See `README.md` for full documentation and usage examples.

### Python version

Python 3.12 is **required** — OTSOpy (the default geomagnetosphere engine) only supports 3.12.

### Environment variables

- `CDF_LIB=/usr/local/cdf/lib` must be set for SpacePy. It is configured in `~/.bashrc`.
- `PATH` must include `$HOME/.local/bin` for pip-installed CLI tools (flake8, pytest, etc.).

### System dependencies (pre-installed in VM snapshot)

The following system packages are required and installed via `apt-get`: `libgeos-dev`, `libproj-dev`, `proj-data`, `libhdf5-dev`, `libnetcdf-dev`, `libncurses-dev`, `gfortran`. NASA CDF library is built from source and installed to `/usr/local/cdf`.

### Lint, test, and run commands

- **Lint:** `flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics` (matches CI; `--exit-zero` means style issues are warnings only)
- **Test:** `pytest pytests/` — runs all tests. The `pytests/test_anisotropic_MAIRE_magnetocosmics.py` tests require the optional Magnetocosmics binary (not installed); these failures are expected and can be excluded with `--ignore=pytests/test_anisotropic_MAIRE_magnetocosmics.py`.
- **Run (hello-world):** See the "Testing that AniMAIRE is working" section in `README.md` for a quick validation script.

### Known test caveats

- `test_anisotropic_dose_rates` in `test_anisotropic_MAIRE.py` has a pre-existing assertion shape mismatch (8 vs 4 rows). It is marked `skipif(IN_GITHUB_ACTIONS)` and would be skipped in CI.
- The `AsympDirsCalculator` import always prints an error about missing Magnetocosmics — this is harmless; OTSO is the default engine and works without it.
- `spaceweather` may warn about stale local data files; run `spaceweather.update_data()` if fresh data is needed.
