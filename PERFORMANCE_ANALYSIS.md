# AniMAIRE Performance Analysis: Slow Components Before and After Asymptotic Direction Caching

This document analyzes the performance bottlenecks in the AniMAIRE codebase, examining both cold runs (before caching) and warm runs (after asymptotic directions are cached).

---

## Executive Summary

**Before caching:** The raw asymptotic direction acquisition (OTSO/Magnetocosmics) dominates runtime—often on the order of tens of minutes to hours per full world map. Secondary costs include row-by-row pandas operations for pitch angle conversion and weighting factor calculation.

**After caching:** The pitch angle conversion and weighting factor calculation become the dominant costs, as they are **not** cached. The `generate_asymp_dir_DF` function is always called with `cache=False` from `generalEngineInstance`, so the `convertAsymptoticDirectionsToPitchAngle` joblib cache is never used in the main pipeline.

---

## Pipeline Overview

The calculation flow is:

1. **Acquire raw asymptotic directions** (OTSO.planet, Magnetocosmics, or file)
2. **Convert to pitch angles** (`generate_asymp_dir_DF` → `convertAsymptoticDirectionsToPitchAngle`)
3. **Acquire weighting factors** (`acquireWeightingFactors`)
4. **Compute dose rates** (`singleParticleEngineInstance.runOverSpecifiedAltitudes`)
5. **Repeat for each particle species** (proton, alpha, etc.)

---

## Part 1: Before Asymptotic Directions Are Cached (Cold Run)

### 1.1 Raw Asymptotic Direction Acquisition — **DOMINANT COST**

**Location:**  
- `generalEngineInstance.acquireDFofAllAsymptoticDirections()`  
- OTSO path: `otso_planet_processing.create_and_convert_full_planet()`  
- Magnetocosmics path: `AsympDirsTools.get_magcos_asymp_dirs()`

**Why it's slow:**
- **OTSO mode:** Calls `OTSO.planet()` (external Fortran) for each zenith/azimuth pair. With default 9 zeniths/azimuths, this means 9 runs. Each run computes asymptotic directions for:
  - All lat/lon grid points (default 37×72 = 2,664 locations)
  - 260 rigidity levels (60 high + 200 low)
  - Total: ~693,000+ asymptotic direction calculations per run × 9 runs
- **Magnetocosmics mode:** Similar scale; MAGNETOCOSMICS performs full particle tracing through the geomagnetic field for each (location, rigidity, zenith, azimuth) combination.
- **Isotropic fast path:** Uses `RigidityPredictor.batch_predict()` instead—ML inference, much faster but only for isotropic PAD with `use_fast_calculation=True`.

**Typical runtime:** 20–60+ minutes for a full world map (from README: "about 40 minutes per full world dose rate map").

---

### 1.2 Pitch Angle Conversion — **SIGNIFICANT COST**

**Location:**  
`AsymptoticDirectionProcessing.convertAsymptoticDirectionsToPitchAngle()` (lines 86–110)

```python
pitch_angle_list = get_apply_method(dataframeToFillFrom)(
    lambda dataframe_row: get_pitch_angle_for_DF_analytic(
        IMFlatitude, IMFlongitude, 
        dataframe_row["Lat"], dataframe_row["Long"]
    ), axis=1
)
```

**Why it's slow:**
- Uses `DataFrame.apply(..., axis=1)` (or `parallel_apply` via pandarallel), which invokes a Python callable for **every row**.
- With ~500,000–5,000,000 rows (depending on grid size and zenith/azimuth count), this means hundreds of thousands of Python function calls.
- Although `get_pitch_angle_for_DF_analytic` is Numba JIT-compiled, the Python call overhead and DataFrame row access dominate.
- No vectorization: the formula is a simple analytic expression and could be evaluated on full numpy arrays in one pass.

**Estimated impact:** Minutes for large dataframes.

---

### 1.3 Weighting Factor Calculation — **SIGNIFICANT COST**

**Location:**  
`AsymptoticDirectionProcessing.acquireWeightingFactors()` (lines 144–232)

**Slow operations:**
1. **Pitch angle weighting** (non-isotropic path, ~line 195):  
   `get_apply_method(new_asymptotic_direction_DF)(pitchAngleFunctionToUse, axis=1)`  
   - Row-by-row apply over full dataframe.
2. **Rigidity weighting** (~line 200):  
   `get_apply_method(new_asymptotic_direction_DF["Rigidity"])(momentaDist.getRigiditySpectrum())`  
   - Apply over Rigidity column only (slightly cheaper).
3. **Combined rigidity+pitch weighting** (~line 203):  
   `get_apply_method(new_asymptotic_direction_DF)(fullRigidityPitchWeightingFactorFunctionToUse, axis=1)`  
   - Another full row-by-row apply.

Each of these is O(N) with Python call overhead. The isotropic fast path skips some of these but still does rigidity weighting and energy conversion.

---

### 1.4 OTSO Planet Conversion — **MODERATE COST**

**Location:**  
`otso_planet_processing.convert_planet_df_to_asymp_format()` (lines 41–76)

```python
for _, row in df.iterrows():
    ...
    for energy_col in energy_columns:
        ...
```

**Why it's slow:**
- `iterrows()` is one of the slowest pandas iteration patterns.
- Nested loop over rows and energy columns.
- Could be vectorized or at least use `.itertuples()` for speed.

---

### 1.5 Dose Rate Calculation — **SIGNIFICANT COST**

**Location:**  
`singleParticleEngineInstance.calc_output_dose_flux()` (lines 84–107)

```python
DFofSpectraForEachCoord = asymp_dir_DF_with_weighting_factors.groupby(
    ["initialLatitude","initialLongitude"]
).apply(spectrum_to_function_conversion_function)

outputDoseRatesForAltitudeRange = get_apply_method(DFofSpectraForEachCoord)(
    lambda spectrum: DAFcalc.calculate_from_rigidity_spec(...)
)
```

**Why it's slow:**
- `groupby().apply()` creates one interpolator per (lat, lon) and one dose calculation per location.
- Default grid: 2,664 locations → 2,664 calls to `DAFcalc.calculate_from_rigidity_spec()`.
- `DAFcalc` (atmosphericRadiationDoseAndFlux) is external; cost depends on its implementation.

---

### 1.6 RigidityPredictor Batch — **MODERATE COST (Isotropic Fast Path Only)**

**Location:**  
`rigidity_predictor.py` `batch_predict()` (lines 193–216)

```python
for _, row in df.iterrows():
    features = self._prepare_features(...)
    features_list.append(features[0])
```

**Why it's slow:**
- `iterrows()` over ~2,664 rows.
- Could be vectorized: feature preparation and prediction are applied uniformly.

---

## Part 2: After Asymptotic Directions Are Cached (Warm Run)

### What Gets Cached

- **OTSO mode:** `OTSOmemory.cache(create_and_convert_planet)` — raw OTSO output is cached.
- **Magnetocosmics mode:** `MAGCOSmemory` in AsympDirsCalculator — raw Magnetocosmics output is cached.
- Both use joblib `Memory`; cache hits return results from disk with minimal recomputation.

### Critical Observation: Pitch Angle Conversion Is Not Cached

In `generalEngineInstance.acquireDFofAllAsymptoticDirections()` (line 222–227):

```python
processed_df = generate_asymp_dir_DF(
    raw_asymp_df,
    self.reference_latitude,
    self.reference_longitude,
    self.date_and_time,
    cache=False   # <-- Always False!
)
```

So even when raw asymptotic directions are cached, **pitch angle conversion runs on every call**. The joblib cache in `generate_asymp_dir_DF` (when `cache=True`) is never used in the main pipeline.

---

### 2.1 Dominant Costs After Caching

1. **Pitch angle conversion** — unchanged; still full cost every run.
2. **Weighting factor calculation** — unchanged; still full cost every run.
3. **Dose rate calculation** — unchanged; still full cost every run.
4. **OTSO planet conversion** — only runs when OTSO result is not in cache; on cache hit, `create_and_convert_planet` is skipped, but when used, conversion still costs.

---

## Part 3: Optimization Opportunities

### High Impact

| Component | Current Approach | Suggested Improvement |
|-----------|------------------|----------------------|
| Pitch angle conversion | Row-by-row `apply` with Numba function | Vectorize: `get_pitch_angle_for_DF_analytic` can be written as a Numba function that operates on `Lat` and `Long` numpy arrays directly |
| `generate_asymp_dir_DF` caching | Always `cache=False` from `generalEngineInstance` | Pass `cache=self.cache_magnetocosmics_runs` (or similar) so identical (raw_asymp_df, IMF, datetime) reuse cached pitch angles |
| acquireWeightingFactors | Multiple row-by-row applies | Vectorize where possible: rigidity spectrum and many PADs (e.g., isotropic, power law) can work on full arrays; create `@numba.njit` array versions |

### Medium Impact

| Component | Current Approach | Suggested Improvement |
|-----------|------------------|----------------------|
| `convert_planet_df_to_asymp_format` | `iterrows()` | Use `itertuples()` or vectorized operations; avoid nested Python loops over rows |
| RigidityPredictor `batch_predict` | `iterrows()` | Build feature matrix with vectorized numpy operations, single `model.predict()` call |
| Dose calculation per location | Sequential apply | Consider parallelization (e.g., `joblib.Parallel`) over lat/lon groups |

### Lower Impact

| Component | Current Approach | Suggested Improvement |
|-----------|------------------|----------------------|
| `get_apply_method` | Chooses progress_apply vs parallel_apply | Ensure pandarallel is properly configured; consider `swifter` or `modin` for larger dataframes |

---

## Part 4: Data Scale Reference

- **Default grid:** 37 lats × 72 lons = 2,664 locations.
- **Default rigidities:** 16 high + 200 low = 216 levels.
- **Single zenith/azimuth:** ~575,000 rows.
- **9 zeniths/azimuths:** ~5.2 million rows (before `get_mean_weighting_factors_for_multi_angle_magcos_runs` groups them).

---

## Summary Table

| Stage | Before Cache | After Cache |
|-------|--------------|-------------|
| Raw asymptotic directions | **Dominant (minutes–hours)** | **Cached (seconds)** |
| Pitch angle conversion | Significant (minutes) | Significant (minutes) |
| Weighting factors | Significant (minutes) | Significant (minutes) |
| Dose calculation | Significant | Significant |
| OTSO conversion | Moderate | Skipped if cached |

**Takeaway:** Enabling pitch angle caching and vectorizing the pitch angle and weighting factor calculations would substantially reduce runtime on warm runs and cold runs alike.
