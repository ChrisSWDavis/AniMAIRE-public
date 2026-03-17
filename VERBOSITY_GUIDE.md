# AniMAIRE verbosity guide

AniMAIRE now runs in quiet mode by default (`verbose=False`) and only emits detailed progress logs when `verbose=True`.

## Core mode: `run_from_spectra`

```python
from AniMAIRE import AniMAIRE
import datetime as dt

result = AniMAIRE.run_from_spectra(
    proton_rigidity_spectrum=lambda x: 2.56 * (x ** -3.41),
    Kp_index=3,
    date_and_time=dt.datetime(2006, 12, 13, 3, 0),
    array_of_lats_and_longs=[[65.0, 25.0]],
    altitudes_in_km=[0.0],
    verbose=False,  # default
)
```

Switch to detailed logs:

```python
result = AniMAIRE.run_from_spectra(
    proton_rigidity_spectrum=lambda x: 2.56 * (x ** -3.41),
    Kp_index=3,
    date_and_time=dt.datetime(2006, 12, 13, 3, 0),
    array_of_lats_and_longs=[[65.0, 25.0]],
    altitudes_in_km=[0.0],
    verbose=True,
)
```

## Power-law Gaussian mode

```python
import numpy as np

result = AniMAIRE.run_from_power_law_gaussian_distribution(
    J0=256_000.0,
    gamma=3.41,
    deltaGamma=0.22,
    sigma=np.sqrt(0.19),
    reference_pitch_angle_latitude=-17.0,
    reference_pitch_angle_longitude=148.0,
    Kp_index=3,
    date_and_time=dt.datetime(2006, 12, 13, 3, 0),
    array_of_lats_and_longs=[[65.0, 25.0]],
    altitudes_in_km=[0.0],
    verbose=True,
)
```

## DLR cosmic-ray model mode

```python
result = AniMAIRE.run_from_DLR_cosmic_ray_model(
    OULU_count_rate_in_seconds=106.0,
    Kp_index=3,
    date_and_time=dt.datetime(2006, 12, 13, 3, 0),
    array_of_lats_and_longs=[[65.0, 25.0]],
    altitudes_in_km=[0.0],
    verbose=True,
)
```

## What `verbose=True` enables

- OTSO calculation status messages used by AniMAIRE wrappers.
- Pandarallel progress bars and worker information.
- Step-by-step asymptotic direction and weighting-factor messages.
- Optional intermediate debug artifact writes used by the engine internals.

With `verbose=False`, these details are suppressed so production runs remain compact.
