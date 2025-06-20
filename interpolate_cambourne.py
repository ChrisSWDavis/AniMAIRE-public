import pandas as pd
from AniMAIRE.MAIREPLUS_event import NeutronMonitorData
import numpy as np

# File paths
f1 = 'gle.oulu.fi/GLE69/C069JUNG.DAT'
f2 = 'gle.oulu.fi/GLE69/C069KIEL.DAT'
outfile = 'gle.oulu.fi/GLE69/C069CMBN_interpolated.DAT'

# Read data
jung = NeutronMonitorData.from_file(f1)
kiel = NeutronMonitorData.from_file(f2)

# Altitudes
jung_alt = 3570  # meters, from JUNG file
cambourne_alt = 88  # meters
lambda_att = 1300  # meters

delta_h = cambourne_alt - jung_alt
atten_factor = np.exp(delta_h / lambda_att)

# Apply attenuation correction to JUNG data
jung['uncorrected_count_rate'] = jung['uncorrected_count_rate'] * atten_factor
jung['corrected_count_rate'] = jung['corrected_count_rate'] * atten_factor

# Merge on timestamp, interval, time_interval, code
merged = pd.merge(
    jung, kiel,
    on=['timestamp', 'interval_seconds', 'time_interval', 'code'],
    suffixes=('_jung', '_kiel')
)

# Average relevant columns
merged['uncorrected_count_rate'] = (
    merged['uncorrected_count_rate_jung'] + merged['uncorrected_count_rate_kiel']) / 2
merged['pressure_mb'] = (
    merged['pressure_mb_jung'] + merged['pressure_mb_kiel']) / 2
merged['corrected_count_rate'] = (
    merged['corrected_count_rate_jung'] + merged['corrected_count_rate_kiel']) / 2
merged['percentage_increase'] = (
    merged['percentage_increase_jung'] + merged['percentage_increase_kiel']) / 2
merged['corrected_percentage_increase'] = (
    merged['corrected_percentage_increase_jung'].fillna(0) + merged['corrected_percentage_increase_kiel'].fillna(0)) / 2
# If both are NaN, set to NaN
mask_nan = merged['corrected_percentage_increase_jung'].isna() & merged['corrected_percentage_increase_kiel'].isna()
merged.loc[mask_nan, 'corrected_percentage_increase'] = float('nan')

# Write header
header = [
    'CAMBOURNE  LATITUDE   50.22    LONGITUDE   354.70    ALTITUDE      88 m',
    'CAMBOURNE  INSTRUMENT    6NM64    NEUTRON MONITOR',
    'CAMBOURNE  STANDARD PRESSURE   1013.0  mb   COEFFICIENT  -0.750  %/mb',
    'CAMBOURNE  PRE-INCREASE BASELINE TIME INTERVAL  050120 050000-060000   UT',
    'CAMBOURNE  PRE-INCREASE AVERAGE COUNTING RATE     122.45  COUNTS PER SECOND',
    'CAMBOURNE  TIME INTERVALS      3600     300',
    'CAMBOURNE  SCALE FACTORS       1.00    1.00',
    'STATION    YYMMDD  SEC   TIME (UT)  CODE   UNCORR. PRESS.     CORR.     INC.\tcorrected',
    '                         INTERVAL    TD      C/S    (mb)       C/S        %'
]

with open(outfile, 'w') as out:
    for l in header:
        out.write(l + '\n')
    for _, row in merged.iterrows():
        out.write(
            f"CAMBOURNE   {row['timestamp'].strftime('%y%m%d')} {int(row['interval_seconds']):4d} {row['time_interval']:>11} {row['code']:>2} "
            f"{row['uncorrected_count_rate']:8.2f} {row['pressure_mb']:7.1f} {row['corrected_count_rate']:8.2f} {row['percentage_increase']:7.1f}\t"
            f"{row['corrected_percentage_increase'] if pd.notnull(row['corrected_percentage_increase']) else '-9999'}\n"
        )
    out.write('***************************************************************************\n')
    out.write('"Comment: Interpolated mean of JUNG and KIEL; Station: Cambourne, Cornwall"\n') 