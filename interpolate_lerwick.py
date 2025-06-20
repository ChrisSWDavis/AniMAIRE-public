import pandas as pd
from AniMAIRE.MAIREPLUS_event import NeutronMonitorData

# File paths
f1 = 'gle.oulu.fi/GLE69/C069OULU_cor.DAT'
f2 = 'gle.oulu.fi/GLE69/C069MOSC_cor.DAT'
outfile = 'gle.oulu.fi/GLE69/C069LRWK_interpolated.DAT'

# Read data
oulu = NeutronMonitorData.from_file(f1)
mosc = NeutronMonitorData.from_file(f2)

# Merge on timestamp, interval, time_interval, code
merged = pd.merge(
    oulu, mosc,
    on=['timestamp', 'interval_seconds', 'time_interval', 'code'],
    suffixes=('_oulu', '_mosc')
)

# Average relevant columns
merged['uncorrected_count_rate'] = (
    merged['uncorrected_count_rate_oulu'] + merged['uncorrected_count_rate_mosc']) / 2
merged['pressure_mb'] = (
    merged['pressure_mb_oulu'] + merged['pressure_mb_mosc']) / 2
merged['corrected_count_rate'] = (
    merged['corrected_count_rate_oulu'] + merged['corrected_count_rate_mosc']) / 2
merged['percentage_increase'] = (
    merged['percentage_increase_oulu'] + merged['percentage_increase_mosc']) / 2
merged['corrected_percentage_increase'] = (
    merged['corrected_percentage_increase_oulu'].fillna(0) + merged['corrected_percentage_increase_mosc'].fillna(0)) / 2
# If both are NaN, set to NaN
mask_nan = merged['corrected_percentage_increase_oulu'].isna() & merged['corrected_percentage_increase_mosc'].isna()
merged.loc[mask_nan, 'corrected_percentage_increase'] = float('nan')

# Write header
header = [
    'LERWICK    LATITUDE   60.15    LONGITUDE    -1.13    ALTITUDE      15 m',
    'LERWICK    INSTRUMENT    6NM64    NEUTRON MONITOR',
    'LERWICK    STANDARD PRESSURE   1013.0  mb   COEFFICIENT  -0.750  %/mb',
    'LERWICK    PRE-INCREASE BASELINE TIME INTERVAL  050120 050000-060000   UT',
    'LERWICK    PRE-INCREASE AVERAGE COUNTING RATE     122.45  COUNTS PER SECOND',
    'LERWICK    TIME INTERVALS      3600     300',
    'LERWICK    SCALE FACTORS       1.00    1.00',
    'STATION    YYMMDD  SEC   TIME (UT)  CODE   UNCORR. PRESS.     CORR.     INC.\tcorrected',
    '                         INTERVAL    TD      C/S    (mb)       C/S        %'
]

with open(outfile, 'w') as out:
    for l in header:
        out.write(l + '\n')
    for _, row in merged.iterrows():
        out.write(
            f"LERWICK     {row['timestamp'].strftime('%y%m%d')} {int(row['interval_seconds']):4d} {row['time_interval']:>11} {row['code']:>2} "
            f"{row['uncorrected_count_rate']:8.2f} {row['pressure_mb']:7.1f} {row['corrected_count_rate']:8.2f} {row['percentage_increase']:7.1f}\t"
            f"{row['corrected_percentage_increase'] if pd.notnull(row['corrected_percentage_increase']) else '-9999'}\n"
        )
    out.write('***************************************************************************\n')
    out.write('"Comment: Interpolated mean of OULU and MOSCOW; Station: Lerwick, Shetland Islands"\n') 