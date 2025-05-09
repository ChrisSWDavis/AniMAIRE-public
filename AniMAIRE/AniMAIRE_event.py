# Import necessary libraries
from AniMAIRE.AniMAIRE import run_from_double_power_law_gaussian_distribution
from AniMAIRE.DoseRateFrame import DoseRateFrame
from AniMAIRE.dose_plotting import create_gle_globe_animation, create_gle_map_animation, plot_dose_map, plot_on_spherical_globe
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import datetime as dt
from tqdm.auto import tqdm  # Progress bar for long-running operations
from joblib import Memory  # For caching computation results
from typing import Union, Sequence, Optional, Dict, Any, Tuple

import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from IPython.display import HTML

import netCDF4  # For exporting data to NetCDF format

# Set up caching to avoid recomputing expensive operations
memory = Memory(location='./.AniMAIRE_event_cache')

# Add a base class for common functionality
class BaseAniMAIREEvent:
    """Template base class for AniMAIRE event types with shared attributes."""
    def __init__(self):
        # Initialize common containers for components and results
        self.dose_rate_components = {}
        self.dose_rates = {}

    def run_AniMAIRE(self, *args: Any, **kwargs: Any) -> Dict[dt.datetime, DoseRateFrame]:
        """Abstract run interface; must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run_AniMAIRE method")

    def summarize_spectra(self) -> Optional[Dict[str, Any]]:
        """
        Provides a summary of the input spectral data including time range and parameter ranges.
        Returns a dict with counts and parameter stats.
        """
        if not hasattr(self, 'spectra') or self.spectra is None:
            print("Spectra data not loaded or formatted yet.")
            return None
        summary = {
            "Number of Timestamps": len(self.spectra),
            "Time Range (UTC)": (self.spectra['datetime'].min(), self.spectra['datetime'].max()),
            "Parameter Ranges": {}
        }
        param_cols = ['J0', 'gamma', 'deltaGamma', 'sigma_1', 'sigma_2', 'B',
                      'alpha_prime', 'reference_pitch_angle_latitude',
                      'reference_pitch_angle_longitude']
        for col in param_cols:
            if col in self.spectra.columns:
                summary["Parameter Ranges"][col] = (self.spectra[col].min(), self.spectra[col].max())
        print("--- Input Spectra Summary ---")
        print(f"Number of Timestamps: {summary['Number of Timestamps']}")
        print(f"Time Range (UTC): {summary['Time Range (UTC)'][0]} to {summary['Time Range (UTC)'][1]}")
        print("Parameter Ranges:")
        for p, (mn, mx) in summary["Parameter Ranges"].items():
            print(f"  {p}: {mn:.2e} to {mx:.2e}")
        print("---------------------------")
        return summary

    def summarize_results(self) -> Optional[Dict[str, Any]]:
        """
        Provides a comprehensive summary of the calculated dose rate results across all dose types.
        Returns a dict with peak values, times, and locations per dose type.
        """
        if not hasattr(self, 'dose_rates') or not self.dose_rates:
            print("AniMAIRE simulation results not available. Run run_AniMAIRE() first.")
            return None
        timestamps = sorted(self.dose_rates.keys())
        first_frame = self.dose_rates[timestamps[0]]
        expected = {'edose':'effective dose in µSv/hr','adose':'ambient dose equivalent in µSv/hr',
                    'dosee':'dose equivalent in µSv/hr','tn1':'>1 MeV neutron flux in n/cm2/s',
                    'tn2':'>10 MeV neutron flux in n/cm2/s','tn3':'>60 MeV neutron flux in n/cm2/s',
                    'SEU':'single event upset rate','SEL':'single event latch-up rate'}
        avail = [c for c in expected if c in first_frame.columns]
        derived = [c for c in first_frame.columns if any(c.startswith(b+' ') for b in ['SEU','SEL'])]
        dose_cols = avail + derived
        if not dose_cols:
            print("No dose rate columns found in the results.")
            return None
        dose_summaries = {}
        for dc in dose_cols:
            peak, t_peak, loc = 0.0, None, None
            for ts, frame in self.dose_rates.items():
                if dc not in frame.columns: continue
                cp = frame[dc].max()
                if cp > peak:
                    peak, t_peak = cp, ts
                    try:
                        row = frame.loc[frame[dc].idxmax()]
                        loc = (row.get('latitude'), row.get('longitude'), row.get('altitude (km)'))
                    except:
                        loc = None
            dose_summaries[dc] = {"Peak Value":peak, "Time of Peak":t_peak, "Location of Peak":loc,
                                  "Description":expected.get(dc,dc)}
        summary = {"Number of Timestamps":len(timestamps),
                   "Time Range (UTC)":(timestamps[0],timestamps[-1]),
                   "Dose Summaries":dose_summaries}
        print("--- Simulation Results Summary ---")
        print(f"Processed {summary['Number of Timestamps']} timestamps {summary['Time Range (UTC)']}")
        for dc, info in summary['Dose Summaries'].items():
            print(f"{dc}: Peak {info['Peak Value']:.3e} at {info['Time of Peak']} Loc {info['Location of Peak']}")
        print("------------------------------")
        return summary

    def get_available_altitudes(self) -> Sequence[float]:
        """
        Get all unique altitudes (km) across stored dose rate frames.
        """
        if not hasattr(self, 'dose_rates') or not self.dose_rates:
            print("No dose rate data available. Run run_AniMAIRE() first.")
            return []
        all_alts = set()
        for frame in self.dose_rates.values():
            all_alts.update(frame.get_altitudes())
        return sorted(all_alts)

    def create_gle_map_animation(self, altitude=12.192, save_gif=False, save_mp4=False, **kwargs):
        """
        Create a 2D map animation at a given altitude over time.
        """
        return create_gle_map_animation(self.dose_rates, altitude, save_gif, save_mp4, **kwargs)

    def create_gle_globe_animation(self, altitude=12.192, save_gif=False, save_mp4=False, **kwargs):
        """
        Create a 3D globe animation at a given altitude over time.
        """
        return create_gle_globe_animation(self.dose_rates, altitude, save_gif, save_mp4, **kwargs)

    def create_spectra_animation(self, output_filename='GLE74_spectra_animation.mp4', fps=2,
                                spectra_xlim=(0.1, 20), spectra_ylim=(1e-16, 1e14), **kwargs):
        """
        Animate rigidity spectra over the event duration.
        """
        timestamps = sorted(self.dose_rates.keys())
        fig, ax = plt.subplots(figsize=(8, 6))

        def update(i):
            ax.clear()
            ts = timestamps[i]
            frame = self.dose_rates[ts]
            frame.plot_spectra(ax=ax, **kwargs)
            ax.set_title(f'Rigidity Spectra at {ts}')
            ax.set_xlim(spectra_xlim)
            ax.set_ylim(spectra_ylim)
            return ax,

        ani = animation.FuncAnimation(fig, update, frames=len(timestamps), blit=False, interval=1000/fps)
        ani.save(output_filename, writer='ffmpeg', fps=fps)
        plt.close(fig)
        return HTML(ani.to_jshtml())

    def create_pad_animation(self, output_filename='GLE74_pad_animation.mp4', fps=2,
                             pad_xlim=(0, 3.14), pad_ylim=(0, 1.2), **kwargs):
        """
        Animate pitch angle distributions over the event duration.
        """
        timestamps = sorted(self.dose_rates.keys())
        fig, ax = plt.subplots(figsize=(8, 6))

        def update(i):
            ax.clear()
            ts = timestamps[i]
            frame = self.dose_rates[ts]
            frame.plot_pitch_angle_distributions(ax=ax, **kwargs)
            ax.set_title(f'Pitch Angle Distributions at {ts}')
            ax.set_xlim(pad_xlim)
            ax.set_ylim(pad_ylim)
            return ax,

        ani = animation.FuncAnimation(fig, update, frames=len(timestamps), blit=False, interval=1000/fps)
        ani.save(output_filename, writer='ffmpeg', fps=fps)
        plt.close(fig)
        return HTML(ani.to_jshtml())

    def create_combined_animation(self, output_filename='GLE74_combined_animation.mp4', fps=2,
                                  spectra_xlim=(0.1, 20), spectra_ylim=(1e-16, 1e14),
                                  pad_xlim=(0, 3.14), pad_ylim=(0, 1.2), **kwargs):
        """
        Animate both rigidity spectra and pitch angle distributions side by side.
        """
        timestamps = sorted(self.dose_rates.keys())
        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        def update(i):
            ax1.clear()
            ax2.clear()
            ts = timestamps[i]
            frame = self.dose_rates[ts]
            frame.plot_spectra(ax=ax1, **kwargs)
            ax1.set_title(f'Rigidity Spectra at {ts}')
            ax1.set_xlim(spectra_xlim)
            ax1.set_ylim(spectra_ylim)
            frame.plot_pitch_angle_distributions(ax=ax2, **kwargs)
            ax2.set_title(f'Pitch Angle Distributions at {ts}')
            ax2.set_xlim(pad_xlim)
            ax2.set_ylim(pad_ylim)
            return ax1, ax2

        ani = animation.FuncAnimation(fig, update, frames=len(timestamps), blit=False, interval=1000/fps)
        ani.save(output_filename, writer='ffmpeg', fps=fps)
        plt.close(fig)
        return HTML(ani.to_jshtml())

    def get_dose_rate_frame(self, timestamp: dt.datetime, nearest: bool = True):
        """Retrieve DoseRateFrame for a timestamp."""
        if timestamp in self.dose_rates: return self.dose_rates[timestamp]
        if not nearest: return None
        times = sorted(self.dose_rates.keys()); arr = [t.timestamp() for t in times]
        idx = int(np.argmin(np.abs(np.array(arr)-timestamp.timestamp())))
        return self.dose_rates[times[idx]]

    def get_dose_rate_at_location(self, latitude, longitude, altitude, timestamp,
                                  dose_type='edose', nearest_ts=True, interpolation_method='linear'):
        """Interpolate dose rate at a geographic location/time."""
        frame = self.get_dose_rate_frame(timestamp, nearest=nearest_ts)
        if frame is None: return None
        tol = 0.1
        data = frame.query(f"`altitude (km)` >= {altitude-tol} & `altitude (km)` <= {altitude+tol}")
        if data.empty: return None
        if data['altitude (km)'].nunique()>1:
            alt0 = data.iloc[(data['altitude (km)']-altitude).abs().argsort()]['altitude (km)'].iloc[0]
            data = data[data['altitude (km)']==alt0]
        if dose_type not in data.columns: return None
        pts = data[['latitude','longitude']].values; vals=data[dose_type].values
        interp=griddata(pts, vals, (latitude,longitude), method=interpolation_method)
        return None if np.isnan(interp) else float(interp)

    def _get_target_grid(self, target_grid=None, n_lat=90, n_lon=180):
        if target_grid is not None: return target_grid
        lats=np.linspace(-90,90,n_lat); lons=np.linspace(-180,180,n_lon)
        lon_g, lat_g = np.meshgrid(lons,lats)
        return np.vstack([lat_g.ravel(), lon_g.ravel()]).T

    def _calculate_time_deltas(self):
        times=sorted(self.dose_rates.keys())
        if len(times)<2: return np.array([1.0])
        diffs = np.diff([t.timestamp() for t in times]); dt=np.zeros(len(times))
        dt[0]=diffs[0]/2; dt[-1]=diffs[-1]/2; dt[1:-1]=(diffs[:-1]+diffs[1:])/2
        return dt/3600.0

    def calculate_integrated_dose(self, altitude, dose_type='edose'):
        """Integrate dose over time on native grid at altitude."""
        times=sorted(self.dose_rates.keys()); first=self.dose_rates[times[0]]
        tol=0.1; df0=first.query(f"`altitude (km)`>={altitude-tol}&`altitude (km)`<={altitude+tol}")
        if df0.empty: return None
        if df0['altitude (km)'].nunique()>1:
            alt0=df0.iloc[(df0['altitude (km)']-altitude).abs().argsort()]['altitude (km)'].iloc[0]
            df0=df0[df0['altitude (km)']==alt0]
        idx=pd.MultiIndex.from_frame(df0[['latitude','longitude']]); acc=pd.Series(0.0,index=idx)
        dts=self._calculate_time_deltas()
        for i,ts in enumerate(times):
            df=self.dose_rates[ts].query(f"`altitude (km)`>={altitude-tol}&`altitude (km)`<={altitude+tol}")
            if df.empty: continue
            if df['altitude (km)'].nunique()>1:
                altn=df.iloc[(df['altitude (km)']-altitude).abs().argsort()]['altitude (km)'].iloc[0]
                df=df[df['altitude (km)']==altn]
            if dose_type not in df.columns: continue
            s=df.set_index(['latitude','longitude'])[dose_type]
            acc=acc.add(s*dts[i],fill_value=0)
        out=acc.reset_index().rename(columns={0:f'integrated_{dose_type}_uSv'})
        return out

    def get_peak_dose_rate_map(self, altitude, dose_type='edose'):
        """Compute peak dose rate per grid point at altitude."""
        times=sorted(self.dose_rates.keys()); first=self.dose_rates[times[0]]
        tol=0.1; df0=first.query(f"`altitude (km)`>={altitude-tol}&`altitude (km)`<={altitude+tol}")
        if df0.empty: return None
        if df0['altitude (km)'].nunique()>1:
            alt0=df0.iloc[(df0['altitude (km)']-altitude).abs().argsort()]['altitude (km)'].iloc[0]
            df0=df0[df0['altitude (km)']==alt0]
        idx=pd.MultiIndex.from_frame(df0[['latitude','longitude']]); peak=pd.Series(-np.inf,index=idx)
        for ts in times:
            df=self.dose_rates[ts].query(f"`altitude (km)`>={altitude-tol}&`altitude (km)`<={altitude+tol}")
            if df.empty: continue
            if df['altitude (km)'].nunique()>1:
                altn=df.iloc[(df['altitude (km)']-altitude).abs().argsort()]['altitude (km)'].iloc[0]
                df=df[df['altitude (km)']==altn]
            if dose_type not in df.columns: continue
            cur=df.set_index(['latitude','longitude'])[dose_type]
            peak=pd.Series(np.maximum(peak.values,cur.reindex(peak.index,fill_value=-np.inf).values),index=peak.index)
        peak.replace(-np.inf,np.nan,inplace=True)
        return peak.reset_index().rename(columns={0:f'peak_{dose_type}_uSv_hr'})

    def plot_integrated_dose_map(self, altitude, dose_type='edose', **plot_kwargs):
        """Plot integrated dose map using native grid."""
        df=self.calculate_integrated_dose(altitude,dose_type)
        if df is None: return None
        df['altitude (km)']=altitude
        args={'plot_title':f'Integrated {dose_type} at {altitude} km','dose_type':f'integrated_{dose_type}_uSv','legend_label':f'Integrated {dose_type}'}
        args.update(plot_kwargs)
        return plot_dose_map(df,**args)[0]

    def plot_peak_dose_rate_map(self, altitude, dose_type='edose', **plot_kwargs):
        """Plot peak dose rate map using native grid."""
        df=self.get_peak_dose_rate_map(altitude,dose_type)
        if df is None: return None
        df['altitude (km)']=altitude
        args={'plot_title':f'Peak {dose_type} at {altitude} km','dose_type':f'peak_{dose_type}_uSv_hr','legend_label':f'Peak {dose_type}'}
        args.update(plot_kwargs)
        return plot_dose_map(df,**args)[0]

    def plot_integrated_dose_globe(self, altitude, dose_type='edose', **plot_kwargs):
        """Plot integrated dose on 3D globe."""
        df=self.calculate_integrated_dose(altitude,dose_type)
        if df is None: return None
        args={'plot_title':f'Integrated {dose_type} at {altitude} km','dose_type':f'integrated_{dose_type}_uSv','legend_label':f'Integrated {dose_type}'}
        args.update(plot_kwargs)
        return plot_on_spherical_globe(df,**args)

    def plot_peak_dose_rate_globe(self, altitude, dose_type='edose', **plot_kwargs):
        """Plot peak dose rate on 3D globe."""
        df=self.get_peak_dose_rate_map(altitude,dose_type)
        if df is None: return None
        args={'plot_title':f'Peak {dose_type} at {altitude} km','dose_type':f'peak_{dose_type}_uSv_hr','legend_label':f'Peak {dose_type}'}
        args.update(plot_kwargs)
        return plot_on_spherical_globe(df,**args)

    def plot_timeseries_at_location(self, latitude, longitude, altitude, dose_type='edose', ax=None,
                                   nearest_ts=True, interpolation_method='linear', **plot_kwargs):
        """Plot time series of dose at a specific location.
        """
        times=sorted(self.dose_rates.keys()); vals=[]; tlist=[]
        for ts in times:
            d=self.get_dose_rate_at_location(latitude,longitude,altitude,ts,dose_type,nearest_ts,interpolation_method)
            if d is not None: vals.append(d); tlist.append(ts)
        if not vals: return None
        if ax is None: fig,ax=plt.subplots(figsize=(10,6))
        ax.plot(tlist,vals,label=dose_type,marker='o')
        ax.set_xlabel('Time (UTC)'); ax.set_ylabel(f'{dose_type} (uSv/hr)'); ax.legend(); ax.grid(True); fig.autofmt_xdate()
        return ax

    def plot_map_at_time(self, timestamp, altitude, ax=None, nearest_ts=True, **plot_kwargs):
        """Plot a 2D dose map at a given timestamp and altitude."""
        frame=self.get_dose_rate_frame(timestamp,nearest_ts)
        if frame is None: return None
        if ax: plot_kwargs['ax']=ax
        return frame.plot_dose_map(altitude=altitude,**plot_kwargs)

    def plot_globe_at_time(self, timestamp, altitude, nearest_ts=True, **plot_kwargs):
        """Plot a 3D globe dose at a given timestamp and altitude."""
        frame=self.get_dose_rate_frame(timestamp,nearest_ts)
        if frame is None: return None
        return frame.plot_on_globe(altitude=altitude,**plot_kwargs)

    def export_to_netcdf(self, filename):
        """Export all dose rates to a NetCDF file."""
        if not self.dose_rates: print('No data to export'); return
        # (existing netcdf logic can be called here via a helper or imported)
        # for brevity, call AniMAIRE_event.export_to_netcdf if needed
        AniMAIRE_event.export_to_netcdf(self, filename)

    def get_all_timestamps(self):
        """
        Return a sorted list of all timestamps represented in the event.
        Returns:
            List of datetime objects (sorted).
        """
        if not hasattr(self, 'dose_rates') or not self.dose_rates:
            return []
        return sorted(self.dose_rates.keys())

class DoublePowerLawGaussianEvent(BaseAniMAIREEvent):
    """
    Event based on a double power-law rigidity spectrum and Gaussian pitch-angle distribution.

    Required input CSV format (columns):
      - Time: str or datetime (UTC) per spectrum row
      - J_0: float, normalization constant
      - gamma: float, spectral index
      - d_gamma: float, modification to spectral index
      - Sigma1: float, first Gaussian sigma for pitch-angle distribution
      - Sigma2: float, second Gaussian sigma for pitch-angle distribution
      - B: float, scaling factor for second Gaussian component
      - SymLat: float, reference pitch-angle latitude (GEO coords)
      - SymLong: float, reference pitch-angle longitude (GEO coords)
    Optional:
      - alpha_prime: float, default = math.pi
    """
    
    def __init__(self, spectra_file_path: str) -> None:
        """
        Initialize an AniMAIRE_event with a spectral data file.
        
        Parameters:
        -----------
        spectra_file_path : str
            Path to the CSV file containing spectral data for the event
        """
        super().__init__()
        self.spectra_file_path = spectra_file_path
        self.raw_spectra_data = pd.read_csv(spectra_file_path)
        self.spectra = self.correctly_formatted_spectra()

    def correctly_formatted_spectra(self):
        """
        Format the input spectral data to match AniMAIRE's expected format.
        Handles column renaming, datetime conversion, and adds default values for missing columns.
        
        Returns:
        --------
        pandas.DataFrame
            Correctly formatted spectra data
        """
        # Map the columns from the input file to the expected AniMAIRE format
        # Based on the GLE74 Spectra_reformatted.csv file structure
        column_mapping = {
            'Time': 'datetime',
            'J_0': 'J0',
            'gamma': 'gamma',
            'd_gamma': 'deltaGamma',
            'Sigma1': 'sigma_1',
            'Sigma2': 'sigma_2',
            'B': 'B',
            'SymLat': 'reference_pitch_angle_latitude',
            'SymLong': 'reference_pitch_angle_longitude'
        }
        
        # Rename the columns according to the mapping
        self.spectra = self.raw_spectra_data.rename(columns=column_mapping)
        
        # Add alpha_prime column with value of pi if it doesn't exist
        import math
        if 'alpha_prime' not in self.spectra.columns:
            self.spectra['alpha_prime'] = math.pi

        # Convert the datetime column to UTC datetime
        # Check if the datetime column is already in datetime format
        if pd.api.types.is_datetime64_any_dtype(self.spectra['datetime']):
            # If it's already a datetime, ensure it's in UTC
            self.spectra['datetime'] = self.spectra['datetime'].dt.tz_localize(None).dt.tz_localize('UTC')
        else:
            # If it's a string, parse it to datetime
            try:
                # Try parsing with full datetime format (if it contains date and time)
                self.spectra['datetime'] = pd.to_datetime(self.spectra['datetime'], utc=True)
            except:
                # If the column only contains time (like "02:00"), add a date part
                # Using 2024-05-11 as the date based on the reformatted CSV
                self.spectra['datetime'] = pd.to_datetime('2024-05-11 ' + self.spectra['datetime'], utc=True)
        
        return self.spectra
    
    def run_AniMAIRE(self, n_timestamps: Optional[int] = None, use_cache: bool = True, **kwargs: Any) -> Dict[dt.datetime, DoseRateFrame]:
        """
        Run AniMAIRE simulation for each timestamp in the spectral data.
        
        Parameters:
        -----------
        n_timestamps : int, optional
            Limit the number of timestamps to process (useful for testing)
        use_cache : bool, default=True
            Whether to use cached results for identical parameter sets
        **kwargs : dict
            Additional keyword arguments to pass to run_from_double_power_law_gaussian_distribution
            
        Returns:
        --------
        dict
            Dictionary of DoseRateFrame objects keyed by timestamp
        """
        # Initialize the dose_rates dictionary
        self.dose_rates = {}  # Use dictionary instead of list
        
        # Process each spectrum in the input data
        for index, spectrum in self.spectra.iterrows():

            # Display progress information
            total_spectra = len(self.spectra)
            percentage_complete = (index / total_spectra) * 100
            print(f"Running AniMAIRE for spectrum {index} ({percentage_complete:.1f}% complete)")
            # Print the datetime for the current spectrum
            print(f"Processing spectrum for datetime: {spectrum['datetime']}")
            
            # Determine whether to use caching or not
            if use_cache:
                # Use cached function to avoid recomputing identical parameter sets
                output_dose_rate = run_animaire_cached(
                    J0=spectrum['J0'],
                    gamma=spectrum['gamma'],
                    deltaGamma=spectrum['deltaGamma'],
                    sigma_1=spectrum['sigma_1'],
                    sigma_2=spectrum['sigma_2'],
                    B=spectrum['B'],
                    alpha_prime=spectrum['alpha_prime'],
                    reference_pitch_angle_latitude=spectrum['reference_pitch_angle_latitude'],
                    reference_pitch_angle_longitude=spectrum['reference_pitch_angle_longitude'],
                    date_and_time=spectrum['datetime'],
                    use_split_spectrum=True,
                    **kwargs
                )
            else:
                # Use the function directly without caching
                output_dose_rate = run_from_double_power_law_gaussian_distribution(
                    J0=spectrum['J0'],
                    gamma=spectrum['gamma'],
                    deltaGamma=spectrum['deltaGamma'],
                    sigma_1=spectrum['sigma_1'],
                    sigma_2=spectrum['sigma_2'],
                    B=spectrum['B'],
                    alpha_prime=spectrum['alpha_prime'],
                    reference_pitch_angle_latitude=spectrum['reference_pitch_angle_latitude'],
                    reference_pitch_angle_longitude=spectrum['reference_pitch_angle_longitude'],
                    date_and_time=spectrum['datetime'],
                    use_split_spectrum=True,
                    **kwargs
                )
            
            # Store dose rate with datetime as key
            self.dose_rates[spectrum['datetime']] = output_dose_rate

            # Check if we need to limit the number of timestamps
            if n_timestamps is not None:
                # Break the loop if we've processed the specified number of timestamps
                if index + 1 >= n_timestamps:
                    print(f"Reached the specified limit of {n_timestamps} timestamps. Stopping.")
                    break

        return self.dose_rates
    
# Legacy alias for backward compatibility
AniMAIRE_event = DoublePowerLawGaussianEvent

# Define a cached version of the run_from_double_power_law_gaussian_distribution function
@memory.cache
def run_animaire_cached(
    J0: float,
    gamma: float,
    deltaGamma: float,
    sigma_1: float,
    sigma_2: float,
    B: float,
    alpha_prime: float,
    reference_pitch_angle_latitude: float,
    reference_pitch_angle_longitude: float,
    date_and_time: Union[dt.datetime, np.datetime64],
    use_split_spectrum: bool,
    **kwargs: Any
) -> DoseRateFrame:
    """Cached version of run_from_double_power_law_gaussian_distribution using Streamlit"""
    return run_from_double_power_law_gaussian_distribution(
        J0=J0,
        gamma=gamma,
        deltaGamma=deltaGamma,
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        B=B,
        alpha_prime=alpha_prime,
        reference_pitch_angle_latitude=reference_pitch_angle_latitude,
        reference_pitch_angle_longitude=reference_pitch_angle_longitude,
        date_and_time=date_and_time,
        use_split_spectrum=use_split_spectrum,
        **kwargs
    )

def run_from_GLE_spectrum_file(
        GLE_spectrum_file: str,
        **kwargs: Any
) -> DoublePowerLawGaussianEvent:
    """
    Perform a run to calculate dose rates using a GLE spectrum file.
    """
    event = DoublePowerLawGaussianEvent(GLE_spectrum_file)
    event.run_AniMAIRE(**kwargs)
    return event



