"""
AniMAIRE_event module provides classes and functions for simulating solar energetic particle events.

This module contains the main event processing classes for AniMAIRE, which calculate radiation dose rates 
from parameterized solar energetic particle spectra. It supports both isotropic and anisotropic 
pitch-angle distributions of cosmic rays.
"""

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
from typing import Union, Sequence, Optional, Dict, Any, Tuple, List

import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from IPython.display import HTML

import netCDF4  # For exporting data to NetCDF format

# Set up caching to avoid recomputing expensive operations
memory = Memory(location='./.AniMAIRE_event_cache')

class BaseAniMAIREEvent:
    """
    Base class for AniMAIRE event simulations with shared functionality.
    
    This class provides common methods for analyzing and visualizing radiation dose rates
    from solar energetic particle events. It serves as a template that specific event types
    can inherit from and extend.
    
    Attributes:
        dose_rate_components (Dict): Storage for component-wise dose rate data
        dose_rates (Dict[dt.datetime, DoseRateFrame]): Storage for calculated dose rates at each timestamp
    """
    
    DEFAULT_ALTITUDE = 12.192  # 40,000 ft in km
    
    def __init__(self) -> None:
        """
        Initialize a new BaseAniMAIREEvent instance with empty containers.
        """
        # Initialize common containers for components and results
        self.dose_rate_components: Dict = {}
        self.dose_rates: Dict[dt.datetime, DoseRateFrame] = {}

    def __repr__(self) -> str:
        """
        Return a string representation of the event object.
        
        Returns:
            str: Concise summary of the event with key information
        """
        class_name = self.__class__.__name__
        n_timestamps = len(self.dose_rates) if hasattr(self, 'dose_rates') and self.dose_rates else 0
        
        timestamp_range = "N/A"
        if n_timestamps > 0:
            timestamps = sorted(self.dose_rates.keys())
            timestamp_range = f"{timestamps[0]} to {timestamps[-1]}"
            
        altitudes = "N/A"
        if n_timestamps > 0:
            all_alts = set()
            for frame in self.dose_rates.values():
                all_alts.update(frame.get_altitudes())
            alt_str = ", ".join(f"{alt:.2f}" for alt in sorted(all_alts))
            altitudes = f"[{alt_str}] km"
        
        # Get class-specific attributes
        info_lines = []
        if hasattr(self, 'data_directory_path'):
            info_lines.append(f"Data: {self.data_directory_path}")
        if hasattr(self, 'reference_station'):
            info_lines.append(f"Reference: {self.reference_station}")
        if hasattr(self, 'spectra_file_path'):
            info_lines.append(f"Spectra: {self.spectra_file_path}")
            
        # Add neutron monitor info
        monitor_info = self._get_monitor_info()
        if monitor_info:
            if isinstance(monitor_info, list):
                if all(isinstance(x, dict) for x in monitor_info):
                    for d in monitor_info:
                        info_lines.append("Monitors: " + ", ".join(f"{k}: {v}" for k, v in d.items()))
                else:
                    for s in monitor_info:
                        info_lines.append(f"Monitor: {s}")
        
        # Format the output
        class_info = f"{class_name} with {n_timestamps} timestamps"
        if info_lines:
            extra_info = ", ".join(info_lines)
            return f"{class_info} ({extra_info})\nTime range: {timestamp_range}\nAltitudes: {altitudes}"
        else:
            return f"{class_info}\nTime range: {timestamp_range}\nAltitudes: {altitudes}"
    
    def _repr_html_(self) -> str:
        """
        Return an HTML representation of the event object for Jupyter notebook display.
        
        Returns:
            str: HTML representation of the event
        """
        class_name = self.__class__.__name__
        n_timestamps = len(self.dose_rates) if hasattr(self, 'dose_rates') and self.dose_rates else 0
        
        timestamp_range = "N/A"
        dose_types = []
        if n_timestamps > 0:
            timestamps = sorted(self.dose_rates.keys())
            timestamp_range = f"{timestamps[0]} to {timestamps[-1]}"
            first_frame = next(iter(self.dose_rates.values()))
            if first_frame is not None:
                expected_types = ['edose', 'adose', 'dosee', 'tn1', 'tn2', 'tn3', 'SEU', 'SEL']
                dose_types = [col for col in expected_types if col in first_frame.columns]
                dose_types += [col for col in first_frame.columns 
                              if any(col.startswith(b+' ') for b in ['SEU', 'SEL'])]
        
        altitudes = []
        if n_timestamps > 0:
            all_alts = set()
            for frame in self.dose_rates.values():
                all_alts.update(frame.get_altitudes())
            altitudes = sorted(all_alts)
            
        # Get class-specific attributes for display
        info_rows = []
        if hasattr(self, 'data_directory_path'):
            info_rows.append(f"<tr><td>Data directory</td><td>{self.data_directory_path}</td></tr>")
        if hasattr(self, 'reference_station'):
            info_rows.append(f"<tr><td>Reference station</td><td>{self.reference_station}</td></tr>")
        if hasattr(self, 'spectra_file_path'):
            info_rows.append(f"<tr><td>Spectra file</td><td>{self.spectra_file_path}</td></tr>")
        
        # Add neutron monitor info
        monitor_info = self._get_monitor_info()
        if monitor_info:
            if isinstance(monitor_info, list):
                if all(isinstance(x, dict) for x in monitor_info):
                    for i, d in enumerate(monitor_info, 1):
                        info_rows.append("<tr><td>Monitor set {}</td><td>{}</td></tr>".format(i, ", ".join(f"{k}: {v}" for k, v in d.items())))
                else:
                    for s in monitor_info:
                        info_rows.append(f"<tr><td>Monitor</td><td>{s}</td></tr>")
        
        # Build the HTML table
        html = f"""
        <div style="background-color:#f8f9fa;padding:12px;border-radius:4px;border:1px solid #ddd;">
            <h3 style="margin-top:0">{class_name}</h3>
            <table>
                <tr><td><b>Timestamps</b></td><td>{n_timestamps}</td></tr>
                <tr><td><b>Time range</b></td><td>{timestamp_range}</td></tr>
                {"".join(info_rows)}
            </table>
            
            <h4 style="margin-top:10px">Available data</h4>
        """
        
        if altitudes:
            html += "<p><b>Altitudes:</b> " + ", ".join([f"{alt:.2f} km" for alt in altitudes]) + "</p>"
            
        if dose_types:
            html += "<p><b>Dose types:</b> " + ", ".join(dose_types) + "</p>"
            
        # Add a brief help text
        html += """
            <p style="margin-top:10px;font-size:0.9em;color:#666;">
                <i>Available methods: summarize_results(), create_gle_map_animation(), plot_integrated_dose_map(), etc.</i>
            </p>
        </div>
        """
        return html

    def run_AniMAIRE(self, *args: Any, **kwargs: Any) -> Dict[dt.datetime, DoseRateFrame]:
        """
        Abstract method to run AniMAIRE simulations.
        
        This method must be implemented by subclasses to perform the actual simulation
        for the specific event type.
        
        Args:
            *args: Variable length argument list for subclass implementations
            **kwargs: Arbitrary keyword arguments for subclass implementations
            
        Returns:
            Dict[dt.datetime, DoseRateFrame]: Dictionary of dose rate frames indexed by timestamp
            
        Raises:
            NotImplementedError: Always raised since this is an abstract method
        """
        raise NotImplementedError("Subclasses must implement run_AniMAIRE method")

    def summarize_spectra(self) -> Optional[Dict[str, Any]]:
        """
        Provide a summary of the input spectral data including time range and parameter ranges.
        
        Analyzes the loaded spectral data and returns statistics about the parameters used
        in the simulation, including minimum and maximum values for each parameter.
        
        Returns:
            Optional[Dict[str, Any]]: Dictionary with counts and parameter statistics,
                                     or None if spectra data is not available
                                     
        Example return structure:
            {
                "Number of Timestamps": int,
                "Time Range (UTC)": (min_datetime, max_datetime),
                "Parameter Ranges": {
                    "J0": (min_val, max_val),
                    "gamma": (min_val, max_val),
                    # ... other parameters
                }
            }
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

    def _get_best_altitude(self, requested_altitude: Optional[float] = None) -> Optional[float]:
        """
        Get the best available altitude for plotting.
        
        If a specific altitude is requested, returns that altitude if available or the nearest available altitude.
        If no altitude is requested, returns the default altitude (12.192 km) if available or the nearest available altitude.
        
        Args:
            requested_altitude (Optional[float], optional): Specific altitude to check for. Defaults to None.
            
        Returns:
            Optional[float]: Best available altitude, or None if no altitudes are available
        """
        available_alts = self.get_available_altitudes()
        if not available_alts:
            return None
            
        target = requested_altitude if requested_altitude is not None else self.DEFAULT_ALTITUDE
        
        # If the exact altitude is available, use it
        if target in available_alts:
            return target
            
        # Find the nearest available altitude
        nearest_alt = min(available_alts, key=lambda x: abs(x - target))
        return nearest_alt

    def create_gle_map_animation(self, altitude: Optional[float] = None, save_gif: bool = False, 
                              save_mp4: bool = False, **kwargs: Any) -> HTML:
        """
        Create a 2D map animation of dose rates at a given altitude over time.
        
        Args:
            altitude (Optional[float], optional): Altitude in kilometers. If None, uses 12.192 km (40,000 ft) if available.
            save_gif (bool, optional): Whether to save as GIF. Defaults to False.
            save_mp4 (bool, optional): Whether to save as MP4. Defaults to False.
            **kwargs: Additional keyword arguments passed to the animation function
            
        Returns:
            HTML: HTML object for displaying the animation in notebooks
        """
        best_altitude = self._get_best_altitude(altitude)
        if best_altitude is None:
            print("No altitude data available.")
            return None
            
        if altitude is not None and abs(best_altitude - altitude) > 0.1:
            print(f"Requested altitude {altitude} km not available. Using nearest available altitude: {best_altitude} km")
            
        return create_gle_map_animation(self.dose_rates, best_altitude, save_gif, save_mp4, **kwargs)

    def create_gle_globe_animation(self, altitude: Optional[float] = None, save_gif: bool = False, 
                                  save_mp4: bool = False, **kwargs: Any) -> HTML:
        """
        Create a 3D globe animation of dose rates at a given altitude over time.
        
        Args:
            altitude (Optional[float], optional): Altitude in kilometers. If None, uses 12.192 km (40,000 ft) if available.
            save_gif (bool, optional): Whether to save as GIF. Defaults to False.
            save_mp4 (bool, optional): Whether to save as MP4. Defaults to False.
            **kwargs: Additional keyword arguments passed to the animation function
            
        Returns:
            HTML: HTML object for displaying the animation in notebooks
        """
        best_altitude = self._get_best_altitude(altitude)
        if best_altitude is None:
            print("No altitude data available.")
            return None
            
        if altitude is not None and abs(best_altitude - altitude) > 0.1:
            print(f"Requested altitude {altitude} km not available. Using nearest available altitude: {best_altitude} km")
            
        return create_gle_globe_animation(self.dose_rates, best_altitude, save_gif, save_mp4, **kwargs)

    def create_spectra_animation(self, output_filename: str = 'GLE74_spectra_animation.mp4', 
                                fps: int = 2, spectra_xlim: Tuple[float, float] = (0.1, 20), 
                                spectra_ylim: Tuple[float, float] = (1e-16, 1e14), 
                                **kwargs: Any) -> HTML:
        """
        Animate rigidity spectra over the event duration.
        
        Creates an animation showing how the rigidity spectrum changes over time during the event.
        
        Args:
            output_filename (str, optional): Output file path. Defaults to 'GLE74_spectra_animation.mp4'.
            fps (int, optional): Frames per second for animation. Defaults to 2.
            spectra_xlim (Tuple[float, float], optional): X-axis limits (GV). Defaults to (0.1, 20).
            spectra_ylim (Tuple[float, float], optional): Y-axis limits (flux). Defaults to (1e-16, 1e14).
            **kwargs: Additional keyword arguments passed to the plot_spectra method
            
        Returns:
            HTML: HTML object for displaying the animation in notebooks
        """
        timestamps = sorted(self.dose_rates.keys())
        fig, ax = plt.subplots(figsize=(8, 6))

        def update(i: int) -> Tuple:
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

    def create_pad_animation(self, output_filename: str = 'GLE74_pad_animation.mp4', 
                            fps: int = 2, pad_xlim: Tuple[float, float] = (0, 3.14), 
                            pad_ylim: Tuple[float, float] = (0, 1.2), **kwargs: Any) -> HTML:
        """
        Animate pitch angle distributions over the event duration.
        
        Creates an animation showing how the pitch angle distribution changes over time during the event.
        
        Args:
            output_filename (str, optional): Output file path. Defaults to 'GLE74_pad_animation.mp4'.
            fps (int, optional): Frames per second for animation. Defaults to 2.
            pad_xlim (Tuple[float, float], optional): X-axis limits (radians). Defaults to (0, 3.14).
            pad_ylim (Tuple[float, float], optional): Y-axis limits (relative intensity). Defaults to (0, 1.2).
            **kwargs: Additional keyword arguments passed to the plot_pitch_angle_distributions method
            
        Returns:
            HTML: HTML object for displaying the animation in notebooks
        """
        timestamps = sorted(self.dose_rates.keys())
        fig, ax = plt.subplots(figsize=(8, 6))

        def update(i: int) -> Tuple:
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

    def create_combined_animation(self, output_filename: str = 'GLE74_combined_animation.mp4', 
                                  fps: int = 2, spectra_xlim: Tuple[float, float] = (0.1, 20), 
                                  spectra_ylim: Tuple[float, float] = (1e-16, 1e14),
                                  pad_xlim: Tuple[float, float] = (0, 3.14), 
                                  pad_ylim: Tuple[float, float] = (0, 1.2), 
                                  **kwargs: Any) -> HTML:
        """
        Animate both rigidity spectra and pitch angle distributions side by side.
        
        Creates a two-panel animation showing how both the rigidity spectrum and
        pitch angle distribution change over time during the event.
        
        Args:
            output_filename (str, optional): Output file path. Defaults to 'GLE74_combined_animation.mp4'.
            fps (int, optional): Frames per second for animation. Defaults to 2.
            spectra_xlim (Tuple[float, float], optional): X-axis limits for spectra (GV). Defaults to (0.1, 20).
            spectra_ylim (Tuple[float, float], optional): Y-axis limits for spectra (flux). Defaults to (1e-16, 1e14).
            pad_xlim (Tuple[float, float], optional): X-axis limits for PAD (radians). Defaults to (0, 3.14).
            pad_ylim (Tuple[float, float], optional): Y-axis limits for PAD (relative intensity). Defaults to (0, 1.2).
            **kwargs: Additional keyword arguments passed to plotting methods
            
        Returns:
            HTML: HTML object for displaying the animation in notebooks
        """
        timestamps = sorted(self.dose_rates.keys())
        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        def update(i: int) -> Tuple[plt.Axes, plt.Axes]:
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

    def get_dose_rate_frame(self, timestamp: dt.datetime, nearest: bool = True) -> Optional[DoseRateFrame]:
        """
        Retrieve DoseRateFrame for a specific timestamp.
        
        Args:
            timestamp (dt.datetime): The timestamp to retrieve the frame for
            nearest (bool, optional): If True and exact timestamp not found, return nearest. Defaults to True.
            
        Returns:
            Optional[DoseRateFrame]: The dose rate frame at the requested time, or None if not found
                                     and nearest=False
        """
        if timestamp in self.dose_rates: return self.dose_rates[timestamp]
        if not nearest: return None
        times = sorted(self.dose_rates.keys()); arr = [t.timestamp() for t in times]
        idx = int(np.argmin(np.abs(np.array(arr)-timestamp.timestamp())))
        return self.dose_rates[times[idx]]

    def get_dose_rate_at_location(self, latitude: float, longitude: float, altitude: float, 
                                  timestamp: dt.datetime, dose_type: str = 'edose', 
                                  nearest_ts: bool = True, 
                                  interpolation_method: str = 'linear') -> Optional[float]:
        """
        Interpolate dose rate at a specific geographic location and time.
        
        Args:
            latitude (float): Geographic latitude in degrees
            longitude (float): Geographic longitude in degrees
            altitude (float): Altitude in kilometers
            timestamp (dt.datetime): Time to retrieve the dose rate for
            dose_type (str, optional): Dose rate type to retrieve. Defaults to 'edose'.
            nearest_ts (bool, optional): If True, use nearest timestamp if exact not found. Defaults to True.
            interpolation_method (str, optional): Method for interpolation. Defaults to 'linear'.
                                                 Can be 'linear', 'nearest', or 'cubic'.
            
        Returns:
            Optional[float]: Interpolated dose rate at the requested location and time, or None if
                            data is not available or interpolation fails
        """
        frame = self.get_dose_rate_frame(timestamp, nearest=nearest_ts)
        if frame is None: return None
        tol = 0.1
        data = frame.query(f"`altitude (km)` >= {altitude-tol} & `altitude (km)` <= {altitude+tol}")
        if data.empty: return None
        if data['altitude (km)'].nunique()>1:
            alt0 = data.iloc[(data['altitude (km)']-altitude).abs().argsort()]['altitude (km)'].iloc[0]
            data = data[data['altitude (km)']==alt0]
        if dose_type not in data.columns: return None
        
        # Handle case with only one latitude/longitude point
        if len(data) == 1:
            # If there's only one point, return its value directly if it matches the requested coordinates
            # or use nearest neighbor approach
            point = data.iloc[0]
            if abs(point['latitude'] - latitude) < 1e-6 and abs(point['longitude'] - longitude) < 1e-6:
                return float(point[dose_type])
            else:
                # Calculate distance to the single point
                dist = ((point['latitude'] - latitude)**2 + (point['longitude'] - longitude)**2)**0.5
                # Return the value if it's close enough, otherwise None
                return float(point[dose_type]) if dist < 10.0 else None
        
        # Normal interpolation for multiple points
        pts = data[['latitude','longitude']].values; vals=data[dose_type].values
        interp=griddata(pts, vals, (latitude,longitude), method=interpolation_method)
        return None if np.isnan(interp) else float(interp)

    def _get_target_grid(self, target_grid: Optional[np.ndarray] = None, 
                         n_lat: int = 90, n_lon: int = 180) -> np.ndarray:
        """
        Get or create a target grid for interpolation.
        
        Args:
            target_grid (Optional[np.ndarray], optional): Existing grid to use. Defaults to None.
            n_lat (int, optional): Number of latitude points if creating grid. Defaults to 90.
            n_lon (int, optional): Number of longitude points if creating grid. Defaults to 180.
            
        Returns:
            np.ndarray: 2D array with shape (n_lat*n_lon, 2) containing lat/lon pairs
        """
        if target_grid is not None: return target_grid
        lats=np.linspace(-90,90,n_lat); lons=np.linspace(-180,180,n_lon)
        lon_g, lat_g = np.meshgrid(lons,lats)
        return np.vstack([lat_g.ravel(), lon_g.ravel()]).T

    def _calculate_time_deltas(self) -> np.ndarray:
        """
        Calculate time intervals between timestamps, in hours.
        
        For each timestamp, calculates the "responsibility period" - half the interval to the
        previous timestamp plus half the interval to the next timestamp.
        
        Returns:
            np.ndarray: Array of time intervals in hours for each timestamp
        """
        times=sorted(self.dose_rates.keys())
        if len(times)<2: return np.array([1.0])
        diffs = np.diff([t.timestamp() for t in times]); dt=np.zeros(len(times))
        dt[0]=diffs[0]/2; dt[-1]=diffs[-1]/2; dt[1:-1]=(diffs[:-1]+diffs[1:])/2
        return dt/3600.0

    def calculate_integrated_dose(self, altitude: float, dose_type: str = 'edose') -> Optional[pd.DataFrame]:
        """
        Integrate dose over time on native grid at specified altitude.
        
        Calculates the time-integrated dose for each grid point at the specified altitude
        by summing the dose rate at each timestamp multiplied by the corresponding time interval.
        
        Args:
            altitude (float): Altitude in kilometers
            dose_type (str, optional): Dose rate type to integrate. Defaults to 'edose'.
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with columns (latitude, longitude, integrated_dose_type_uSv),
                                   or None if no data available at specified altitude
        """
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

    def plot_integrated_dose_map(self, altitude: Optional[float] = None, dose_type: str = 'edose', 
                              show_monitors: bool = True, **plot_kwargs: Any) -> Optional[plt.Axes]:
        """
        Plot integrated dose map using native grid.
        
        Args:
            altitude (Optional[float], optional): Altitude in kilometers. If None, uses 12.192 km (40,000 ft) if available.
            dose_type (str, optional): Dose rate type to integrate. Defaults to 'edose'.
            show_monitors (bool, optional): Whether to display neutron monitor locations. Defaults to True.
                Note: Monitor plotting is only supported in AnisotropicMAIREPLUSevent.
            **plot_kwargs: Additional keyword arguments passed to plot_dose_map function
            
        Returns:
            Optional[plt.Axes]: Matplotlib axes object with the plot, or None if no data available
        """
        best_altitude = self._get_best_altitude(altitude)
        if best_altitude is None:
            print("No altitude data available.")
            return None
            
        if altitude is not None and abs(best_altitude - altitude) > 0.1:
            print(f"Requested altitude {altitude} km not available. Using nearest available altitude: {best_altitude} km")
            
        df = self.calculate_integrated_dose(best_altitude, dose_type)
        if df is None: return None
        df['altitude (km)'] = best_altitude
        args = {'plot_title': f'Integrated {dose_type} at {best_altitude} km', 
                'dose_type': f'integrated_{dose_type}_uSv', 
                'legend_label': f'Integrated {dose_type}'}
        args.update(plot_kwargs)
        
        ax, _ = plot_dose_map(df, **args)
        return ax

    def plot_peak_dose_rate_map(self, altitude: Optional[float] = None, dose_type: str = 'edose', 
                               show_monitors: bool = True, **plot_kwargs: Any) -> Optional[plt.Axes]:
        """
        Plot peak dose rate map using native grid.
        
        Args:
            altitude (Optional[float], optional): Altitude in kilometers. If None, uses 12.192 km (40,000 ft) if available.
            dose_type (str, optional): Dose rate type to analyze. Defaults to 'edose'.
            show_monitors (bool, optional): Whether to display neutron monitor locations. Defaults to True.
                Note: Monitor plotting is only supported in AnisotropicMAIREPLUSevent.
            **plot_kwargs: Additional keyword arguments passed to plot_dose_map function
            
        Returns:
            Optional[plt.Axes]: Matplotlib axes object with the plot, or None if no data available
        """
        best_altitude = self._get_best_altitude(altitude)
        if best_altitude is None:
            print("No altitude data available.")
            return None
            
        if altitude is not None and abs(best_altitude - altitude) > 0.1:
            print(f"Requested altitude {altitude} km not available. Using nearest available altitude: {best_altitude} km")
            
        df = self.get_peak_dose_rate_map(best_altitude, dose_type)
        if df is None: return None
        df['altitude (km)'] = best_altitude
        args = {'plot_title': f'Peak {dose_type} at {best_altitude} km', 
                'dose_type': f'peak_{dose_type}_uSv_hr', 
                'legend_label': f'Peak {dose_type}'}
        args.update(plot_kwargs)
        
        ax, _ = plot_dose_map(df, **args)
        return ax

    def plot_integrated_dose_globe(self, altitude: Optional[float] = None, dose_type: str = 'edose',
                                  show_monitors: bool = True, **plot_kwargs: Any) -> Optional[plt.Figure]:
        """
        Plot integrated dose on 3D globe.
        
        Args:
            altitude (Optional[float], optional): Altitude in kilometers. If None, uses 12.192 km (40,000 ft) if available.
            dose_type (str, optional): Dose rate type to integrate. Defaults to 'edose'.
            show_monitors (bool, optional): Whether to display neutron monitor locations. Defaults to True.
                Note: Monitor plotting is only supported in AnisotropicMAIREPLUSevent.
            **plot_kwargs: Additional keyword arguments passed to plot_on_spherical_globe function
            
        Returns:
            Optional[plt.Figure]: Matplotlib figure object with the plot, or None if no data available
        """
        best_altitude = self._get_best_altitude(altitude)
        if best_altitude is None:
            print("No altitude data available.")
            return None
            
        if altitude is not None and abs(best_altitude - altitude) > 0.1:
            print(f"Requested altitude {altitude} km not available. Using nearest available altitude: {best_altitude} km")
            
        df = self.calculate_integrated_dose(best_altitude, dose_type)
        if df is None: return None
        args = {'plot_title': f'Integrated {dose_type} at {best_altitude} km', 
                'dose_type': f'integrated_{dose_type}_uSv', 
                'legend_label': f'Integrated {dose_type}'}
        args.update(plot_kwargs)
        
        return plot_on_spherical_globe(df, **args)

    def plot_peak_dose_rate_globe(self, altitude: Optional[float] = None, dose_type: str = 'edose',
                                  show_monitors: bool = True, **plot_kwargs: Any) -> Optional[plt.Figure]:
        """
        Plot peak dose rate on 3D globe.
        
        Args:
            altitude (Optional[float], optional): Altitude in kilometers. If None, uses 12.192 km (40,000 ft) if available.
            dose_type (str, optional): Dose rate type to analyze. Defaults to 'edose'.
            show_monitors (bool, optional): Whether to display neutron monitor locations. Defaults to True.
                Note: Monitor plotting is only supported in AnisotropicMAIREPLUSevent.
            **plot_kwargs: Additional keyword arguments passed to plot_on_spherical_globe function
            
        Returns:
            Optional[plt.Figure]: Matplotlib figure object with the plot, or None if no data available
        """
        best_altitude = self._get_best_altitude(altitude)
        if best_altitude is None:
            print("No altitude data available.")
            return None
            
        if altitude is not None and abs(best_altitude - altitude) > 0.1:
            print(f"Requested altitude {altitude} km not available. Using nearest available altitude: {best_altitude} km")
            
        df = self.get_peak_dose_rate_map(best_altitude, dose_type)
        if df is None: return None
        args = {'plot_title': f'Peak {dose_type} at {best_altitude} km', 
                'dose_type': f'peak_{dose_type}_uSv_hr', 
                'legend_label': f'Peak {dose_type}'}
        args.update(plot_kwargs)
        
        return plot_on_spherical_globe(df, **args)

    def plot_map_at_time(self, timestamp: dt.datetime, altitude: Optional[float] = None, ax: Optional[plt.Axes] = None,
                        nearest_ts: bool = True, show_monitors: bool = True, **plot_kwargs: Any) -> Optional[plt.Axes]:
        """
        Plot a 2D dose map at a given timestamp and altitude.
        
        Args:
            timestamp (dt.datetime): Timestamp to plot
            altitude (Optional[float], optional): Altitude in kilometers. If None, uses 12.192 km (40,000 ft) if available.
            ax (Optional[plt.Axes], optional): Matplotlib axes to use for plotting. Defaults to None.
            nearest_ts (bool, optional): If True, use nearest timestamp when exact not found. Defaults to True.
            show_monitors (bool, optional): Whether to display neutron monitor locations. Defaults to True.
                Note: Monitor plotting is only supported in AnisotropicMAIREPLUSevent.
            **plot_kwargs: Additional keyword arguments passed to plot_dose_map function
            
        Returns:
            Optional[plt.Axes]: Matplotlib axes object with the plot, or None if no data available
        """
        frame = self.get_dose_rate_frame(timestamp, nearest_ts)
        if frame is None: return None
        
        best_altitude = self._get_best_altitude(altitude)
        if best_altitude is None:
            print("No altitude data available.")
            return None
            
        if altitude is not None and abs(best_altitude - altitude) > 0.1:
            print(f"Requested altitude {altitude} km not available. Using nearest available altitude: {best_altitude} km")
        
        if ax: plot_kwargs['ax'] = ax
        
        return frame.plot_dose_map(altitude=best_altitude, **plot_kwargs)

    def plot_globe_at_time(self, timestamp: dt.datetime, altitude: Optional[float] = None, 
                          nearest_ts: bool = True, show_monitors: bool = True, **plot_kwargs: Any) -> Optional[plt.Figure]:
        """
        Plot a 3D globe dose at a given timestamp and altitude.
        
        Args:
            timestamp (dt.datetime): Timestamp to plot
            altitude (Optional[float], optional): Altitude in kilometers. If None, uses 12.192 km (40,000 ft) if available.
            nearest_ts (bool, optional): If True, use nearest timestamp when exact not found. Defaults to True.
            show_monitors (bool, optional): Whether to display neutron monitor locations. Defaults to True.
                Note: Monitor plotting is only supported in AnisotropicMAIREPLUSevent.
            **plot_kwargs: Additional keyword arguments passed to plot_on_globe function
            
        Returns:
            Optional[plt.Figure]: Matplotlib figure object with the plot, or None if no data available
        """
        frame = self.get_dose_rate_frame(timestamp, nearest_ts)
        if frame is None: return None
        
        best_altitude = self._get_best_altitude(altitude)
        if best_altitude is None:
            print("No altitude data available.")
            return None
            
        if altitude is not None and abs(best_altitude - altitude) > 0.1:
            print(f"Requested altitude {altitude} km not available. Using nearest available altitude: {best_altitude} km")
        
        return frame.plot_on_globe(altitude=best_altitude, **plot_kwargs)

    def export_to_netcdf(self, filename: str) -> None:
        """
        Export all dose rates to a NetCDF file.
        
        Args:
            filename (str): Path to output NetCDF file
            
        Returns:
            None
        """
        if not self.dose_rates: 
            print('No data to export')
            return
        # (existing netcdf logic can be called here via a helper or imported)
        # for brevity, call AniMAIRE_event.export_to_netcdf if needed
        AniMAIRE_event.export_to_netcdf(self, filename)

    def get_all_timestamps(self) -> List[dt.datetime]:
        """
        Return a sorted list of all timestamps represented in the event.
        
        Returns:
            List[dt.datetime]: List of datetime objects (sorted chronologically)
        """
        if not hasattr(self, 'dose_rates') or not self.dose_rates:
            return []
        return sorted(self.dose_rates.keys())

    def _get_monitor_info(self):
        """
        Try to extract neutron monitor information from the event instance.
        Returns a list of dicts with keys 'primary', 'secondary', 'normalisation', or a list of strings.
        """
        # Try AnisotropicMAIREPLUSevent style
        if hasattr(self, 'get_processed_monitor_sets'):
            try:
                return self.get_processed_monitor_sets()
            except Exception:
                pass
        # Try MAIREPLUS_event style (locations or names)
        info = []
        if hasattr(self, 'params'):
            # Try to get monitor locations if present
            for key in ['neutron_monitor_1_location', 'neutron_monitor_2_location', 'normalisation_monitor_location']:
                if key in self.params:
                    info.append(f"{key}: {self.params[key][0] if isinstance(self.params[key], list) else self.params[key]}")
            return info if info else None
        # Try direct attributes
        for key in ['neutron_monitor_1_location', 'neutron_monitor_2_location', 'normalisation_monitor_location']:
            if hasattr(self, key):
                info.append(f"{key}: {getattr(self, key)}")
        # Try nm_set attribute (if present)
        if hasattr(self, 'nm_set'):
            nm_set = getattr(self, 'nm_set')
            try:
                info.append(f"primary: {nm_set.get_primary().get_station_name()}")
                info.append(f"secondary: {nm_set.get_secondary().get_station_name()}")
                info.append(f"normalisation: {nm_set.get_normalisation().get_station_name()}")
            except Exception:
                pass
        return info if info else None

    def _get_monitor_locations(self) -> Optional[pd.DataFrame]:
        """
        Extract the locations of all neutron monitors used in this event.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame with columns (latitude, longitude, name, type) or None if no data available
        """
        locations = []
        
        # Try AnisotropicMAIREPLUSevent style
        if hasattr(self, 'isotropic_dose_runs') and self.isotropic_dose_runs:
            for nm_set in self.isotropic_dose_runs.keys():
                try:
                    # Extract primary monitor location
                    try:
                        lat, lon, alt = nm_set.get_primary().get_location()
                        name = nm_set.get_primary().get_station_name()
                        locations.append({
                            'latitude': lat, 'longitude': lon, 'name': name, 'type': 'primary'
                        })
                    except Exception as e:
                        print(f"Warning: Error extracting monitor locations: {e}")
                    
                    # Extract secondary monitor location
                    try:
                        lat, lon, alt = nm_set.get_secondary().get_location()
                        name = nm_set.get_secondary().get_station_name()
                        locations.append({
                            'latitude': lat, 'longitude': lon, 'name': name, 'type': 'secondary'
                        })
                    except Exception as e:
                        print(f"Warning: Error extracting monitor locations: {e}")
                    
                    # Extract normalization monitor location
                    try:
                        lat, lon, alt = nm_set.get_normalisation().get_location()
                        name = nm_set.get_normalisation().get_station_name()
                        locations.append({
                            'latitude': lat, 'longitude': lon, 'name': name, 'type': 'normalisation'
                        })
                    except Exception as e:
                        print(f"Warning: Error extracting monitor locations: {e}")
                except Exception as e:
                    print(f"Warning: Error extracting monitor locations: {e}")
        
        # Try MAIREPLUS_event style
        elif hasattr(self, 'params'):
            for key, type_name in [
                ('neutron_monitor_1_location', 'primary'),
                ('neutron_monitor_2_location', 'secondary'),
                ('normalisation_monitor_location', 'normalisation')
            ]:
                if key in self.params and self.params[key] is not None:
                    loc = self.params[key]
                    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                        locations.append({
                            'latitude': loc[0], 'longitude': loc[1], 
                            'name': f"{type_name} monitor", 'type': type_name
                        })
                    elif isinstance(loc, dict) and 'latitude' in loc and 'longitude' in loc:
                        locations.append({
                            'latitude': loc['latitude'], 'longitude': loc['longitude'],
                            'name': f"{type_name} monitor", 'type': type_name
                        })
        
        # If we found monitors, convert to DataFrame
        if locations:
            df = pd.DataFrame(locations)
            # Sort by type priority (primary > secondary > normalisation) and drop duplicates keeping first occurrence
            type_priority = {'primary': 0, 'secondary': 1, 'normalisation': 2}
            df['type_priority'] = df['type'].map(type_priority)
            df = df.sort_values('type_priority').drop_duplicates(subset=['latitude', 'longitude'])
            df = df.drop('type_priority', axis=1)
            
            # Normalize longitudes from 0-360 to -180 to +180
            df['longitude'] = df['longitude'].apply(lambda x: ((x + 180) % 360) - 180)
            
            return df
        return None

    def plot_timeseries_at_location(self, latitude: float, longitude: float, altitude: float, 
                                   dose_type: str = 'edose', ax: Optional[plt.Axes] = None,
                                   nearest_ts: bool = True, interpolation_method: str = 'linear', 
                                   **plot_kwargs: Any) -> Optional[plt.Axes]:
        """
        Plot time series of dose at a specific location.
        
        Args:
            latitude (float): Geographic latitude in degrees
            longitude (float): Geographic longitude in degrees
            altitude (float): Altitude in kilometers
            dose_type (str, optional): Dose rate type to plot. Defaults to 'edose'.
            ax (Optional[plt.Axes], optional): Matplotlib axes to use for plotting. Defaults to None.
            nearest_ts (bool, optional): If True, use nearest timestamp when exact not found. Defaults to True.
            interpolation_method (str, optional): Method for spatial interpolation. Defaults to 'linear'.
            **plot_kwargs: Additional keyword arguments passed to plot function
            
        Returns:
            Optional[plt.Axes]: Matplotlib axes object with the plot, or None if no data available
        """
        times = sorted(self.dose_rates.keys()); vals = []; tlist = []
        for ts in times:
            d = self.get_dose_rate_at_location(latitude, longitude, altitude, ts, dose_type, nearest_ts, interpolation_method)
            if d is not None: vals.append(d); tlist.append(ts)
        if not vals: return None
        if ax is None: fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(tlist, vals, label=dose_type, marker='o')
        ax.set_xlabel('Time (UTC)'); ax.set_ylabel(f'{dose_type} (uSv/hr)'); ax.legend(); ax.grid(True); plt.gcf().autofmt_xdate()
        return ax

    def to_dataframe(self) -> pd.DataFrame:
        """
        Concatenate all dose rate frames in the event into a single DataFrame, adding a 'timestamp' column for each row.
        
        Returns:
            pd.DataFrame: Combined DataFrame of all dose rate frames with an added 'timestamp' column.
        """
        if not hasattr(self, 'dose_rates') or not self.dose_rates:
            print("No dose rate data available. Run run_AniMAIRE() first.")
            return pd.DataFrame()
        frames = []
        for ts, frame in self.dose_rates.items():
            df = pd.DataFrame(frame).copy()
            df['timestamp'] = ts
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

class DoublePowerLawGaussianEvent(BaseAniMAIREEvent):
    """
    Event class for modeling solar particle events using double power-law rigidity spectrum and Gaussian pitch-angle distribution.
    
    This class handles the simulation of solar energetic particle events based on parameterized spectral
    data. It loads spectrum data from a CSV file, formats it appropriately, and provides methods
    to run the AniMAIRE simulations.
    
    Required input CSV format (columns):
      - Time: str or datetime (UTC) per spectrum row
      - J_0: float, normalization constant (particles/cm²/s/sr/GV)
      - gamma: float, spectral index for low rigidity part
      - d_gamma: float, modification to spectral index at higher rigidities
      - Sigma1: float, first Gaussian sigma for pitch-angle distribution (radians)
      - Sigma2: float, second Gaussian sigma for pitch-angle distribution (radians)
      - B: float, scaling factor for second Gaussian component
      - SymLat: float, reference pitch-angle latitude (GEO coords, degrees)
      - SymLong: float, reference pitch-angle longitude (GEO coords, degrees)
    
    Optional:
      - alpha_prime: float, default = math.pi (maximum pitch angle in radians)
      
    Attributes:
        spectra_file_path (str): Path to the CSV file containing spectral data
        raw_spectra_data (pd.DataFrame): Raw data loaded from the CSV file
        spectra (pd.DataFrame): Formatted spectral data used for simulations
        dose_rates (Dict[dt.datetime, DoseRateFrame]): Results from the simulation
    """
    
    def __init__(self, spectra_file_path: str) -> None:
        """
        Initialize an event with a spectral data file.
        
        Args:
            spectra_file_path (str): Path to the CSV file containing spectral data for the event
        """
        super().__init__()
        self.spectra_file_path: str = spectra_file_path
        self.raw_spectra_data: pd.DataFrame = pd.read_csv(spectra_file_path)
        self.spectra: pd.DataFrame = self.correctly_formatted_spectra()

    def correctly_formatted_spectra(self) -> pd.DataFrame:
        """
        Format the input spectral data to match AniMAIRE's expected format.
        
        Handles column renaming, datetime conversion, and adds default values for missing columns.
        
        Returns:
            pd.DataFrame: Correctly formatted spectra data
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
    
    def run_AniMAIRE(self, n_timestamps: Optional[int] = None, use_cache: bool = True, 
                    **kwargs: Any) -> Dict[dt.datetime, DoseRateFrame]:
        """
        Run AniMAIRE simulation for each timestamp in the spectral data.
        
        Processes each row of the spectral data to calculate dose rates at different
        locations and altitudes based on the parameterized particle spectrum.
        
        Args:
            n_timestamps (Optional[int], optional): Limit the number of timestamps to process. 
                                                   Useful for testing. Defaults to None (all timestamps).
            use_cache (bool, optional): Whether to use cached results for identical parameter sets. 
                                       Defaults to True.
            **kwargs: Additional keyword arguments passed to run_from_double_power_law_gaussian_distribution
            
        Returns:
            Dict[dt.datetime, DoseRateFrame]: Dictionary of dose rate frames keyed by timestamp
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
    """
    Cached version of run_from_double_power_law_gaussian_distribution function.
    
    This function provides a caching wrapper around the main AniMAIRE calculation
    function to avoid recomputing results for identical parameter sets.
    
    Args:
        J0 (float): Normalization constant (particles/cm²/s/sr/GV)
        gamma (float): Spectral index for the low rigidity part of the spectrum
        deltaGamma (float): Change in spectral index for the high rigidity part
        sigma_1 (float): First Gaussian width parameter (radians)
        sigma_2 (float): Second Gaussian width parameter (radians) 
        B (float): Scaling factor for the second Gaussian component
        alpha_prime (float): Maximum pitch angle (radians)
        reference_pitch_angle_latitude (float): Reference latitude for pitch angle (degrees)
        reference_pitch_angle_longitude (float): Reference longitude for pitch angle (degrees)
        date_and_time (Union[dt.datetime, np.datetime64]): Date and time for the calculation
        use_split_spectrum (bool): Whether to use a split spectrum (True) or single power law (False)
        **kwargs: Additional keyword arguments passed to the underlying function
        
    Returns:
        DoseRateFrame: Object containing the calculated dose rates and metadata
    """
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
    Create and run an AniMAIRE simulation using a GLE spectrum file.
    
    This is a convenience function that creates a DoublePowerLawGaussianEvent instance
    from a spectrum file and runs the simulation in one step.
    
    Args:
        GLE_spectrum_file (str): Path to the CSV file containing spectral data
        **kwargs: Additional keyword arguments passed to the run_AniMAIRE method
        
    Returns:
        DoublePowerLawGaussianEvent: Event object with calculated dose rates
    """
    event = DoublePowerLawGaussianEvent(GLE_spectrum_file)
    event.run_AniMAIRE(**kwargs)
    return event



