from AniMAIRE.AniMAIRE import run_from_double_power_law_gaussian_distribution
from AniMAIRE.DoseRateFrame import DoseRateFrame
from AniMAIRE.dose_plotting import create_gle_globe_animation, create_gle_map_animation, plot_dose_map, plot_on_spherical_globe
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import datetime as dt
from tqdm.auto import tqdm
from joblib import Memory

import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from IPython.display import HTML

import netCDF4

memory = Memory(location='./.AniMAIRE_event_cache')

class AniMAIRE_event():
    def __init__(self, spectra_file_path):
        self.spectra_file_path = spectra_file_path
        self.raw_spectra_data = pd.read_csv(spectra_file_path)
        self.spectra = self.correctly_formatted_spectra()

    def correctly_formatted_spectra(self):
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
    
    def summarize_spectra(self):
        """Provides a summary of the input spectral data."""
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
        for param, (min_val, max_val) in summary["Parameter Ranges"].items():
            print(f"  {param}: {min_val:.2e} to {max_val:.2e}")
        print("---------------------------")
            
        return summary

    def summarize_results(self):
        """Provides a comprehensive summary of the calculated dose rate results across all dose types."""
        if not hasattr(self, 'dose_rates') or not self.dose_rates:
            print("AniMAIRE simulation results not available. Run run_AniMAIRE() first.")
            return None

        timestamps = sorted(self.dose_rates.keys())
        first_frame = self.dose_rates[timestamps[0]]
        
        # Get a list of all dose rate columns
        # Standard dose rate columns from the README
        expected_dose_columns = {
            'edose': 'effective dose in µSv/hr',
            'adose': 'ambient dose equivalent in µSv/hr',
            'dosee': 'dose equivalent in µSv/hr',
            'tn1': '>1 MeV neutron flux in n/cm2/s',
            'tn2': '>10 MeV neutron flux in n/cm2/s',
            'tn3': '>60 MeV neutron flux in n/cm2/s',
            'SEU': 'single event upset rate in upsets/second/bit',
            'SEL': 'single event latch-up rate in latch-ups/second/device'
        }
        
        # Find which of the expected dose columns actually exist in the data
        available_dose_columns = [col for col in expected_dose_columns.keys() 
                               if col in first_frame.columns]
        
        # Also check for any derived columns that might be present
        derived_columns = [col for col in first_frame.columns 
                         if any(col.startswith(f"{base} ") for base in ['SEU', 'SEL'])]
        
        # Combine all dose rate columns
        dose_columns = available_dose_columns + derived_columns
        
        if not dose_columns:
            print("No dose rate columns found in the results.")
            return None
        
        # Dictionary to store peak values, times, and locations for each dose type
        dose_summaries = {}
        
        for dose_col in dose_columns:
            peak_dose_rate = 0.0
            time_of_peak = None
            location_of_peak = None  # (lat, lon, alt)
            total_dose_rates = []  # Store all dose rates to find overall peak
            
            # Create a friendly name for the dose column
            if dose_col in expected_dose_columns:
                column_description = expected_dose_columns[dose_col]
            elif dose_col.startswith('SEU ('):
                column_description = dose_col  # Use the full name for derived columns
            elif dose_col.startswith('SEL ('):
                column_description = dose_col
            else:
                column_description = f"dose rate ({dose_col})"

            for ts, frame in self.dose_rates.items():
                if dose_col not in frame.columns:
                    continue
                    
                # Find peak dose rate in the current frame
                current_peak = frame[dose_col].max()
                total_dose_rates.append(current_peak)  # Collect peak from each frame for stats
                
                if current_peak > peak_dose_rate:
                    peak_dose_rate = current_peak
                    time_of_peak = ts
                    
                    try:
                        # Find the row with the max value
                        max_row = frame.loc[frame[dose_col].idxmax()]
                        
                        # Get location info directly from the row
                        if 'latitude' in max_row and 'longitude' in max_row and 'altitude (km)' in max_row:
                            location_of_peak = (
                                max_row['latitude'],
                                max_row['longitude'],
                                max_row['altitude (km)']
                            )
                        else:
                            # Alternative approach if the dataframe structure is different
                            # Try using unravel_index but with better error handling
                            arr = frame[dose_col].values
                            peak_idx = np.argmax(arr)
                            
                            # Get the shape to determine dimensionality
                            arr_shape = arr.shape
                            
                            if len(arr_shape) >= 2:  # If we have a 2D or higher array
                                peak_indices = np.unravel_index(peak_idx, arr_shape)
                                
                                # Only try to access indices if they exist
                                lat_idx = peak_indices[0] if len(peak_indices) > 0 else 0
                                lon_idx = peak_indices[1] if len(peak_indices) > 1 else 0
                                
                                # Make sure we don't index out of bounds
                                if hasattr(frame, 'latitude') and hasattr(frame, 'longitude'):
                                    latitudes = frame.latitude.unique()
                                    longitudes = frame.longitude.unique()
                                    
                                    if lat_idx < len(latitudes) and lon_idx < len(longitudes):
                                        location_of_peak = (
                                            latitudes[lat_idx],
                                            longitudes[lon_idx],
                                            frame['altitude (km)'].iloc[0] if 'altitude (km)' in frame.columns else None
                                        )
                            else:
                                # For 1D data, get location from the row
                                if isinstance(frame.index, pd.MultiIndex):
                                    # If we have a MultiIndex, try to get lat/lon from there
                                    idx = frame.index[peak_idx]
                                    location_of_peak = (
                                        idx[0] if 'latitude' in frame.index.names else None,
                                        idx[1] if 'longitude' in frame.index.names else None,
                                        frame['altitude (km)'].iloc[0] if 'altitude (km)' in frame.columns else None
                                    )
                    except Exception as e:
                        print(f"Warning: Could not determine peak location for {dose_col}: {e}")
                        location_of_peak = None

            dose_summaries[dose_col] = {
                "Peak Value": peak_dose_rate,
                "Time of Peak": time_of_peak,
                "Location of Peak": location_of_peak,
                "Description": column_description
            }

        # Create general summary
        summary = {
            "Number of Timestamps": len(timestamps),
            "Time Range (UTC)": (timestamps[0], timestamps[-1]),
            "Altitude (km)": first_frame["altitude (km)"].iloc[0] if "altitude (km)" in first_frame.columns else None,
            "Latitude Range (deg)": (first_frame.latitude.min(), first_frame.latitude.max()) if "latitude" in first_frame.columns else None,
            "Longitude Range (deg)": (first_frame.longitude.min(), first_frame.longitude.max()) if "longitude" in first_frame.columns else None,
            "Dose Summaries": dose_summaries
        }
        
        # Print summary to console
        print("--- Simulation Results Summary ---")
        print(f"Number of Timestamps Processed: {summary['Number of Timestamps']}")
        print(f"Time Range (UTC): {summary['Time Range (UTC)'][0]} to {summary['Time Range (UTC)'][1]}")
        
        if summary['Altitude (km)'] is not None:
            print(f"Altitude (km): {summary['Altitude (km)']:.3f}")
            
        if summary['Latitude Range (deg)'] is not None:
            print(f"Latitude Range (deg): {summary['Latitude Range (deg)'][0]} to {summary['Latitude Range (deg)'][1]}")
            
        if summary['Longitude Range (deg)'] is not None:
            print(f"Longitude Range (deg): {summary['Longitude Range (deg)'][0]} to {summary['Longitude Range (deg)'][1]}")
        
        print("\nDose Rate Summaries:")
        for dose_col, dose_info in summary["Dose Summaries"].items():
            print(f"\n{dose_col} ({dose_info['Description']}):")
            print(f"  Peak Value: {dose_info['Peak Value']:.3e}")
            
            if dose_info['Time of Peak']:
                print(f"  Time of Peak: {dose_info['Time of Peak']}")
                
            if dose_info['Location of Peak']:
                peak_loc = dose_info['Location of Peak']
                loc_str = "  Location of Peak: "
                if peak_loc[0] is not None:
                    loc_str += f"Lat {peak_loc[0]:.2f}, "
                if peak_loc[1] is not None:
                    loc_str += f"Lon {peak_loc[1]:.2f}, "
                if peak_loc[2] is not None:
                    loc_str += f"Alt {peak_loc[2]:.3f} km"
                print(loc_str)
                
        print("------------------------------")

        return summary

    def run_AniMAIRE(self, n_timestamps: int = None, use_cache: bool = True, **kwargs):
        # Run AniMAIRE

        self.dose_rates = {}  # Use dictionary instead of list
        for index, spectrum in self.spectra.iterrows():

            # Run AniMAIRE
            total_spectra = len(self.spectra)
            percentage_complete = (index / total_spectra) * 100
            print(f"Running AniMAIRE for spectrum {index} ({percentage_complete:.1f}% complete)")
            # Print the datetime for the current spectrum
            print(f"Processing spectrum for datetime: {spectrum['datetime']}")
            
            # Determine whether to use caching or not
            if use_cache:
                # Use Streamlit's cached function
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
    
    def get_available_altitudes(self):
        """
        Get all unique altitude values available across all dose rate frames.

        Returns:
        --------
        list
            Sorted list of unique altitude values in km across all frames
        """
        if not hasattr(self, 'dose_rates') or not self.dose_rates:
            print("No dose rate data available. Run run_AniMAIRE() first.")
            return []
            
        # Get a set of all altitudes from all timestamps
        all_altitudes = set()
        for ts, frame in self.dose_rates.items():
            frame_altitudes = frame.get_altitudes()
            all_altitudes.update(frame_altitudes)
            
        return sorted(all_altitudes)
    
    def create_gle_map_animation(self, altitude=12.192, save_gif=False, save_mp4=False):
        return create_gle_map_animation(self.dose_rates, altitude, save_gif, save_mp4)
    
    def create_gle_globe_animation(self, altitude=12.192, save_gif=False, save_mp4=False):
        return create_gle_globe_animation(self.dose_rates, altitude, save_gif, save_mp4)
    
    def create_spectra_animation(gle_object, output_filename='GLE74_spectra_animation.mp4', fps=2, 
                                spectra_xlim=(0.1, 20), spectra_ylim=(1e-16, 1e14)):
        """
        Create an animation showing the evolution of rigidity spectra over time.
        
        Parameters:
        -----------
        gle_object : GLEObject
            The GLE object containing dose rate frames
        output_filename : str
            Filename for the output animation
        fps : int
            Frames per second for the animation
        spectra_xlim : tuple
            x-axis limits (min, max) for the spectra plot (Rigidity in GV)
        spectra_ylim : tuple
            y-axis limits (min, max) for the spectra plot (Flux)
        """
        # Get the timestamps in chronological order
        timestamps = sorted(gle_object.dose_rates.keys())
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Function to update the plot for each frame
        def update(frame):
            # Clear previous plot
            ax.clear()
            
            # Get the timestamp and dose rate frame for this animation frame
            timestamp = timestamps[frame]
            dose_rate_frame = gle_object.dose_rates[timestamp]
            
            # Plot the spectra
            dose_rate_frame.plot_spectra(ax=ax)
            ax.set_title(f'Rigidity Spectra at {timestamp.strftime("%Y-%m-%d %H:%M")}')
            ax.set_xlim(spectra_xlim)
            ax.set_ylim(spectra_ylim)
            
            return ax,
        
        # Create the animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(timestamps), blit=False, interval=1000/fps
        )
        
        # Save the animation
        ani.save(output_filename, writer='ffmpeg', fps=fps)
        plt.close(fig)
        
        # Display the animation in the notebook
        return HTML(ani.to_jshtml())


    def create_pad_animation(gle_object, output_filename='GLE74_pad_animation.mp4', fps=2, 
                            pad_xlim=(0, 3.14), pad_ylim=(0, 1.2)):
        """
        Create an animation showing the evolution of pitch angle distributions over time.
        
        Parameters:
        -----------
        gle_object : GLEObject
            The GLE object containing dose rate frames
        output_filename : str
            Filename for the output animation
        fps : int
            Frames per second for the animation
        pad_xlim : tuple
            x-axis limits (min, max) for the pitch angle distribution plot (radians)
        pad_ylim : tuple
            y-axis limits (min, max) for the pitch angle distribution plot (relative intensity)
        """
        # Get the timestamps in chronological order
        timestamps = sorted(gle_object.dose_rates.keys())
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Function to update the plot for each frame
        def update(frame):
            # Clear previous plot
            ax.clear()
            
            # Get the timestamp and dose rate frame for this animation frame
            timestamp = timestamps[frame]
            dose_rate_frame = gle_object.dose_rates[timestamp]
            
            # Plot the pitch angle distribution
            dose_rate_frame.plot_pitch_angle_distributions(ax=ax)
            ax.set_title(f'Pitch Angle Distributions at {timestamp.strftime("%Y-%m-%d %H:%M")}')
            ax.set_xlim(pad_xlim)
            ax.set_ylim(pad_ylim)
            
            return ax,
        
        # Create the animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(timestamps), blit=False, interval=1000/fps
        )
        
        # Save the animation
        ani.save(output_filename, writer='ffmpeg', fps=fps)
        plt.close(fig)
        
        # Display the animation in the notebook
        return HTML(ani.to_jshtml())


    def create_combined_animation(gle_object, output_filename='GLE74_combined_animation.mp4', fps=2, 
                                spectra_xlim=(0.1, 20), spectra_ylim=(1e-16, 1e14), 
                                pad_xlim=(0, 3.14), pad_ylim=(0, 1.2)):
        """
        Create an animation showing the evolution of both rigidity spectra and pitch angle distributions over time.
        
        Parameters:
        -----------
        gle_object : GLEObject
            The GLE object containing dose rate frames
        output_filename : str
            Filename for the output animation
        fps : int
            Frames per second for the animation
        spectra_xlim : tuple
            x-axis limits (min, max) for the spectra plot (Rigidity in GV)
        spectra_ylim : tuple
            y-axis limits (min, max) for the spectra plot (Flux)
        pad_xlim : tuple
            x-axis limits (min, max) for the pitch angle distribution plot (radians)
        pad_ylim : tuple
            y-axis limits (min, max) for the pitch angle distribution plot (relative intensity)
        """
        # Get the timestamps in chronological order
        timestamps = sorted(gle_object.dose_rates.keys())
        
        # Create the figure and subplots
        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])  # For spectra
        ax2 = fig.add_subplot(gs[0, 1])  # For pitch angle distributions
        
        # Function to update the plot for each frame
        def update(frame):
            # Clear previous plots
            ax1.clear()
            ax2.clear()
            
            # Get the timestamp and dose rate frame for this animation frame
            timestamp = timestamps[frame]
            dose_rate_frame = gle_object.dose_rates[timestamp]
            
            # Plot the spectra on the left subplot
            dose_rate_frame.plot_spectra(ax=ax1)
            ax1.set_title(f'Rigidity Spectra at {timestamp.strftime("%Y-%m-%d %H:%M")}')
            ax1.set_xlim(spectra_xlim)
            ax1.set_ylim(spectra_ylim)
            
            # Plot the pitch angle distribution on the right subplot
            dose_rate_frame.plot_pitch_angle_distributions(ax=ax2)
            ax2.set_title(f'Pitch Angle Distributions at {timestamp.strftime("%Y-%m-%d %H:%M")}')
            ax2.set_xlim(pad_xlim)
            ax2.set_ylim(pad_ylim)
            
            # Add a main title
            plt.suptitle(f'timestamp - {timestamp.strftime("%Y-%m-%d %H:%M")}', fontsize=14)
            
            return ax1, ax2
        
        # Create the animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(timestamps), blit=False, interval=1000/fps
        )
        
        # Save the animation
        ani.save(output_filename, writer='ffmpeg', fps=fps)
        plt.close(fig)
        
        # Display the animation in the notebook
        return HTML(ani.to_jshtml())

    # --- Data Access Methods ---

    def get_dose_rate_frame(self, timestamp: dt.datetime, nearest: bool = True) -> DoseRateFrame | None:
        """
        Retrieves the DoseRateFrame for a specific timestamp or the nearest one.

        Args:
            timestamp: The target timestamp.
            nearest: If True, find the nearest available timestamp if the exact one is not found. 
                     If False, return None if the exact timestamp is missing.

        Returns:
            The DoseRateFrame corresponding to the timestamp, or None.
        """
        if not hasattr(self, 'dose_rates') or not self.dose_rates:
            print("Warning: No dose rate data available. Run run_AniMAIRE first.")
            return None

        if timestamp in self.dose_rates:
            return self.dose_rates[timestamp]
        elif nearest:
            available_times = np.array(sorted(self.dose_rates.keys()))
            # Convert target timestamp and available times to numerical representation for comparison
            target_ts_num = timestamp.timestamp()
            available_times_num = np.array([t.timestamp() for t in available_times])
            
            nearest_idx = np.argmin(np.abs(available_times_num - target_ts_num))
            nearest_time = available_times[nearest_idx]
            print(f"Warning: Exact timestamp {timestamp} not found. Using nearest: {nearest_time}")
            return self.dose_rates[nearest_time]
        else:
            print(f"Warning: Exact timestamp {timestamp} not found.")
            return None

    def get_dose_rate_at_location(self, latitude: float, longitude: float, altitude: float, timestamp: dt.datetime, dose_type: str = 'edose', nearest_ts: bool = True, interpolation_method: str = 'linear') -> float | None:
        """
        Retrieves the interpolated dose rate at a specific geographic location, altitude, and time.

        Args:
            latitude: Target latitude in degrees.
            longitude: Target longitude in degrees.
            altitude: Target altitude in km.
            timestamp: Target timestamp.
            dose_type: The name of the dose rate column to interpolate (e.g., 'edose', 'adose'). Defaults to 'edose'.
            nearest_ts: If True, use the nearest available timestamp if the exact one is not found.
            interpolation_method: Method for griddata ('linear', 'nearest', 'cubic'). Default is 'linear'.

        Returns:
            The interpolated dose rate in uSv/hr, or None if data is unavailable or interpolation fails.
        """
        frame = self.get_dose_rate_frame(timestamp, nearest=nearest_ts)
        if frame is None:
            return None # Message already printed by get_dose_rate_frame

        # Filter data near the target altitude (using pandas query for efficiency)
        # Use a small tolerance if altitudes might not match exactly
        alt_tolerance = 0.1 # Adjust tolerance as needed based on altitude grid spacing
        data_at_alt = frame.query(f"`altitude (km)` >= {altitude - alt_tolerance} and `altitude (km)` <= {altitude + alt_tolerance}")
        
        if data_at_alt.empty:
             print(f"Warning: No data found near altitude {altitude} km for timestamp {frame.timestamp}.")
             # Optional: Could try interpolating between altitude layers if available
             return None

        # If multiple altitude layers match within tolerance, maybe take the closest one or average?
        # For simplicity, let's take the one closest to the target altitude if multiple exist.
        if data_at_alt['altitude (km)'].nunique() > 1:
             closest_alt = data_at_alt.iloc[(data_at_alt['altitude (km)'] - altitude).abs().argsort()]['altitude (km)'].iloc[0]
             data_at_alt = data_at_alt[data_at_alt['altitude (km)'] == closest_alt]

        if data_at_alt.empty: # Check again after potential filtering
            print(f"Warning: No data points remaining after altitude selection for timestamp {frame.timestamp}.")
            return None

        # Prepare data for interpolation
        points = data_at_alt[['latitude', 'longitude']].values
        if dose_type not in data_at_alt.columns:
            print(f"Error: Specified dose_type '{dose_type}' not found in DataFrame columns: {data_at_alt.columns}")
            return None
        values = data_at_alt[dose_type].values 
        target_point = (latitude, longitude)

        try:
            # Perform 2D interpolation using griddata
            interpolated_dose = griddata(points, values, target_point, method=interpolation_method)
            
            if np.isnan(interpolated_dose):
                # Handle cases where interpolation returns NaN (e.g., target outside convex hull for linear/cubic)
                # Try nearest neighbor as a fallback?
                print(f"Warning: Interpolation ({interpolation_method}) resulted in NaN for Lat={latitude}, Lon={longitude} at {frame.timestamp}. Target might be outside data coverage.")
                # Optional: Fallback to nearest neighbor
                # interpolated_dose = griddata(points, values, target_point, method='nearest')
                # if np.isnan(interpolated_dose):
                #     print("Fallback interpolation ('nearest') also failed.")
                #     return None
                return None # Return None if primary interpolation fails

            return float(interpolated_dose)
        except Exception as e:
            print(f"Error during interpolation at {frame.timestamp} for Lat={latitude}, Lon={longitude}: {e}")
            return None

    # --- Analysis Methods ---

    def _get_target_grid(self, target_grid=None, n_lat=90, n_lon=180):
        """ Helper to define the target grid for interpolation. """
        if target_grid is not None:
            return target_grid # Use user-provided grid
            
        # If no grid provided, create a default global grid
        latitudes = np.linspace(-90, 90, n_lat)
        longitudes = np.linspace(-180, 180, n_lon)
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        target_grid_points = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T
        return target_grid_points
        
    def _calculate_time_deltas(self):
        """ Helper to calculate time intervals between frames in hours. """
        timestamps = sorted(self.dose_rates.keys())
        if len(timestamps) < 2:
            # Assume a default duration (e.g., 1 hour) if only one frame
            # Or handle this case based on desired behavior
            return np.array([1.0]) if len(timestamps) == 1 else np.array([]) 
            
        # Calculate time differences in seconds
        time_diffs_sec = np.diff([ts.timestamp() for ts in timestamps])
        
        # Use trapezoidal rule: dt for a point is the average of the intervals before and after it
        dt_sec = np.zeros(len(timestamps))
        dt_sec[0] = time_diffs_sec[0] / 2
        dt_sec[-1] = time_diffs_sec[-1] / 2
        dt_sec[1:-1] = (time_diffs_sec[:-1] + time_diffs_sec[1:]) / 2
        
        # Convert deltas to hours
        dt_hours = dt_sec / 3600.0
        return dt_hours

    def calculate_integrated_dose(self, altitude: float, dose_type: str = 'edose') -> pd.DataFrame | None:
        """
        Calculates the total integrated dose (in uSv) over the event duration
        on the native grid points at a specific altitude, for a given dose type.
        Assumes the spatial grid is consistent across all timestamps.

        Args:
            altitude: The altitude (in km) for which to calculate integrated dose.
            dose_type: The name of the dose rate column to integrate (e.g., 'edose', 'adose'). Defaults to 'edose'.

        Returns:
            A pandas DataFrame with columns ['latitude', 'longitude', 'integrated_<dose_type>_uSv'],
            representing the dose integrated over time at each native grid point, or None if calculation fails.
        """
        if not hasattr(self, 'dose_rates') or not self.dose_rates:
            print("Warning: No dose rate data available. Run run_AniMAIRE first.")
            return None
            
        timestamps = sorted(self.dose_rates.keys())
        if len(timestamps) < 1:
             print("Warning: Need at least one dose rate frame to calculate integrated dose.")
             return None

        # --- Determine native grid and initialize accumulator --- 
        first_frame = self.dose_rates[timestamps[0]]
        # Filter first frame to get coordinates at the target altitude
        alt_tolerance = 0.1 
        data_at_alt_first = first_frame.query(f"`altitude (km)` >= {altitude - alt_tolerance} and `altitude (km)` <= {altitude + alt_tolerance}")
        if data_at_alt_first.empty:
            print(f"Warning: No data found near altitude {altitude} km in the first frame. Cannot determine grid.")
            return None
        if data_at_alt_first['altitude (km)'].nunique() > 1:
            closest_alt = data_at_alt_first.iloc[(data_at_alt_first['altitude (km)'] - altitude).abs().argsort()]['altitude (km)'].iloc[0]
            data_at_alt_first = data_at_alt_first[data_at_alt_first['altitude (km)'] == closest_alt]
        if data_at_alt_first.empty:
             print(f"Warning: No data points remaining after altitude selection in the first frame. Cannot determine grid.")
             return None
             
        # Use lat/lon from the filtered first frame as the reference grid index
        native_index = pd.MultiIndex.from_frame(data_at_alt_first[['latitude', 'longitude']])
        integrated_dose_series = pd.Series(0.0, index=native_index, dtype=float) # Accumulator Series
        # --- End Grid Determination --- 
        
        dt_hours = self._calculate_time_deltas()
        if len(dt_hours) != len(timestamps):
             print("Error: Mismatch between number of timestamps and calculated time deltas.")
             return None

        print(f"Calculating integrated {dose_type} dose at {altitude} km on native grid...")
        for i, timestamp in enumerate(tqdm(timestamps)):
            frame = self.dose_rates[timestamp]
            
            # Filter data near the target altitude
            data_at_alt = frame.query(f"`altitude (km)` >= {altitude - alt_tolerance} and `altitude (km)` <= {altitude + alt_tolerance}")
            if data_at_alt.empty:
                # print(f"Warning: No data found near altitude {altitude} km for timestamp {timestamp}. Skipping frame.")
                continue # Silently skip frame if no data at target alt
            if data_at_alt['altitude (km)'].nunique() > 1:
                closest_alt = data_at_alt.iloc[(data_at_alt['altitude (km)'] - altitude).abs().argsort()]['altitude (km)'].iloc[0]
                data_at_alt = data_at_alt[data_at_alt['altitude (km)'] == closest_alt]
            if data_at_alt.empty:
                continue
                
            if dose_type not in data_at_alt.columns:
                 print(f"Warning: Specified dose_type '{dose_type}' not found in frame at {timestamp}. Skipping frame contribution.")
                 continue
                 
            # Prepare current frame's data with the native index for alignment
            current_frame_series = data_at_alt.set_index(['latitude', 'longitude'])[dose_type]
            
            # Align with the accumulator series (add 0 for missing points in current frame) and add contribution
            integrated_dose_series = integrated_dose_series.add(current_frame_series * dt_hours[i], fill_value=0.0)

        # Create result DataFrame
        output_col_name = f'integrated_{dose_type}_uSv'
        result_df = integrated_dose_series.reset_index()
        result_df.rename(columns={0: output_col_name}, inplace=True)
            
        return result_df

    def get_peak_dose_rate_map(self, altitude: float, dose_type: str = 'edose') -> pd.DataFrame | None:
        """
        Finds the maximum dose rate experienced at each native grid point over the event duration
        at a specific altitude, for a given dose type.
        Assumes the spatial grid is consistent across all timestamps.

        Args:
            altitude: The altitude (in km) to analyze.
            dose_type: The name of the dose rate column to analyze (e.g., 'edose', 'adose'). Defaults to 'edose'.

        Returns:
            A pandas DataFrame with columns ['latitude', 'longitude', 'peak_<dose_type>_uSv_hr'],
            representing the peak dose rate at each native grid point, or None if calculation fails.
        """
        if not hasattr(self, 'dose_rates') or not self.dose_rates:
            print("Warning: No dose rate data available. Run run_AniMAIRE first.")
            return None
            
        timestamps = sorted(self.dose_rates.keys())
        if not timestamps:
             print("Warning: No timestamps available.")
             return None

        # --- Determine native grid and initialize accumulator --- 
        first_frame = self.dose_rates[timestamps[0]]
        alt_tolerance = 0.1 # Use same tolerance as before
        data_at_alt_first = first_frame.query(f"`altitude (km)` >= {altitude - alt_tolerance} and `altitude (km)` <= {altitude + alt_tolerance}")
        if data_at_alt_first.empty:
            print(f"Warning: No data found near altitude {altitude} km in the first frame. Cannot determine grid.")
            return None
        if data_at_alt_first['altitude (km)'].nunique() > 1:
            closest_alt = data_at_alt_first.iloc[(data_at_alt_first['altitude (km)'] - altitude).abs().argsort()]['altitude (km)'].iloc[0]
            data_at_alt_first = data_at_alt_first[data_at_alt_first['altitude (km)'] == closest_alt]
        if data_at_alt_first.empty:
             print(f"Warning: No data points remaining after altitude selection in the first frame. Cannot determine grid.")
             return None
             
        native_index = pd.MultiIndex.from_frame(data_at_alt_first[['latitude', 'longitude']])
        peak_dose_series = pd.Series(-np.inf, index=native_index, dtype=float) # Accumulator Series, init with -inf
        # --- End Grid Determination --- 

        print(f"Calculating peak {dose_type} rate map at {altitude} km on native grid...")
        for timestamp in tqdm(timestamps):
            frame = self.dose_rates[timestamp]
            
            # Filter data near the target altitude
            data_at_alt = frame.query(f"`altitude (km)` >= {altitude - alt_tolerance} and `altitude (km)` <= {altitude + alt_tolerance}")
            if data_at_alt.empty:
                continue # Silently skip frame
            if data_at_alt['altitude (km)'].nunique() > 1:
                closest_alt = data_at_alt.iloc[(data_at_alt['altitude (km)'] - altitude).abs().argsort()]['altitude (km)'].iloc[0]
                data_at_alt = data_at_alt[data_at_alt['altitude (km)'] == closest_alt]
            if data_at_alt.empty:
                continue

            if dose_type not in data_at_alt.columns:
                 print(f"Warning: Specified dose_type '{dose_type}' not found in frame at {timestamp}. Skipping this frame.")
                 continue
                 
            # Prepare current frame's data with the native index for alignment
            current_frame_series = data_at_alt.set_index(['latitude', 'longitude'])[dose_type]
            
            # Align with the accumulator series (fill missing points with -inf) and update maximum
            aligned_current, aligned_peak = current_frame_series.align(peak_dose_series, join='right', fill_value=-np.inf)
            peak_dose_series = pd.Series(np.maximum(aligned_peak.values, aligned_current.values), index=peak_dose_series.index)
        
        # Replace any remaining -inf with NaN (points never covered by data)
        peak_dose_series.replace(-np.inf, np.nan, inplace=True)
        
        # Create result DataFrame
        output_col_name = f'peak_{dose_type}_uSv_hr'
        result_df = peak_dose_series.reset_index()
        result_df.rename(columns={0: output_col_name}, inplace=True)
            
        return result_df

    # --- Plotting and Animation Methods ---

    def plot_integrated_dose_map(self, altitude: float, dose_type: str = 'edose', **plot_kwargs):
        """
        Calculates and plots the integrated dose map for a given altitude and dose type,
        using the native grid from the simulation results.

        Args:
            altitude: The altitude (in km) for analysis.
            dose_type: The dose rate column to integrate and plot (e.g., 'edose', 'adose').
            **plot_kwargs: Additional keyword arguments passed to the plot_dose_map function.
                          (e.g., cmap, vmin, vmax).

        Returns:
            Matplotlib axes object containing the map, or None if plotting fails.
        """
        # Calculate the specific integrated dose type
        output_col_name = f'integrated_{dose_type}_uSv'
        integrated_dose_df = self.calculate_integrated_dose(
            altitude=altitude, 
            dose_type=dose_type # Pass dose_type here
        )
        
        if integrated_dose_df is None or integrated_dose_df.empty:
            print("Error: Could not calculate integrated dose for plotting.")
            return None
            
        # Add the altitude column back in, as plot_dose_map expects it
        integrated_dose_df['altitude (km)'] = altitude
            
        # Prepare plot arguments dynamically
        legend_label = f'Integrated {dose_type.replace("_"," ")} (uSv)'
        plot_args = {
            'plot_title': f'Integrated {dose_type.replace("_"," ")} at {altitude:.1f} km ({self.spectra["datetime"].min().strftime("%Y-%m-%d %H:%M")} to {self.spectra["datetime"].max().strftime("%H:%M")}) ',
            'dose_type': output_col_name, # Use the calculated column name
            'legend_label': legend_label
        }
        plot_args.update(plot_kwargs) # Allow user to override defaults
        
        # Assuming plot_dose_map can take a value_column argument
        # plot_dose_map creates its own figure/axes internally
        map_ax, _ = plot_dose_map(integrated_dose_df, **plot_args) # Removed ax=ax
        return map_ax

    def plot_peak_dose_rate_map(self, altitude: float, dose_type: str = 'edose', **plot_kwargs):
        """
        Calculates and plots the peak dose rate map for a given altitude and dose type,
        using the native grid from the simulation results.

        Args:
            altitude: The altitude (in km) for analysis.
            dose_type: The dose rate column to analyze and plot (e.g., 'edose', 'adose').
            **plot_kwargs: Additional keyword arguments passed to the plot_dose_map function.

        Returns:
            Matplotlib axes object containing the map, or None if plotting fails.
        """
        # Calculate the specific peak dose rate type
        output_col_name = f'peak_{dose_type}_uSv_hr'
        peak_dose_rate_df = self.get_peak_dose_rate_map(
            altitude=altitude, 
            dose_type=dose_type # Pass dose_type here
        )
        
        if peak_dose_rate_df is None or peak_dose_rate_df.empty:
            print("Error: Could not calculate peak dose rate map for plotting.")
            return None
            
        # Add the altitude column back in, as plot_dose_map expects it
        peak_dose_rate_df['altitude (km)'] = altitude
            
        # Prepare plot arguments dynamically
        legend_label = f'Peak {dose_type.replace("_"," ")} (uSv/hr)'
        plot_args = {
            'plot_title': f'Peak {dose_type.replace("_"," ")} Rate at {altitude:.1f} km ({self.spectra["datetime"].min().strftime("%Y-%m-%d %H:%M")} to {self.spectra["datetime"].max().strftime("%H:%M")}) ',
            'dose_type': output_col_name, # Use the calculated column name
            'legend_label': legend_label
        }
        plot_args.update(plot_kwargs)
        
        try:
            # Assuming plot_dose_map can take a value_column argument
            # plot_dose_map creates its own figure/axes internally
            map_ax, _ = plot_dose_map(peak_dose_rate_df, **plot_args) # Removed ax=ax
            return map_ax
        except Exception as e:
            print(f"Error plotting peak dose rate map: {e}")
            return None

    def plot_integrated_dose_globe(self, altitude: float, dose_type: str = 'edose', **plot_kwargs):
        """
        Calculates and plots the integrated dose on a 3D globe for a given altitude and dose type,
        using the native grid from the simulation results.

        Args:
            altitude: The altitude (in km) for analysis.
            dose_type: The dose rate column to integrate and plot (e.g., 'edose', 'adose').
            **plot_kwargs: Additional keyword arguments passed to plot_on_spherical_globe.

        Returns:
            Matplotlib Figure object containing the globe, or None if plotting fails.
        """
        # Calculate the specific integrated dose type
        output_col_name = f'integrated_{dose_type}_uSv'
        integrated_dose_df = self.calculate_integrated_dose(
            altitude=altitude, 
            dose_type=dose_type # Pass dose_type here
        )
        
        if integrated_dose_df is None or integrated_dose_df.empty:
            print("Error: Could not calculate integrated dose for plotting.")
            return None

        # Prepare plot arguments dynamically
        legend_label = f'Integrated {dose_type.replace("_"," ")} (uSv)'
        plot_args = {
            'plot_title': f'Integrated {dose_type.replace("_"," ")} at {altitude:.1f} km',
            'dose_type': output_col_name, # Use the calculated column name
            'legend_label': legend_label
        }
        plot_args.update(plot_kwargs)

        try:
            globe_fig = plot_on_spherical_globe(integrated_dose_df, **plot_args)
            return globe_fig
        except Exception as e:
            print(f"Error plotting integrated dose globe: {e}")
            return None

    def plot_peak_dose_rate_globe(self, altitude: float, dose_type: str = 'edose', **plot_kwargs):
        """
        Calculates and plots the peak dose rate on a 3D globe for a given altitude and dose type,
        using the native grid from the simulation results.

        Args:
            altitude: The altitude (in km) for analysis.
            dose_type: The dose rate column to analyze and plot (e.g., 'edose', 'adose').
            **plot_kwargs: Additional keyword arguments passed to plot_on_spherical_globe.

        Returns:
            Matplotlib Figure object containing the globe, or None if plotting fails.
        """
        # Calculate the specific peak dose rate type
        output_col_name = f'peak_{dose_type}_uSv_hr'
        peak_dose_rate_df = self.get_peak_dose_rate_map(
            altitude=altitude, 
            dose_type=dose_type # Pass dose_type here
        )
        
        if peak_dose_rate_df is None or peak_dose_rate_df.empty:
            print("Error: Could not calculate peak dose rate map for plotting.")
            return None

        # Prepare plot arguments dynamically
        legend_label = f'Peak {dose_type.replace("_"," ")} (uSv/hr)'
        plot_args = {
            'plot_title': f'Peak {dose_type.replace("_"," ")} Rate at {altitude:.1f} km',
            'dose_type': output_col_name, # Use the calculated column name
            'legend_label': legend_label
        }
        plot_args.update(plot_kwargs)

        try:
            globe_fig = plot_on_spherical_globe(peak_dose_rate_df, **plot_args)
            return globe_fig
        except Exception as e:
            print(f"Error plotting peak dose rate globe: {e}")
            return None

    def plot_timeseries_at_location(self, latitude: float, longitude: float, altitude: float, dose_type: str = 'edose', ax=None, nearest_ts: bool = True, interpolation_method: str = 'linear', **plot_kwargs):
        """
        Plots the dose rate time series for a specific dose type at a specific 
        geographic location and altitude.

        Args:
            latitude: Target latitude in degrees.
            longitude: Target longitude in degrees.
            altitude: Target altitude in km.
            dose_type: The dose rate column to plot (e.g., 'edose', 'adose'). Defaults to 'edose'.
            ax: Matplotlib axes object to plot on. If None, a new figure and axes are created.
            nearest_ts: Passed to get_dose_rate_at_location for timestamp matching.
            interpolation_method: Passed to get_dose_rate_at_location for interpolation.
            **plot_kwargs: Additional keyword arguments passed to ax.plot().

        Returns:
            Matplotlib axes object containing the plot.
        """
        if not hasattr(self, 'dose_rates') or not self.dose_rates:
            print("Warning: No dose rate data available. Run run_AniMAIRE first.")
            return None

        timestamps = sorted(self.dose_rates.keys())
        dose_rates_at_loc = []
        valid_timestamps = []

        print(f"Extracting {dose_type} time series for Lat={latitude}, Lon={longitude}, Alt={altitude} km...")
        for ts in timestamps:
            dose_rate = self.get_dose_rate_at_location(
                latitude, longitude, altitude, ts, 
                dose_type=dose_type, # Pass dose_type here
                nearest_ts=nearest_ts, 
                interpolation_method=interpolation_method
            )
            if dose_rate is not None:
                dose_rates_at_loc.append(dose_rate)
                valid_timestamps.append(ts)
            # else: # Optional: Add handling for missing data points in the series
            #     dose_rates_at_loc.append(np.nan)
            #     valid_timestamps.append(ts)

        if not dose_rates_at_loc:
            print("Error: Could not retrieve any valid dose rate data for the specified location.")
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        # Set default plot settings if not provided in kwargs
        plot_settings = {
            'label': f'{dose_type} @ Lat={latitude}, Lon={longitude}, Alt={altitude} km',
            'marker': 'o',
            'linestyle': '-'
        }
        plot_settings.update(plot_kwargs) # Override defaults with user kwargs

        ax.plot(valid_timestamps, dose_rates_at_loc, **plot_settings)

        ax.set_xlabel("Time (UTC)")
        # Make Y label dynamic based on dose_type
        y_label = f'{dose_type.replace("_"," ").capitalize()} Rate (uSv/hr)' 
        ax.set_ylabel(y_label)
        ax.set_title(f"{dose_type.replace('_',' ').capitalize()} Time Series")
        ax.grid(True)
        ax.legend()
        fig.autofmt_xdate() # Improve date formatting on x-axis

        return ax

    def plot_map_at_time(self, timestamp: dt.datetime, altitude: float, ax=None, nearest_ts: bool = True, **plot_kwargs):
        """
        Plots a 2D dose rate map for a specific timestamp and altitude.

        Args:
            timestamp: The target timestamp for the map.
            altitude: The altitude (in km) for the map.
            ax: Matplotlib axes object to plot on. If None, a new figure and axes are created.
            nearest_ts: If True, use the nearest available timestamp if the exact one is not found.
            **plot_kwargs: Additional keyword arguments passed to the underlying DoseRateFrame.plot_dose_map().

        Returns:
            Matplotlib axes object containing the map, or None if plotting fails.
        """
        frame = self.get_dose_rate_frame(timestamp, nearest=nearest_ts)
        if frame is None:
            return None # Message already printed

        try:
            # Ensure the axis is handled correctly if provided
            if ax is not None:
                plot_kwargs['ax'] = ax 
                
            # Call the DoseRateFrame's plotting method
            map_ax = frame.plot_dose_map(altitude=altitude, **plot_kwargs)
            return map_ax
        except Exception as e:
            print(f"Error plotting map for timestamp {frame.timestamp}: {e}")
            return None

    def plot_globe_at_time(self, timestamp: dt.datetime, altitude: float, nearest_ts: bool = True, **plot_kwargs):
        """
        Plots the dose rate on a 3D globe for a specific timestamp and altitude.

        Args:
            timestamp: The target timestamp for the globe plot.
            altitude: The altitude (in km) for the globe plot.
            nearest_ts: If True, use the nearest available timestamp if the exact one is not found.
            **plot_kwargs: Additional keyword arguments passed to the underlying DoseRateFrame.plot_on_globe().

        Returns:
            Matplotlib Figure object containing the globe, or None if plotting fails.
        """
        frame = self.get_dose_rate_frame(timestamp, nearest=nearest_ts)
        if frame is None:
            return None # Message already printed

        try:
            # Call the DoseRateFrame's plotting method
            globe_fig = frame.plot_on_globe(altitude=altitude, **plot_kwargs)
            return globe_fig
        except Exception as e:
            print(f"Error plotting globe for timestamp {frame.timestamp}: {e}")
            return None

    # --- Data Export Methods ---

    def export_to_netcdf(self, filename: str):
        """
        Exports the calculated dose rate data for the entire event to a NetCDF file.

        Assumes that the latitude, longitude, and altitude grids are consistent 
        across all DoseRateFrame objects in self.dose_rates.

        Args:
            filename: The path (including filename) for the output NetCDF file.
        """
        if not hasattr(self, 'dose_rates') or not self.dose_rates:
            print("Error: No dose rate data available to export. Run run_AniMAIRE first.")
            return

        print(f"Preparing data for NetCDF export to {filename}...")
        
        # Get sorted timestamps
        timestamps = sorted(self.dose_rates.keys())
        n_times = len(timestamps)
        
        # Get grid information from the first frame (assuming consistency)
        first_frame = self.dose_rates[timestamps[0]]
        try:
            # Infer grid from unique coordinate values
            altitudes = np.unique(first_frame['altitude (km)'].values)
            latitudes = np.unique(first_frame['latitude'].values)
            longitudes = np.unique(first_frame['longitude'].values)
            n_alts = len(altitudes)
            n_lats = len(latitudes)
            n_lons = len(longitudes)
            
            # Check if the number of points matches the inferred grid size
            if len(first_frame) != n_alts * n_lats * n_lons:
                 print("Warning: Number of data points in the first frame does not match inferred grid size (n_alts * n_lats * n_lons).")
                 print(f"Points: {len(first_frame)}, Grid Size: {n_alts}*{n_lats}*{n_lons} = {n_alts * n_lats * n_lons}")
                 print("Data might be sparse or unstructured. Exporting raw structure.")
                 # Fallback or alternative structured export might be needed here
                 # For now, proceed assuming we can reshape or handle it.
                 # A more robust approach might store data as 1D arrays if unstructured.
                 pass # Continue cautiously

        except KeyError as e:
             print(f"Error: Missing expected coordinate column in DoseRateFrame: {e}. Cannot determine grid.")
             return
        except Exception as e:
             print(f"Error inferring grid structure: {e}")
             return

        # Create a 4D NumPy array to hold all dose rate data
        # Dimensions: (time, altitude, latitude, longitude)
        all_dose_rates = np.full((n_times, n_alts, n_lats, n_lons), np.nan, dtype=np.float32)

        # Populate the array
        for i, ts in enumerate(tqdm(timestamps, desc="Structuring data")):
            frame = self.dose_rates[ts]
            # Pivot or reshape the DataFrame into the grid structure
            # This assumes a MultiIndex or similar structure might be useful, or manual reshaping
            try:
                # Simple pivot assuming one value per coord combination
                # This might fail if data is not perfectly structured or has duplicates
                frame_pivot = frame.pivot_table(index=['altitude (km)', 'latitude'], columns='longitude', values='edose')
                # Ensure consistent coordinate order if pivot_table changes it
                frame_pivot = frame_pivot.reindex(index=pd.MultiIndex.from_product([altitudes, latitudes], names=['altitude (km)', 'latitude']), 
                                                columns=longitudes)
                all_dose_rates[i, :, :, :] = frame_pivot.values.reshape((n_alts, n_lats, n_lons))
            except Exception as e:
                print(f"Warning: Could not pivot/reshape data for timestamp {ts}. Error: {e}. Leaving as NaN.")
                # Alternative: Iterate through rows and assign manually (slower)
                # for _, row in frame.iterrows():
                #     alt_idx = np.where(altitudes == row['altitude (km)'])[0][0]
                #     lat_idx = np.where(latitudes == row['latitude'])[0][0]
                #     lon_idx = np.where(longitudes == row['longitude'])[0][0]
                #     all_dose_rates[i, alt_idx, lat_idx, lon_idx] = row['edose']

        # Create and write the NetCDF file
        try:
            with netCDF4.Dataset(filename, 'w', format='NETCDF4') as ncfile:
                print("Writing NetCDF file...")
                # --- Create Dimensions ---
                ncfile.createDimension('time', n_times)
                ncfile.createDimension('altitude', n_alts)
                ncfile.createDimension('latitude', n_lats)
                ncfile.createDimension('longitude', n_lons)

                # --- Create Coordinate Variables ---
                time_var = ncfile.createVariable('time', 'f8', ('time',))
                alt_var = ncfile.createVariable('altitude', 'f4', ('altitude',))
                lat_var = ncfile.createVariable('latitude', 'f4', ('latitude',))
                lon_var = ncfile.createVariable('longitude', 'f4', ('longitude',))
                
                # Coordinate Variable Attributes
                time_var.units = 'seconds since 1970-01-01 00:00:00 UTC'
                time_var.calendar = 'gregorian'
                time_var.standard_name = 'time'
                time_var.long_name = 'Time'
                
                alt_var.units = 'km'
                alt_var.standard_name = 'altitude'
                alt_var.long_name = 'Altitude above mean sea level'
                alt_var.axis = 'Z'
                
                lat_var.units = 'degrees_north'
                lat_var.standard_name = 'latitude'
                lat_var.long_name = 'Latitude'
                lat_var.axis = 'Y'
                
                lon_var.units = 'degrees_east'
                lon_var.standard_name = 'longitude'
                lon_var.long_name = 'Longitude'
                lon_var.axis = 'X'
                
                # Write Coordinate Data
                time_var[:] = [ts.timestamp() for ts in timestamps]
                alt_var[:] = altitudes
                lat_var[:] = latitudes
                lon_var[:] = longitudes
                
                # --- Create Data Variable ---
                edose_var = ncfile.createVariable('effective_dose_rate', 'f4', ('time', 'altitude', 'latitude', 'longitude'), zlib=True, complevel=4, fill_value=np.nan)
                
                # Data Variable Attributes
                edose_var.units = 'uSv/hr'
                edose_var.long_name = 'Effective Dose Rate'
                edose_var.coordinates = 'time altitude latitude longitude'
                # Add provenance/source info if available
                if hasattr(self, 'spectra_file_path'):
                    edose_var.source_spectra_file = self.spectra_file_path
                    
                # Write Data
                edose_var[:,:,:,:] = all_dose_rates

                # --- Global Attributes ---
                ncfile.title = 'AniMAIRE Event Simulation Results'
                ncfile.institution = 'Generated by AniMAIRE' # Or more specific affiliation
                ncfile.source = 'AniMAIRE atmospheric radiation model'
                ncfile.history = f'Created {dt.datetime.utcnow().isoformat()}Z'
                ncfile.Conventions = 'CF-1.6' # Or later version if applicable
                if hasattr(self, 'spectra_file_path'):
                    ncfile.spectra_file = self.spectra_file_path
                ncfile.event_start_time_utc = timestamps[0].isoformat()
                ncfile.event_end_time_utc = timestamps[-1].isoformat()

            print(f"Successfully exported data to {filename}")

        except Exception as e:
            print(f"Error writing NetCDF file: {e}")

# Define a cached version of the run_from_double_power_law_gaussian_distribution function
@memory.cache
def run_animaire_cached(J0, gamma, deltaGamma, sigma_1, sigma_2, B, alpha_prime, 
                         reference_pitch_angle_latitude, reference_pitch_angle_longitude, 
                         date_and_time, use_split_spectrum, **kwargs):
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
        **kwargs
) -> AniMAIRE_event:
    """
    Perform a run to calculate dose rates using a GLE spectrum file.
    """
    GLE_to_run = AniMAIRE_event(GLE_spectrum_file)
    GLE_to_run.run_AniMAIRE(**kwargs)
    return GLE_to_run