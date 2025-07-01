from AniMAIRE.AniMAIRE_event import BaseAniMAIREEvent
from AniMAIRE.DoseRateFrame import DoseRateFrame
from AniMAIRE.MAIREPLUS_event import MAIREPLUS_event, full_NM_set
from extract_monitor_sets import get_monitor_sets_from_directory
from typing import Dict, List, Tuple, Optional, Union, Any, Sequence, TypeVar
import datetime
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass
from functools import lru_cache
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import os
import imageio
from IPython.display import HTML
import matplotlib.animation as animation
import numpy as np

# Type aliases for clarity
# MonitorSet = TypeVar('MonitorSet')  # Replace with actual type if available


class ZeroDoseMAIREPLUSEvent(MAIREPLUS_event):
    """
    A dummy MAIREPLUS_event that returns zero dose rates for all calculations.
    Used when normalization monitor activity is below the minimum threshold.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with the same parameters as MAIREPLUS_event but skip actual initialization."""
        # Store the parameters without calling parent __init__ to avoid computation
        self.params = kwargs
        self.datetime = kwargs.get('datetime', [])
        self.dose_rates = {}
        self._is_zero_dose = True
        
        # Set required attributes from the parameters
        self.neutron_monitor_1_location = kwargs.get('neutron_monitor_1_location')
        self.neutron_monitor_2_location = kwargs.get('neutron_monitor_2_location')
        self.normalisation_monitor_location = kwargs.get('normalisation_monitor_location')
        
    def run_AniMAIRE(self, use_cache: bool = True):
        """Override to create zero-filled dose rate frames without computation."""
        # Create zero dose rate frames for each timestamp
        timestamps = self.datetime if hasattr(self.datetime, '__iter__') else [self.datetime]
        
        for ts in timestamps:
            # Create a minimal dose rate frame with zeros
            # We need to match the structure expected by the interpolation code
            dose_df = pd.DataFrame({
                'latitude': np.arange(-90, 91, 5),  # Basic grid
                'longitude': np.arange(-180, 181, 10),
                'altitude': 12.192,  # Default altitude
                'edose': 0.0,
                'adose': 0.0,
                'dosee': 0.0,
                'tn1': 0.0,
                'SEU': 0.0,
                'SEL': 0.0
            })
            
            # Expand to create full grid
            lats = dose_df['latitude'].unique()
            lons = dose_df['longitude'].unique()
            full_grid = pd.DataFrame(
                [(lat, lon) for lat in lats for lon in lons],
                columns=['latitude', 'longitude']
            )
            full_grid['altitude'] = 12.192
            full_grid['edose'] = 0.0
            full_grid['adose'] = 0.0
            full_grid['dosee'] = 0.0
            full_grid['tn1'] = 0.0
            full_grid['SEU'] = 0.0
            full_grid['SEL'] = 0.0
            
            # Create DoseRateFrame
            dose_frame = DoseRateFrame(full_grid)
            dose_frame.particle_distributions = []  # Empty particle distributions
            
            # Convert numpy datetime64 to Python datetime if needed
            if isinstance(ts, np.datetime64):
                ts = pd.Timestamp(ts).to_pydatetime()
            
            self.dose_rates[ts] = dose_frame
    
    def get_all_timestamps(self):
        """Return all timestamps for this event."""
        return list(self.dose_rates.keys())
    
    def __repr__(self):
        """String representation indicating this is a zero-dose event."""
        return f"ZeroDoseMAIREPLUSEvent(timestamps={len(self.dose_rates)})"


@dataclass
class EventBoundaries:
    """Data class to store event boundary information."""
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    baseline_rate: Optional[float]

    @property
    def is_valid(self) -> bool:
        """Check if all boundary values are valid."""
        return all(x is not None for x in (self.start_time, self.end_time, self.baseline_rate))

class AnisotropicMAIREPLUSevent(BaseAniMAIREEvent):
    """
    Anisotropic AniMAIRE event class for analyzing solar events using neutron monitor data.
    This class provides a comprehensive implementation for analyzing anisotropic solar events
    using neutron monitor data, with built-in spatial interpolation and dose rate calculation.
    """
    EARTH_RADIUS_KM: float = 6371.0  # Earth's radius in kilometers

    def __init__(
        self,
        data_directory_path: str,
        intensity_threshold_percent: float = 8.0,
        max_sets_to_process: int = None,
        timestamps_per_set: int = None,
        reference_station: str = 'OULU',
        min_normalisation_percentage_increase: float = 3.0,
        **kwargs
    ):
        """
        Initialize the AnisotropicAniMAIREevent with analysis parameters.

        Parameters
        ----------
        data_directory_path : str
            Path to the directory containing neutron monitor data.
        intensity_threshold_percent : float, optional
            Percentage threshold for determining event boundaries.
        max_sets_to_process : int, optional
            Maximum number of monitor sets to process.
        timestamps_per_set : int, optional
            Number of timestamps to use from each set.
        reference_station : str, optional
            Name of the reference neutron monitor station.
        min_normalisation_percentage_increase : float, optional
            Minimum percentage increase required for normalization monitor to be used.
            If below this threshold, dose rates will be set to 0.0. Default is 3.0%.
        **kwargs : dict
            Additional keyword arguments for analysis customization.
        """
        super().__init__()
        self.data_directory_path = data_directory_path
        self.intensity_threshold_percent = intensity_threshold_percent
        self.max_sets_to_process = max_sets_to_process
        self.timestamps_per_set = timestamps_per_set
        self.reference_station = reference_station
        self.min_normalisation_percentage_increase = min_normalisation_percentage_increase
        self.run_kwargs = kwargs
        self._monitor_sets: Optional[List[full_NM_set]] = None

    @property
    def monitor_sets(self) -> List[full_NM_set]:
        """Lazy loading of monitor sets."""
        if self._monitor_sets is None:
            self._monitor_sets = get_monitor_sets_from_directory(self.data_directory_path)
        return self._monitor_sets

    @staticmethod
    @lru_cache(maxsize=1024)
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points using the haversine formula.
        This implementation is cached for performance and uses numpy for vectorized operations.

        Parameters
        ----------
        lat1, lon1, lat2, lon2 : float
            Latitude and longitude of the two points in decimal degrees.

        Returns
        -------
        float
            Distance in kilometers between the points.
        """
        # Normalize coordinates
        lat1, lat2 = np.clip([lat1, lat2], -90, 90)
        lon1, lon2 = np.mod([lon1, lon2], 360)

        # Handle longitude wraparound
        if abs(lon1 - lon2) > 180:
            if lon1 > lon2:
                lon2 += 360
            else:
                lon1 += 360

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * AnisotropicMAIREPLUSevent.EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

    def _add_monitor_distance(self, df: pd.DataFrame, monitor_location: Union[Sequence[float], Dict[str, float]]) -> pd.DataFrame:
        """
        Add monitor distance calculations to a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing latitude and longitude columns.
        monitor_location : Union[Sequence[float], Dict[str, float]]
            Monitor location as either [lat, lon] or {'latitude': lat, 'longitude': lon}.

        Returns
        -------
        pd.DataFrame
            DataFrame with added monitor_distance_km column.
        """
        if isinstance(monitor_location, dict):
            mon_lat, mon_lon = monitor_location['latitude'], monitor_location['longitude']
        else:
            mon_lat, mon_lon = monitor_location[0], monitor_location[1]

        df = df.copy()
        df['monitor_distance_km'] = df.apply(
            lambda row: self._haversine_distance(
                row['latitude'], row['longitude'], 
                mon_lat, mon_lon
            ),
            axis=1
        )
        return df

    def _calculate_weights(self, distances: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculate interpolation weights using inverse square distance weighting.

        Parameters
        ----------
        distances : NDArray[np.float64]
            Array of distances to each monitor.

        Returns
        -------
        NDArray[np.float64]
            Normalized weights for each monitor.
        """
        with np.errstate(divide='ignore'):
            weights = 1 / (distances**2)
        weights[~np.isfinite(weights)] = 0
        sum_weights = np.sum(weights,axis=0)
        return weights / sum_weights

    def _interpolate_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Interpolate values between multiple monitor dataframes.

        Parameters
        ----------
        dataframes : List[pd.DataFrame]
            List of dataframes with monitor_distance_km column.

        Returns
        -------
        pd.DataFrame
            Interpolated dataframe.
        """
        if not dataframes:
            raise ValueError("At least one dataframe required for interpolation")

        distances = np.array([df['monitor_distance_km'].values for df in dataframes])
        self.anisotropic_weights = self._calculate_weights(distances)

        # Vectorized weighted sum calculation with NaN values treated as 0.0
        print(self.anisotropic_weights)
        weighted_sum = sum(df.fillna(0.0).multiply(w) for df, w in zip(dataframes, self.anisotropic_weights))
        return weighted_sum.drop(columns=['monitor_distance_km'])

    def identify_solar_event_boundaries(
        self,
        monitor_sets: List[full_NM_set],
        reference_station: str,
        intensity_threshold_percent: float
    ) -> EventBoundaries:
        """
        Identify solar event boundaries using the reference monitor station.

        Parameters
        ----------
        monitor_sets : List[full_NM_set]
            List of neutron monitor sets.
        reference_station : str
            Name of the reference neutron monitor station.
        intensity_threshold_percent : float
            Percentage threshold for determining event boundaries.

        Returns
        -------
        EventBoundaries
            Object containing start time, end time, and baseline rate.

        Raises
        ------
        RuntimeError
            If the reference station is not found in the monitor sets.
        """
        target_set = next(
            (nm_set for nm_set in monitor_sets 
             if nm_set.normalisation.get_station_name() == reference_station),
            None
        )
        if not target_set:
            raise RuntimeError(f"Reference station '{reference_station}' not found in monitor sets. Check your input data and reference station name.")

        start_time = target_set.get_event_start_datetime(
            threshold_percentage=intensity_threshold_percent, use_corrected=False
        )
        end_time = target_set.get_event_end_datetime(
            threshold_percentage=intensity_threshold_percent, use_corrected=False
        )
        baseline_rate = target_set.normalisation.get_baseline_rate()
        print(f"Start time: {start_time}, End time: {end_time}, Baseline rate: {baseline_rate}")

        return EventBoundaries(start_time, end_time, baseline_rate)

    def generate_isotropic_dose_runs(
        self,
        monitor_sets: List[full_NM_set],
        event_boundaries: EventBoundaries,
        max_sets_to_process: int,
        timestamps_per_set: int,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[full_NM_set, MAIREPLUS_event]:
        """
        Generate isotropic dose runs for the event period.

        Parameters
        ----------
        monitor_sets : List[full_NM_set]
            List of neutron monitor sets.
        event_boundaries : EventBoundaries
            Event boundary information.
        max_sets_to_process : int
            Maximum number of monitor sets to process.
        timestamps_per_set : int
            Number of timestamps to use from each set.

        Returns
        -------
        Dict[full_NM_set, MAIREPLUS_event]
            Dictionary mapping monitor sets to their MAIREPLUS models.
        """
        if not event_boundaries.is_valid:
            return {}

        isotropic_dose_runs = {}
        for nm_set in monitor_sets[:max_sets_to_process]:
            filtered_set = nm_set.filter_by_datetime(
                start_time=event_boundaries.start_time,
                end_time=event_boundaries.end_time
            )
            if filtered_set.get_primary().empty or filtered_set.get_secondary().empty:
                continue

            # Check if normalization monitor percentage increase meets minimum threshold
            normalisation_data = filtered_set.get_normalisation().head(timestamps_per_set)
            max_normalisation_percentage = normalisation_data['percentage_increase'].max()
            
            if max_normalisation_percentage < self.min_normalisation_percentage_increase:
                print(f"Using zero dose for monitor set with {nm_set.get_normalisation().get_station_name()} as normalization: "
                      f"max percentage increase ({max_normalisation_percentage:.2f}%) below threshold "
                      f"({self.min_normalisation_percentage_increase:.2f}%)")
                
                # Create a zero-dose event instead of skipping
                MAIREPLUS_events = ZeroDoseMAIREPLUSEvent(
                    neutron_monitor_1_percentage_increase=filtered_set.get_primary().head(timestamps_per_set)['percentage_increase'].values,
                    neutron_monitor_2_percentage_increase=filtered_set.get_secondary().head(timestamps_per_set)['percentage_increase'].values,
                    normalisation_monitor_percentage_increase=normalisation_data['percentage_increase'].values,
                    OULU_gcr_count_rate_in_seconds=event_boundaries.baseline_rate,
                    datetime=filtered_set.get_primary().head(timestamps_per_set)['timestamp'].values,
                    neutron_monitor_1_location=filtered_set.get_primary().get_location(),
                    neutron_monitor_2_location=filtered_set.get_secondary().get_location(),
                    normalisation_monitor_location=filtered_set.get_normalisation().get_location(),
                    **self.run_kwargs
                )
            else:
                # Normal processing for monitors above threshold
                MAIREPLUS_events = MAIREPLUS_event(
                    neutron_monitor_1_percentage_increase=filtered_set.get_primary().head(timestamps_per_set)['percentage_increase'].values,
                    neutron_monitor_2_percentage_increase=filtered_set.get_secondary().head(timestamps_per_set)['percentage_increase'].values,
                    normalisation_monitor_percentage_increase=normalisation_data['percentage_increase'].values,
                    OULU_gcr_count_rate_in_seconds=event_boundaries.baseline_rate,
                    datetime=filtered_set.get_primary().head(timestamps_per_set)['timestamp'].values,
                    neutron_monitor_1_location=filtered_set.get_primary().get_location(),
                    neutron_monitor_2_location=filtered_set.get_secondary().get_location(),
                    normalisation_monitor_location=filtered_set.get_normalisation().get_location(),
                    **self.run_kwargs
                )
            
            MAIREPLUS_events.run_AniMAIRE(use_cache=use_cache)
            isotropic_dose_runs[nm_set] = MAIREPLUS_events

        return isotropic_dose_runs

    def create_interpolated_dose_map(
        self,
        events: Union[List[MAIREPLUS_event], MAIREPLUS_event],
        timestamp: datetime.datetime
    ) -> DoseRateFrame:
        """
        Create an interpolated dose map for a specific timestamp.

        Parameters
        ----------
        events : Union[List[MAIREPLUS_event], MAIREPLUS_event]
            Event(s) containing dose rates.
        timestamp : datetime.datetime
            Target timestamp.

        Returns
        -------
        DoseRateFrame
            Interpolated dose rate frame.
        """
        if not isinstance(events, (list, tuple)):
            events = [events]

        dose_frames = []
        for event in events:
            dose_frame = event.dose_rates[timestamp]
            dose_frame = dose_frame.fillna(0.0)
            try:
                norm_location = event.nm_set.get_normalisation().get_location()
            except AttributeError:
                norm_location = event.run_kwargs['normalisation_monitor_location']
            
            distance_df = self._add_monitor_distance(dose_frame, norm_location)
            dose_frames.append(distance_df)

        return self._interpolate_dataframes(dose_frames)

    def create_global_dose_maps(
        self,
        isotropic_dose_runs: Union[List[MAIREPLUS_event], Dict[full_NM_set, MAIREPLUS_event]],
        event_timestamps: List[datetime.datetime]
    ) -> Dict[datetime.datetime, DoseRateFrame]:
        """
        Create interpolated dose maps for all timestamps.

        Parameters
        ----------
        isotropic_dose_runs : Union[List[MAIREPLUS_event], Dict[full_NM_set, MAIREPLUS_event]]
            Isotropic dose runs to use for interpolation.
        event_timestamps : List[datetime.datetime]
            List of timestamps to process.

        Returns
        -------
        Dict[datetime.datetime, DoseRateFrame]
            Mapping of timestamps to dose rate frames.
        """
        isotropic_dose_list = list(isotropic_dose_runs.values()) if isinstance(isotropic_dose_runs, dict) else isotropic_dose_runs
        return {
            timestamp: self.create_interpolated_dose_map(isotropic_dose_list, timestamp)
            for timestamp in event_timestamps
        }

    def run_AniMAIRE(self, use_cache: bool = True, monitor_set_slice=None) -> Dict[datetime.datetime, DoseRateFrame]:
        """
        Run the complete AniMAIRE analysis pipeline.

        Parameters
        ----------
        use_cache : bool, optional
            Whether to use caching for MAIREPLUS event calculations. Default is True.
        monitor_set_slice : slice, list, or None, optional
            Slice or list of indices to select which monitor sets to run for. Default is None (all).

        Returns
        -------
        Dict[datetime.datetime, DoseRateFrame]
            Dictionary mapping timestamps to dose rate frames.

        Raises
        ------
        RuntimeError
            If event boundaries are invalid, no dose runs are generated, or no timestamps are found.
        """
        monitor_sets = self.monitor_sets
        max_sets_to_process = self.max_sets_to_process
        # Apply slicing if requested
        if monitor_set_slice is not None:
            if isinstance(monitor_set_slice, slice):
                monitor_sets = monitor_sets[monitor_set_slice]
            elif isinstance(monitor_set_slice, (list, tuple, np.ndarray)):
                monitor_sets = [monitor_sets[i] for i in monitor_set_slice]
            else:
                raise ValueError("monitor_set_slice must be a slice, list, tuple, or None.")
            max_sets_to_process = len(monitor_sets)

        event_boundaries = self.identify_solar_event_boundaries(
            monitor_sets=self.monitor_sets,
            reference_station=self.reference_station,
            intensity_threshold_percent=self.intensity_threshold_percent
        )
        if not event_boundaries.is_valid:
            raise RuntimeError("Event boundaries could not be identified. Check your input data and parameters.")

        self.isotropic_dose_runs = self.generate_isotropic_dose_runs(
            monitor_sets=monitor_sets,
            event_boundaries=event_boundaries,
            max_sets_to_process=max_sets_to_process,
            timestamps_per_set=self.timestamps_per_set,
            use_cache=use_cache
        )
        if not self.isotropic_dose_runs:
            raise RuntimeError("No isotropic dose runs could be generated. Check your monitor sets and event boundaries.")

        first_run = next(iter(self.isotropic_dose_runs.values()), None)
        if first_run is None or not hasattr(first_run, 'get_all_timestamps'):
            raise RuntimeError("No valid MAIREPLUS_event runs found in isotropic dose runs.")
        event_timestamps = list(first_run.get_all_timestamps())
        if not event_timestamps:
            raise RuntimeError("No timestamps found in the MAIREPLUS_event runs. Check your input data.")

        self.dose_rates = self.create_global_dose_maps(self.isotropic_dose_runs, event_timestamps)
        return self.dose_rates

    def get_processed_monitor_sets(self) -> List[Dict[str, str]]:
        """
        Get information about all neutron monitor sets that were successfully processed.
        This method should be called after run_AniMAIRE has been executed.

        Returns
        -------
        List[Dict[str, str]]
            List of dictionaries containing information about each processed monitor set.
            Each dictionary contains:
            - 'primary': Name of the primary monitor
            - 'secondary': Name of the secondary monitor
            - 'normalisation': Name of the normalisation monitor

        Raises
        ------
        RuntimeError
            If called before run_AniMAIRE or if no monitor sets were processed.
        """
        if not hasattr(self, 'isotropic_dose_runs') or not self.isotropic_dose_runs:
            raise RuntimeError(
                "No processed monitor sets found. "
                "Make sure to run run_AniMAIRE() first."
            )

        monitor_info = []
        for nm_set in self.isotropic_dose_runs.keys():
            try:
                info = {
                    'primary': nm_set.get_primary().get_station_name(),
                    'secondary': nm_set.get_secondary().get_station_name(),
                    'normalisation': nm_set.get_normalisation().get_station_name()
                }
                monitor_info.append(info)
            except AttributeError as e:
                print(f"Warning: Could not get complete information for a monitor set: {e}")
                continue

        if not monitor_info:
            raise RuntimeError("No valid monitor set information could be retrieved.")

        return monitor_info

    def print_processed_monitor_sets(self) -> None:
        """
        Print information about all neutron monitor sets that were successfully processed in a nicely formatted way.
        This method should be called after run_AniMAIRE has been executed.

        Raises
        ------
        RuntimeError
            If called before run_AniMAIRE or if no monitor sets were processed.
        """
        monitor_info = self.get_processed_monitor_sets()
        
        print(f"Successfully processed {len(monitor_info)} monitor sets:")
        print("-" * 60)
        for i, info in enumerate(monitor_info, 1):
            print(f"{i:2d}. Primary: {info['primary']:<12} | "
                  f"Secondary: {info['secondary']:<12} | "
                  f"Norm: {info['normalisation']}")
        print("-" * 60)

    def plot_isotropic_gle_spectra(self, timestamp, ax=None, min_rigidity=0.1, max_rigidity=20, 
                                   show_legend=True, alpha=0.7, spectrum_type='both', show_monitors=True, **kwargs):
        """
        Plot all isotropic spectra that were used to generate the anisotropic event for a particular timestamp.
        
        This method plots the rigidity spectra from all the MAIREPLUS_event objects (isotropic dose runs)
        that were generated for the specified timestamp. The method can distinguish between GLE 
        (Ground Level Enhancement) and GCR (Galactic Cosmic Ray) spectra by examining the spectrum 
        types: MAIREPLUS_spectrum instances are classified as GLE, while DLRmodelSpectrum instances 
        are classified as GCR.
        
        Parameters
        ----------
        timestamp : datetime.datetime
            The timestamp for which to plot the spectra
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created
        min_rigidity : float, optional
            Minimum rigidity in GV for spectrum plot (default: 0.1)
        max_rigidity : float, optional
            Maximum rigidity in GV for spectrum plot (default: 20)
        show_legend : bool, optional
            Whether to show a legend with monitor set names (default: True)
        alpha : float, optional
            Transparency level for the plot lines (default: 0.7)
        spectrum_type : str, optional
            Which type of spectra to plot: 'gle', 'gcr', or 'both' (default: 'both')
            - 'gle': Plot only GLE (Ground Level Enhancement) spectra
            - 'gcr': Plot only GCR (Galactic Cosmic Ray) spectra  
            - 'both': Plot both GLE and GCR spectra
        show_monitors : bool, optional
            Whether to display neutron monitor locations (default: True)
        **kwargs : additional keyword arguments
            Passed to the underlying rigiditySpectrum.plot() method
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot
            
        Raises
        ------
        RuntimeError
            If called before run_AniMAIRE or if no isotropic dose runs are available
        ValueError
            If the specified timestamp is not found in the dose runs
        """
        # Validate spectrum_type parameter
        valid_spectrum_types = {'gle', 'gcr', 'both'}
        if spectrum_type not in valid_spectrum_types:
            raise ValueError(f"spectrum_type must be one of {valid_spectrum_types}, got '{spectrum_type}'")
        
        # Check if isotropic dose runs are available
        if not hasattr(self, 'isotropic_dose_runs') or not self.isotropic_dose_runs:
            raise RuntimeError(
                "No isotropic dose runs found. "
                "Make sure to run run_AniMAIRE() first."
            )
        
        # Validate timestamp
        available_timestamps = []
        for maire_event in self.isotropic_dose_runs.values():
            available_timestamps.extend(maire_event.get_all_timestamps())
        
        if timestamp not in available_timestamps:
            available_str = ", ".join([str(ts) for ts in sorted(set(available_timestamps))])
            raise ValueError(
                f"Timestamp {timestamp} not found in isotropic dose runs. "
                f"Available timestamps: {available_str}"
            )
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        
        plotted_count = 0
        
        # Plot spectra from each monitor set
        for nm_set, maire_event in self.isotropic_dose_runs.items():
            # Check if this MAIREPLUS_event has the requested timestamp
            if timestamp in maire_event.dose_rates:
                dose_rate_frame = maire_event.dose_rates[timestamp]
                
                # Get monitor set name for legend
                try:
                    primary_name = nm_set.get_primary().get_station_name()
                    secondary_name = nm_set.get_secondary().get_station_name()
                    norm_name = nm_set.get_normalisation().get_station_name()
                    monitor_label = f"{primary_name}-{secondary_name} (norm: {norm_name})"
                except (AttributeError, Exception):
                    monitor_label = f"Monitor set {plotted_count + 1}"
                
                # Determine which spectra to plot based on spectrum types
                spectra_to_plot = []
                
                if hasattr(dose_rate_frame, 'particle_distributions') and dose_rate_frame.particle_distributions:
                    for i, particle_dist in enumerate(dose_rate_frame.particle_distributions):
                        if hasattr(particle_dist, 'momentum_distribution') and \
                           hasattr(particle_dist.momentum_distribution, 'rigidity_spectrum'):
                            
                            # Check the type of rigidity spectrum to determine if it's GLE or GCR
                            rigidity_spectrum = particle_dist.momentum_distribution.rigidity_spectrum
                            
                            # Import the spectrum classes for type checking
                            from AniMAIRE.MAIREPLUS_spectrum import MAIREPLUS_spectrum
                            from AniMAIRE.anisotropic_MAIRE_engine.spectralCalculations.rigiditySpectrum import DLRmodelSpectrum, PowerLawSpectrum
                            
                            spectrum_type_detected = None
                            if isinstance(rigidity_spectrum, MAIREPLUS_spectrum):
                                spectrum_type_detected = 'GLE'
                            elif isinstance(rigidity_spectrum, DLRmodelSpectrum):
                                spectrum_type_detected = 'GCR'
                            elif isinstance(rigidity_spectrum, PowerLawSpectrum):
                                # PowerLawSpectrum could be either, but if it's not MAIREPLUS_spectrum specifically,
                                # it's likely a GLE spectrum (MAIREPLUS inherits from PowerLawSpectrum)
                                spectrum_type_detected = 'GLE'
                            else:
                                # Unknown spectrum type - skip
                                continue
                            
                            # Only include this spectrum if it matches the requested type
                            if spectrum_type == 'both' or \
                               (spectrum_type == 'gle' and spectrum_type_detected == 'GLE') or \
                               (spectrum_type == 'gcr' and spectrum_type_detected == 'GCR'):
                                spectra_to_plot.append((spectrum_type_detected, particle_dist, i))
                
                # Plot each selected spectrum type
                for spectrum_type_label, particle_dist, particle_index in spectra_to_plot:
                    # Create label for this spectrum
                    particle_z = particle_dist.particle_species.atomicNumber
                    
                    # Count different spectrum types being plotted
                    spectrum_types_in_plot = set(item[0] for item in spectra_to_plot)
                    
                    if len(spectrum_types_in_plot) > 1:
                        # Multiple spectrum types - include type in label
                        if len(dose_rate_frame.particle_distributions) > 1:
                            spectrum_label = f"{monitor_label} {spectrum_type_label} (Z={particle_z})"
                        else:
                            spectrum_label = f"{monitor_label} {spectrum_type_label}"
                    else:
                        # Single spectrum type - use original labeling
                        if len(dose_rate_frame.particle_distributions) > 1:
                            spectrum_label = f"{monitor_label} (Z={particle_z})"
                        else:
                            spectrum_label = monitor_label
                    
                    # Plot the spectrum
                    particle_dist.momentum_distribution.rigidity_spectrum.plot(
                        ax=ax, 
                        min_rigidity=min_rigidity, 
                        max_rigidity=max_rigidity, 
                        title=None,
                        alpha=alpha,
                        **kwargs
                    )
                    
                    # Set the label for the most recent line
                    if show_legend:
                        ax.lines[-1].set_label(spectrum_label)
                
                # Only increment plotted_count if we actually plotted something
                if spectra_to_plot:
                    plotted_count += 1
        
        # Customize the plot
        title_map = {
            'gle': 'Isotropic GLE Spectra',
            'gcr': 'Isotropic GCR Spectra', 
            'both': 'Isotropic GLE + GCR Spectra'
        }
        ax.set_title(f'{title_map[spectrum_type]} at {timestamp.strftime("%Y-%m-%d %H:%M")}')
        ax.set_xlabel('Rigidity (GV)')
        ax.set_ylabel('Flux (particles/m²/sr/s/GV)')
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.set_xlim(min_rigidity, max_rigidity)
        
        # Add legend if requested and we have labels
        if show_legend and plotted_count > 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     fontsize='small', framealpha=0.9)
        
        # Add text annotation with count
        ax.text(0.02, 0.98, f'{plotted_count} monitor sets plotted', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                fontsize='small')
        
        # Add monitor locations if requested
        if show_monitors:
            self._plot_monitor_locations(ax, is_3d_plot=False)
        
        plt.tight_layout()
        return ax

    def _plot_monitor_locations(self, ax, is_3d_plot: bool = False, scatter_size: int = 30) -> None:
        """
        Plot neutron monitor locations on a given matplotlib axes.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        is_3d_plot : bool, optional
            Whether the plot is a 3D globe plot (using cartopy). Default is False.
        scatter_size : int, optional
            Size of the scatter points. Default is 30.
        """
        monitor_df = self._get_monitor_locations()
        if monitor_df is None or monitor_df.empty:
            return

        # Define colors and markers for different monitor types
        monitor_styles = {
            'primary': {'color': 'red', 'marker': 'o', 'label': 'Primary Monitor'},
            'secondary': {'color': 'blue', 'marker': 's', 'label': 'Secondary Monitor'},
            'normalisation': {'color': 'green', 'marker': '^', 'label': 'Normalisation Monitor'}
        }
        
        transform = None if not is_3d_plot else ccrs.PlateCarree()
        
        # First plot the connecting lines between monitor pairs
        if hasattr(self, 'isotropic_dose_runs') and self.isotropic_dose_runs:
            for nm_set in self.isotropic_dose_runs.keys():
                try:
                    # Get primary monitor location
                    primary_loc = nm_set.get_primary().get_location()
                    secondary_loc = nm_set.get_secondary().get_location()
                    
                    # Draw dotted line between primary and secondary monitors
                    line_kwargs = {
                        'color': 'black',
                        'linestyle': ':',
                        'linewidth': 1.5,
                        'alpha': 0.8,
                        'zorder': 99  # Just below the monitor points
                    }
                    if transform:
                        line_kwargs['transform'] = transform
                    
                    ax.plot([primary_loc[1], secondary_loc[1]], 
                           [primary_loc[0], secondary_loc[0]], 
                           **line_kwargs)
                except Exception as e:
                    print(f"Warning: Could not plot monitor pair line: {e}")
        
        # Plot each type of monitor with different styling
        for monitor_type, style in monitor_styles.items():
            type_df = monitor_df[monitor_df['type'] == monitor_type]
            if not type_df.empty:
                scatter_kwargs = {
                    'c': style['color'],
                    'marker': style['marker'],
                    's': scatter_size,
                    'edgecolors': 'black',
                    'linewidths': 1,
                    'zorder': 100,
                    'label': style['label']
                }
                if transform:
                    scatter_kwargs['transform'] = transform
                
                ax.scatter(
                    type_df['longitude'], type_df['latitude'],
                    **scatter_kwargs
                )
                
                # Add monitor names with appropriate styling
                for _, row in type_df.iterrows():
                    text_kwargs = {
                        'color': 'black',
                        'fontsize': 8,
                        'ha': 'center',
                        'va': 'bottom',
                        'zorder': 101
                    }
                    if transform:
                        text_kwargs['transform'] = transform
                        text_kwargs['fontweight'] = 'bold'
                    
                    ax.text(
                        row['longitude'], row['latitude'], row['name'],
                        **text_kwargs
                    )
        
        # Update legend to include monitor markers
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='lower left')

    def plot_map_at_time(self, timestamp: datetime.datetime, altitude: Optional[float] = None, ax: Optional[plt.Axes] = None,
                        nearest_ts: bool = True, show_monitors: bool = True, **plot_kwargs: Any) -> Optional[plt.Axes]:
        """
        Plot a 2D dose map at a given timestamp and altitude.
        
        Args:
            timestamp (dt.datetime): Timestamp to plot
            altitude (Optional[float], optional): Altitude in kilometers. If None, uses 12.192 km (40,000 ft) if available.
            ax (Optional[plt.Axes], optional): Matplotlib axes to use for plotting. Defaults to None.
            nearest_ts (bool, optional): If True, use nearest timestamp when exact not found. Defaults to True.
            show_monitors (bool, optional): Whether to display neutron monitor locations. Defaults to True.
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
        
        # Get plot result
        ax = frame.plot_dose_map(altitude=best_altitude, **plot_kwargs)
        
        # Add monitor locations if requested
        if show_monitors and ax:
            self._plot_monitor_locations(ax, is_3d_plot=False)
        
        return ax

    def plot_globe_at_time(self, timestamp: datetime.datetime, altitude: Optional[float] = None, 
                          nearest_ts: bool = True, show_monitors: bool = True, **plot_kwargs: Any) -> Optional[plt.Figure]:
        """
        Plot a 3D globe dose at a given timestamp and altitude.
        
        Args:
            timestamp (dt.datetime): Timestamp to plot
            altitude (Optional[float], optional): Altitude in kilometers. If None, uses 12.192 km (40,000 ft) if available.
            nearest_ts (bool, optional): If True, use nearest timestamp when exact not found. Defaults to True.
            show_monitors (bool, optional): Whether to display neutron monitor locations. Defaults to True.
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
        
        # Create plot
        fig = frame.plot_on_globe(altitude=best_altitude, **plot_kwargs)
        
        # Add monitor locations if requested
        if show_monitors and fig is not None:
            # Get the first axes from the figure (assuming it's the main plot)
            ax = fig.axes[0]
            self._plot_monitor_locations(ax, is_3d_plot=True)
            
        return fig 

    def plot_integrated_dose_map(self, altitude: Optional[float] = None, dose_type: str = 'edose', 
                              show_monitors: bool = True, **plot_kwargs: Any) -> Optional[plt.Axes]:
        """
        Plot integrated dose map using native grid.
        
        Args:
            altitude (Optional[float], optional): Altitude in kilometers. If None, uses 12.192 km (40,000 ft) if available.
            dose_type (str, optional): Dose rate type to integrate. Defaults to 'edose'.
            show_monitors (bool, optional): Whether to display neutron monitor locations. Defaults to True.
            **plot_kwargs: Additional keyword arguments passed to plot_dose_map function
            
        Returns:
            Optional[plt.Axes]: Matplotlib axes object with the plot, or None if no data available
        """
        # Call parent class method to get the base plot
        ax = super().plot_integrated_dose_map(altitude, dose_type, show_monitors=False, **plot_kwargs)
        
        # Add monitor locations if requested and plot was successful
        if show_monitors and ax:
            self._plot_monitor_locations(ax, is_3d_plot=False)
            
        return ax

    def plot_peak_dose_rate_map(self, altitude: Optional[float] = None, dose_type: str = 'edose', 
                               show_monitors: bool = True, **plot_kwargs: Any) -> Optional[plt.Axes]:
        """
        Plot peak dose rate map using native grid.
        
        Args:
            altitude (Optional[float], optional): Altitude in kilometers. If None, uses 12.192 km (40,000 ft) if available.
            dose_type (str, optional): Dose rate type to analyze. Defaults to 'edose'.
            show_monitors (bool, optional): Whether to display neutron monitor locations. Defaults to True.
            **plot_kwargs: Additional keyword arguments passed to plot_dose_map function
            
        Returns:
            Optional[plt.Axes]: Matplotlib axes object with the plot, or None if no data available
        """
        # Call parent class method to get the base plot
        ax = super().plot_peak_dose_rate_map(altitude, dose_type, show_monitors=False, **plot_kwargs)
        
        # Add monitor locations if requested and plot was successful
        if show_monitors and ax:
            self._plot_monitor_locations(ax, is_3d_plot=False)
            
        return ax 

    def plot_integrated_dose_globe(self, altitude: Optional[float] = None, dose_type: str = 'edose',
                                  show_monitors: bool = True, **plot_kwargs: Any) -> Optional[plt.Figure]:
        """
        Plot integrated dose on 3D globe.
        
        Args:
            altitude (Optional[float], optional): Altitude in kilometers. If None, uses 12.192 km (40,000 ft) if available.
            dose_type (str, optional): Dose rate type to integrate. Defaults to 'edose'.
            show_monitors (bool, optional): Whether to display neutron monitor locations. Defaults to True.
            **plot_kwargs: Additional keyword arguments passed to plot_on_spherical_globe function
            
        Returns:
            Optional[plt.Figure]: Matplotlib figure object with the plot, or None if no data available
        """
        # Call parent class method to get the base plot
        fig = super().plot_integrated_dose_globe(altitude, dose_type, show_monitors=False, **plot_kwargs)
        
        # Add monitor locations if requested and plot was successful
        if show_monitors and fig is not None:
            # Get the first axes from the figure (assuming it's the main plot)
            ax = fig.axes[0]
            self._plot_monitor_locations(ax, is_3d_plot=True)
            
        return fig

    def plot_peak_dose_rate_globe(self, altitude: Optional[float] = None, dose_type: str = 'edose',
                                  show_monitors: bool = True, **plot_kwargs: Any) -> Optional[plt.Figure]:
        """
        Plot peak dose rate on 3D globe.
        
        Args:
            altitude (Optional[float], optional): Altitude in kilometers. If None, uses 12.192 km (40,000 ft) if available.
            dose_type (str, optional): Dose rate type to analyze. Defaults to 'edose'.
            show_monitors (bool, optional): Whether to display neutron monitor locations. Defaults to True.
            **plot_kwargs: Additional keyword arguments passed to plot_on_spherical_globe function
            
        Returns:
            Optional[plt.Figure]: Matplotlib figure object with the plot, or None if no data available
        """
        # Call parent class method to get the base plot
        fig = super().plot_peak_dose_rate_globe(altitude, dose_type, show_monitors=False, **plot_kwargs)
        
        # Add monitor locations if requested and plot was successful
        if show_monitors and fig is not None:
            # Get the first axes from the figure (assuming it's the main plot)
            ax = fig.axes[0]
            self._plot_monitor_locations(ax, is_3d_plot=True)
            
        return fig 

    def plot_timeseries_at_location(self, latitude: float, longitude: float, altitude: float, 
                                   dose_type: str = 'edose', ax: Optional[plt.Axes] = None,
                                   nearest_ts: bool = True, interpolation_method: str = 'linear', 
                                   **plot_kwargs: Any) -> Optional[plt.Axes]:
        """
        Plot time series of dose at a specific location, showing GLE and GCR components.
        
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
        times = sorted(self.dose_rates.keys())
        total_vals = []
        gcr_vals = []
        gle_vals = []
        tlist = []
        
        for ts in times:
            frame = self.get_dose_rate_frame(ts, nearest_ts)
            if frame is None:
                continue
                
            # Get the total dose rate
            total_d = self.get_dose_rate_at_location(latitude, longitude, altitude, ts, dose_type, nearest_ts, interpolation_method)
            
            if total_d is not None:
                # Get GCR and GLE components from particle distributions
                gcr_d = 0.0
                gle_d = 0.0
                
                if hasattr(frame, 'particle_distributions') and frame.particle_distributions:
                    for particle_dist in frame.particle_distributions:
                        if hasattr(particle_dist, 'momentum_distribution') and \
                           hasattr(particle_dist.momentum_distribution, 'rigidity_spectrum'):
                            
                            # Import the spectrum classes for type checking
                            from ..MAIREPLUS_spectrum import MAIREPLUS_spectrum
                            from ..anisotropic_MAIRE_engine.spectralCalculations.rigiditySpectrum import DLRmodelSpectrum
                            
                            rigidity_spectrum = particle_dist.momentum_distribution.rigidity_spectrum
                            
                            if isinstance(rigidity_spectrum, DLRmodelSpectrum):
                                gcr_d = total_d * 0.5  # Approximate GCR component
                            elif isinstance(rigidity_spectrum, MAIREPLUS_spectrum):
                                gle_d = total_d * 0.5  # Approximate GLE component
                
                total_vals.append(total_d)
                gcr_vals.append(gcr_d)
                gle_vals.append(gle_d)
                tlist.append(ts)
        
        if not total_vals:
            return None
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot components
        ax.plot(tlist, total_vals, label='Total', color='black', marker='o', linewidth=2)
        ax.plot(tlist, gcr_vals, label='GCR Component', color='blue', marker='s', linestyle='--')
        ax.plot(tlist, gle_vals, label='GLE Component', color='red', marker='^', linestyle='--')
        
        ax.set_xlabel('Time (UTC)')
        ax.set_ylabel(f'{dose_type} (uSv/hr)')
        ax.legend()
        ax.grid(True)
        plt.gcf().autofmt_xdate()
        
        return ax 