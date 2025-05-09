from AniMAIRE.AniMAIRE_event import BaseAniMAIREEvent
from AniMAIRE.MAIREPLUS_event import MAIREPLUS_event
from extract_monitor_sets import get_monitor_sets_from_directory
from typing import Dict, List, Tuple, Optional, Union, Any, Sequence, TypeVar
import datetime
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass
from functools import lru_cache

# Type aliases for clarity
DoseRateFrame = TypeVar('DoseRateFrame', bound=pd.DataFrame)
MonitorSet = TypeVar('MonitorSet')  # Replace with actual type if available

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
        max_sets_to_process: int = 4,
        timestamps_per_set: int = 2,
        reference_station: str = 'OULU',
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
        **kwargs : dict
            Additional keyword arguments for analysis customization.
        """
        super().__init__()
        self.data_directory_path = data_directory_path
        self.intensity_threshold_percent = intensity_threshold_percent
        self.max_sets_to_process = max_sets_to_process
        self.timestamps_per_set = timestamps_per_set
        self.reference_station = reference_station
        self.analysis_kwargs = kwargs
        self._monitor_sets: Optional[List[MonitorSet]] = None

    @property
    def monitor_sets(self) -> List[MonitorSet]:
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
            weights = 1 / (distances ** 2)
        weights[~np.isfinite(weights)] = 0
        sum_weights = np.sum(weights)
        return weights / sum_weights if sum_weights > 0 else weights

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
        weights = self._calculate_weights(distances)

        # Vectorized weighted sum calculation with NaN values treated as 0.0
        weighted_sum = sum(df.fillna(0.0).multiply(w) for df, w in zip(dataframes, weights))
        return weighted_sum.drop(columns=['monitor_distance_km'])

    def identify_solar_event_boundaries(
        self,
        monitor_sets: List[MonitorSet],
        reference_station: str,
        intensity_threshold_percent: float
    ) -> EventBoundaries:
        """
        Identify solar event boundaries using the reference monitor station.

        Parameters
        ----------
        monitor_sets : List[MonitorSet]
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
        monitor_sets: List[MonitorSet],
        event_boundaries: EventBoundaries,
        max_sets_to_process: int,
        timestamps_per_set: int,
        use_cache: bool = True
    ) -> Dict[MonitorSet, MAIREPLUS_event]:
        """
        Generate isotropic dose runs for the event period.

        Parameters
        ----------
        monitor_sets : List[MonitorSet]
            List of neutron monitor sets.
        event_boundaries : EventBoundaries
            Event boundary information.
        max_sets_to_process : int
            Maximum number of monitor sets to process.
        timestamps_per_set : int
            Number of timestamps to use from each set.

        Returns
        -------
        Dict[MonitorSet, MAIREPLUS_event]
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

            MAIREPLUS_events = MAIREPLUS_event(
                neutron_monitor_1_percentage_increase=filtered_set.get_primary().head(timestamps_per_set)['percentage_increase'].values,
                neutron_monitor_2_percentage_increase=filtered_set.get_secondary().head(timestamps_per_set)['percentage_increase'].values,
                normalisation_monitor_percentage_increase=filtered_set.get_normalisation().head(timestamps_per_set)['percentage_increase'].values,
                OULU_gcr_count_rate_in_seconds=event_boundaries.baseline_rate,
                datetime=filtered_set.get_primary().head(timestamps_per_set)['timestamp'].values,
                neutron_monitor_1_location=filtered_set.get_primary().get_location(),
                neutron_monitor_2_location=filtered_set.get_secondary().get_location(),
                normalisation_monitor_location=filtered_set.get_normalisation().get_location(),
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
            try:
                norm_location = event.nm_set.get_normalisation().get_location()
            except AttributeError:
                norm_location = event.run_kwargs['normalisation_monitor_location']
            
            distance_df = self._add_monitor_distance(dose_frame, norm_location)
            dose_frames.append(distance_df)

        return self._interpolate_dataframes(dose_frames)

    def create_global_dose_maps(
        self,
        isotropic_dose_runs: Union[List[MAIREPLUS_event], Dict[MonitorSet, MAIREPLUS_event]],
        event_timestamps: List[datetime.datetime]
    ) -> Dict[datetime.datetime, DoseRateFrame]:
        """
        Create interpolated dose maps for all timestamps.

        Parameters
        ----------
        isotropic_dose_runs : Union[List[MAIREPLUS_event], Dict[MonitorSet, MAIREPLUS_event]]
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

    def run_AniMAIRE(self, use_cache: bool = True) -> Dict[datetime.datetime, DoseRateFrame]:
        """
        Run the complete AniMAIRE analysis pipeline.

        Parameters
        ----------
        use_cache : bool, optional
            Whether to use caching for MAIREPLUS event calculations. Default is True.

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
        event_boundaries = self.identify_solar_event_boundaries(
            monitor_sets=monitor_sets,
            reference_station=self.reference_station,
            intensity_threshold_percent=self.intensity_threshold_percent
        )
        if not event_boundaries.is_valid:
            raise RuntimeError("Event boundaries could not be identified. Check your input data and parameters.")

        self.isotropic_dose_runs = self.generate_isotropic_dose_runs(
            monitor_sets=monitor_sets,
            event_boundaries=event_boundaries,
            max_sets_to_process=self.max_sets_to_process,
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
        Print a formatted summary of all neutron monitor sets that were successfully processed.
        This method should be called after run_AniMAIRE has been executed.

        Raises
        ------
        RuntimeError
            If called before run_AniMAIRE or if no monitor sets were processed.
        """
        monitor_info = self.get_processed_monitor_sets()
        
        print("\nProcessed Neutron Monitor Sets:")
        print("-" * 60)
        for i, info in enumerate(monitor_info, 1):
            print(f"Set {i}:")
            print(f"  Primary Monitor:      {info['primary']}")
            print(f"  Secondary Monitor:    {info['secondary']}")
            print(f"  Normalisation Monitor: {info['normalisation']}")
            print("-" * 60) 