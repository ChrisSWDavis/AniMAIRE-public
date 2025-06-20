import re
import copy
from AniMAIRE.AniMAIRE import run_maireplus_spectrum
from AniMAIRE.AniMAIRE_event import BaseAniMAIREEvent, memory
from AniMAIRE.DoseRateFrame import DoseRateFrame


import numpy as np
import pandas as pd


import datetime as dt
from typing import Any, Dict, Optional, Sequence, Union, List

# Add a subclass for MAIREPLUS spectrum based events
# Define a cached version of the MAIREPLUS run function
@memory.cache
def run_maireplus_cached(
        neutron_monitor_1_percentage_increase,
        neutron_monitor_2_percentage_increase,
        normalisation_monitor_percentage_increase,
        OULU_gcr_count_rate_in_seconds,
        datetime,
        kp_index=None,
        neutron_monitor_1_location=(65.0, 25.0, 0.0),
        neutron_monitor_2_location=(50.0, 5.0, 0.0),
        normalisation_monitor_location=(65.0, 25.0, 0.0),
        use_fast_calculation=True,
        **kwargs
):
    """Cached version of run_maireplus_spectrum."""
    return run_maireplus_spectrum(
        neutron_monitor_1_percentage_increase=neutron_monitor_1_percentage_increase,
        neutron_monitor_2_percentage_increase=neutron_monitor_2_percentage_increase,
        normalisation_monitor_percentage_increase=normalisation_monitor_percentage_increase,
        OULU_gcr_count_rate_in_seconds=OULU_gcr_count_rate_in_seconds,
        datetime=datetime,
        kp_index=kp_index,
        neutron_monitor_1_location=neutron_monitor_1_location,
        neutron_monitor_2_location=neutron_monitor_2_location,
        normalisation_monitor_location=normalisation_monitor_location,
        use_fast_calculation=use_fast_calculation,
        **kwargs
    )

class NeutronMonitorData(pd.DataFrame):
    """
    A class for handling neutron monitor data, inheriting from pandas DataFrame.
    Provides methods for parsing and analyzing neutron monitor data files.
    """
    
    @property
    def _constructor(self):
        """
        Return the constructor to use for creating new instances of this class.
        This ensures that operations like filtering return a NeutronMonitorData object.
        """
        return NeutronMonitorData
    
    @classmethod
    def extract_station_info(cls, lines):
        """
        Extract station information from the header lines of a neutron monitor data file.
        
        Parameters:
        lines (list): List of lines from the neutron monitor data file
        
        Returns:
        dict: Dictionary containing station information
        """
        station_info = {}
        
        for line in lines[:20]:  # Check first few lines for station info
            parts = line.split()
            if len(parts) < 3:
                continue
                
            if 'LATITUDE' in line:
                try:
                    idx = parts.index('LATITUDE') + 1
                    station_info['latitude'] = float(parts[idx])
                except (ValueError, IndexError):
                    station_info['latitude'] = np.nan
                    
            if 'LONGITUDE' in line:
                try:
                    idx = parts.index('LONGITUDE') + 1
                    longitude = float(parts[idx])
                    # Convert to longitude east if it's in west (negative values)
                    if longitude < 0:
                        longitude = 360 + longitude
                    station_info['longitude'] = longitude
                except (ValueError, IndexError):
                    station_info['longitude'] = np.nan
                    
            if 'ALTITUDE' in line:
                try:
                    idx = parts.index('ALTITUDE') + 1
                    # Convert altitude from meters to kilometers
                    station_info['altitude'] = float(parts[idx]) / 1000.0
                except (ValueError, IndexError):
                    station_info['altitude'] = np.nan
                    
            if 'STANDARD PRESSURE' in line:
                station_info['standard_pressure'] = cls._extract_first_numeric_value(parts)
                
            if 'PRE-INCREASE AVERAGE' in line and 'COUNTING RATE' in line:
                baseline_rate = cls._extract_baseline_rate(parts)
                if baseline_rate is not None:
                    station_info['baseline_rate'] = baseline_rate
                    
        return station_info

    @staticmethod
    def _extract_first_numeric_value(parts):
        """Extract the first valid numeric value from a list of strings."""
        for p in parts:
            try:
                return float(p)
            except ValueError:
                continue
        return np.nan

    @staticmethod
    def _extract_baseline_rate(parts):
        """Extract the baseline rate from a list of strings."""
        for p in parts:
            try:
                val = float(p)
                # Typically baseline rates are between 1 and 10000
                if 1 <= val <= 10000:
                    return val
            except ValueError:
                continue
        return np.nan

    @classmethod
    def _parse_data_line(cls, line, station_info):
        """
        Parse a single data line from a neutron monitor data file.
        
        Parameters:
        line (str): A line from the neutron monitor data file
        station_info (dict): Dictionary containing station information
        
        Returns:
        dict or None: Dictionary containing parsed data, or None if parsing failed
        """
        match = re.match(r'^(\w+)\s+(\d{6})\s+(\d+)\s+([\d-]+)\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)', line)
        
        if not match:
            return None
            
        try:
            station = match.group(1)
            date_str = match.group(2)  # YYMMDD format
            seconds = int(match.group(3))
            time_interval = match.group(4)
            code = match.group(5)
            uncorrected = float(match.group(6))
            pressure = float(match.group(7))
            corrected = float(match.group(8))
            percentage_increase = float(match.group(9))
            
            timestamp = cls._parse_timestamp(date_str, time_interval)
            corrected_percentage = cls._parse_corrected_percentage(line, match)
            
            # Create data entry
            entry = {
                'station': station,
                'timestamp': timestamp,
                'interval_seconds': seconds,
                'time_interval': time_interval,
                'code': code,
                'uncorrected_count_rate': uncorrected,
                'pressure_mb': pressure,
                'corrected_count_rate': corrected,
                'percentage_increase': percentage_increase,
                'corrected_percentage_increase': corrected_percentage
            }
            
            # Add station info if available
            for key, value in station_info.items():
                entry[key] = value
                
            return entry
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _parse_timestamp(date_str, time_interval):
        """Parse date and time from strings into a datetime object."""
        try:
            year = 2000 + int(date_str[0:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            
            # Parse the time interval to get start time
            start_time = time_interval.split('-')[0]
            hour = int(start_time[0:2])
            minute = int(start_time[2:4])
            second = int(start_time[4:6])
            
            return dt.datetime(year, month, day, hour, minute, second)
        except (ValueError, IndexError):
            return np.nan

    @staticmethod
    def _parse_corrected_percentage(line, match):
        """Parse the corrected percentage increase from the line."""
        remaining = line[match.end():].strip()
        if remaining and remaining != '-9999':
            try:
                return float(remaining)
            except ValueError:
                return np.nan
        return np.nan

    @classmethod
    def from_file(cls, file_path):
        """
        Create a NeutronMonitorData object from a neutron monitor data file.
        
        Parameters:
        file_path (str): Path to the neutron monitor data file
        
        Returns:
        NeutronMonitorData: DataFrame containing the neutron monitor data
        """
        # Open and read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Extract station information from header
        station_info = cls.extract_station_info(lines)
        
        # Process data lines
        data = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            entry = cls._parse_data_line(line, station_info)
            if entry:
                data.append(entry)
        
        # Create DataFrame with the data
        df = pd.DataFrame(data)
        
        # Convert any string columns that should be numeric
        numeric_cols = ['uncorrected_count_rate', 'pressure_mb', 'corrected_count_rate', 
                       'percentage_increase', 'corrected_percentage_increase',
                       'latitude', 'longitude', 'altitude', 'standard_pressure', 'baseline_rate']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Store original values and replace negative/zero percentages with 0.0
        if 'percentage_increase' in df.columns:
            df['original_percentage_increase'] = df['percentage_increase']
            df.loc[df['percentage_increase'] <= 0.0, 'percentage_increase'] = 0.0
            
        if 'corrected_percentage_increase' in df.columns:
            df['original_corrected_percentage_increase'] = df['corrected_percentage_increase']
            df.loc[df['corrected_percentage_increase'] <= 0.0, 'corrected_percentage_increase'] = 0.0
        
        # Return as NeutronMonitorData instance
        return cls(copy.deepcopy(df))

    def find_exceeding_percentage(self, percentage=8):
        """
        Find records where corrected count rate exceeds a given percentage more than the baseline rate.
        
        Parameters:
        percentage (float): Percentage threshold (default: 8)
        
        Returns:
        NeutronMonitorData: DataFrame containing filtered records
        """
        threshold = 1 + (percentage / 100)
        filtered = self[self['corrected_count_rate'] > self['baseline_rate'] * threshold]
        return filtered
    
    def get_latitude(self):
        """
        Get the latitude of the neutron monitor station.
        
        Returns:
        float: Latitude value or NaN if not available
        """
        if 'latitude' in self.columns and not self['latitude'].isna().all():
            return self['latitude'].iloc[0]
        return np.nan
    
    def get_longitude(self):
        """
        Get the longitude of the neutron monitor station.
        
        Returns:
        float: Longitude value or NaN if not available
        """
        if 'longitude' in self.columns and not self['longitude'].isna().all():
            return self['longitude'].iloc[0]
        return np.nan
    
    def get_altitude(self):
        """
        Get the altitude of the neutron monitor station in kilometers.
        
        Returns:
        float: Altitude value in km or NaN if not available
        """
        if 'altitude' in self.columns and not self['altitude'].isna().all():
            return self['altitude'].iloc[0]
        return np.nan
    
    def get_standard_pressure(self):
        """
        Get the standard pressure at the neutron monitor station.
        
        Returns:
        float: Standard pressure value or NaN if not available
        """
        if 'standard_pressure' in self.columns and not self['standard_pressure'].isna().all():
            return self['standard_pressure'].iloc[0]
        return np.nan
    
    def get_baseline_rate(self):
        """
        Get the baseline count rate of the neutron monitor station.
        
        Returns:
        float: Baseline rate value or NaN if not available
        """
        if 'baseline_rate' in self.columns and not self['baseline_rate'].isna().all():
            return self['baseline_rate'].iloc[0]
        return np.nan
    
    def get_station_name(self):
        """
        Get the name of the neutron monitor station.
        
        Returns:
        str: Station name or None if not available
        """
        if 'station' in self.columns and not self['station'].isna().all():
            return self['station'].iloc[0]
        return None
    
    def get_location(self):
        """
        Get the location of the neutron monitor station as a tuple (latitude, longitude, altitude).
        
        Returns:
        tuple: (latitude, longitude, altitude) or (NaN, NaN, NaN) if not available
        """
        return (self.get_latitude(), self.get_longitude(), self.get_altitude())

class NM_pair:
    def __init__(self, primary:NeutronMonitorData, secondary:NeutronMonitorData):
        # Get monitor information from the NM_info dictionary
        self.primary = primary
        self.secondary = secondary
        
        # Create a name for the pair
        self.name = f"{primary.get_station_name()}-{secondary.get_station_name()}"
        
        # Get locations from the NM_info dictionary
        # Find the full name keys that contain the monitor codes
        primary_key = primary.get_station_name()
        secondary_key = secondary.get_station_name()
        
        if not primary_key or not secondary_key:
            raise ValueError(f"Could not find monitor codes {primary} and/or {secondary} in NM_info dictionary")
        
        self.primary_location = primary.get_location()
        
        self.secondary_location = secondary.get_location()

        self.midpoint_location = self.get_midpoint()

    def __str__(self):
        """
        Return a string representation of the NM_pair object with useful information.
        
        Returns:
        --------
        str
            A formatted string containing the pair name and location information
        """
        return (f"NM Pair: {self.name}\n"
                f"Primary: {self.primary} at ({self.primary_location[0]:.2f}°, {self.primary_location[1]:.2f}°, {self.primary_location[2]:.3f} km)\n"
                f"Secondary: {self.secondary} at ({self.secondary_location[0]:.2f}°, {self.secondary_location[1]:.2f}°, {self.secondary_location[2]:.3f} km)\n"
                f"Midpoint: ({self.midpoint_location[0]:.2f}°, {self.midpoint_location[1]:.2f}°)")
    
    def __repr__(self):
        """
        Return a string representation for display in notebooks.
        """
        return self.__str__()

    def get_midpoint(self):
        """
        Calculate the geographic midpoint between the primary and secondary neutron monitors,
        ignoring altitude differences.
        
        Returns:
        --------
        list
            A list containing [latitude_deg, longitude_deg] of the midpoint
        """
        # Extract coordinates
        lat1, lon1, _ = self.primary_location
        lat2, lon2, _ = self.secondary_location
        
        # Convert to radians for spherical calculations
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        # Calculate the midpoint using spherical geometry
        # Convert to Cartesian coordinates
        x1 = np.cos(lat1_rad) * np.cos(lon1_rad)
        y1 = np.cos(lat1_rad) * np.sin(lon1_rad)
        z1 = np.sin(lat1_rad)
        
        x2 = np.cos(lat2_rad) * np.cos(lon2_rad)
        y2 = np.cos(lat2_rad) * np.sin(lon2_rad)
        z2 = np.sin(lat2_rad)
        
        # Calculate midpoint in Cartesian coordinates
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        z_mid = (z1 + z2) / 2
        
        # Convert back to spherical coordinates
        lon_mid = np.arctan2(y_mid, x_mid)
        hyp = np.sqrt(x_mid**2 + y_mid**2)
        lat_mid = np.arctan2(z_mid, hyp)
        
        # Convert to degrees
        lat_mid_deg = np.degrees(lat_mid)
        lon_mid_deg = np.degrees(lon_mid)
        
        return [lat_mid_deg, lon_mid_deg]
    
    def filter_by_threshold(self, threshold_percentage=8.0, use_corrected=True):
        """
        Filter both primary and secondary monitors to only include data points 
        where the percentage increase exceeds the given threshold.
        
        Parameters:
        -----------
        threshold_percentage : float
            The minimum percentage increase to include in the filtered data
        use_corrected : bool
            Whether to use corrected_percentage_increase (if available) or percentage_increase
            
        Returns:
        --------
        tuple
            A tuple containing (filtered_primary, filtered_secondary)
        """
        filtered_primary = self._filter_monitor_by_threshold(self.primary, threshold_percentage, use_corrected)
        filtered_secondary = self._filter_monitor_by_threshold(self.secondary, threshold_percentage, use_corrected)
        return NM_pair(filtered_primary, filtered_secondary)
    
    def _filter_monitor_by_threshold(self, monitor, threshold_percentage, use_corrected=True):
        """
        Helper method to filter a single monitor's data by percentage increase threshold.
        
        Parameters:
        -----------
        monitor : NeutronMonitorData
            The monitor data to filter
        threshold_percentage : float
            The minimum percentage increase to include
        use_corrected : bool
            Whether to use corrected_percentage_increase (if available) or percentage_increase
            
        Returns:
        --------
        NeutronMonitorData
            Filtered monitor data
        """
        # Check if corrected_percentage_increase is available and should be used
        if use_corrected and 'corrected_percentage_increase' in monitor.columns and not monitor['corrected_percentage_increase'].isna().all():
            return monitor[monitor['corrected_percentage_increase'] >= threshold_percentage]
        else:
            # Fall back to regular percentage_increase
            return monitor[monitor['percentage_increase'] >= threshold_percentage]
    
    def filter_by_datetime(self, start_time=None, end_time=None):
        """
        Filter both primary and secondary monitors to only include data points within the specified datetime range.

        Parameters:
        -----------
        start_time : datetime or str
            The start time of the filter range. If string, should be in format 'YYYY-MM-DD HH:MM:SS'
        end_time : datetime or str
            The end time of the filter range. If string, should be in format 'YYYY-MM-DD HH:MM:SS'

        Returns:
        --------
        NM_pair
            A new NM_pair object containing the filtered data
        """
        # Convert string dates to datetime if needed
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)

        def filter_monitor_by_time(monitor):
            if start_time is not None and end_time is not None:
                return monitor[(monitor['timestamp'] >= start_time) & (monitor['timestamp'] <= end_time)]
            elif start_time is not None:
                return monitor[monitor['timestamp'] >= start_time]
            elif end_time is not None:
                return monitor[monitor['timestamp'] <= end_time]
            else:
                return monitor.copy()

        filtered_primary = filter_monitor_by_time(self.primary)
        filtered_secondary = filter_monitor_by_time(self.secondary)
        return NM_pair(filtered_primary, filtered_secondary)

class full_NM_set:
    """
    A class representing a complete neutron monitor set, including a pair of neutron monitors
    and a specific normalization neutron monitor.
    
    This class extends the functionality of the NM_pair class by adding normalization
    information and methods.
    
    Attributes:
    -----------
    nm_pair : NM_pair
        The neutron monitor pair object
    normalisation_station_code : str
        The station code for the normalization neutron monitor
    normalisation_location : list
        The geographic coordinates [latitude_deg, longitude_deg, altitude_in_km] of the normalization station
    """
    
    def __init__(self, primary:NeutronMonitorData, secondary:NeutronMonitorData, normalisation:NeutronMonitorData):
        """
        Initialize a full neutron monitor set with a pair and normalization information.
        
        Parameters:
        -----------
        primary : str
            The station code for the primary neutron monitor
        secondary : str
            The station code for the secondary neutron monitor
        normalisation_station_code : str
            The station code for the normalization neutron monitor
        normalisation_location : list, optional
            The geographic coordinates [latitude_deg, longitude_deg, altitude_in_km] of the normalization station.
            If None, the location will be retrieved from NM_info.
        """
        self.nm_pair = NM_pair(primary, secondary)
        self.normalisation = normalisation
    
    def __str__(self):
        """
        Return a string representation of the full_NM_set object with useful information.
        """
        norm_loc = self.get_normalisation_location()
        return (f"Full Neutron Monitor Set:\n"
                f"{self.nm_pair}\n"
                f"Normalization: {self.normalisation} at ({norm_loc[0]:.2f}°, {norm_loc[1]:.2f}°, {norm_loc[2]:.3f} km)")
    
    def __repr__(self):
        """
        Return a string representation for display in notebooks.
        """
        return self.__str__()
    
    def get_primary(self):
        """Return the primary neutron monitor station code."""
        return self.nm_pair.primary
    
    def get_secondary(self):
        """Return the secondary neutron monitor station code."""
        return self.nm_pair.secondary
    
    def get_normalisation(self):
        """Return the normalization neutron monitor station code."""
        return self.normalisation
    
    def get_primary_location(self):
        """Return the primary neutron monitor location."""
        return self.nm_pair.primary_location
    
    def get_secondary_location(self):
        """Return the secondary neutron monitor location."""
        return self.nm_pair.secondary_location
    
    def get_normalisation_location(self):
        """Return the normalization neutron monitor location."""
        return self.normalisation.get_location()
    
    def get_midpoint_location(self):
        """Return the midpoint location between primary and secondary monitors."""
        return self.nm_pair.get_midpoint()
    
    def filter_by_threshold(self, threshold_percentage=8.0, use_corrected=True, norm_only=False):
        """
        Filter monitors based on percentage increase threshold.
        
        Parameters:
        -----------
        threshold_percentage : float
            The minimum percentage increase to include in the filtered data
        use_corrected : bool
            Whether to use corrected_percentage_increase (if available) or percentage_increase
        norm_only : bool
            If True, only filter the normalization monitor by threshold and filter primary/secondary
            monitors by the datetime range of the filtered normalization data
            
        Returns:
        --------
        full_NM_set
            A new full_NM_set object containing the filtered data
        """
        # Filter normalization monitor
        if use_corrected and 'corrected_percentage_increase' in self.normalisation.columns and not self.normalisation['corrected_percentage_increase'].isna().all():
            filtered_norm = self.normalisation[self.normalisation['corrected_percentage_increase'] >= threshold_percentage]
        else:
            filtered_norm = self.normalisation[self.normalisation['percentage_increase'] >= threshold_percentage]
        
        if norm_only and not filtered_norm.empty:
            # Get the datetime range from the filtered normalization data
            start_time = filtered_norm.index.min()
            end_time = filtered_norm.index.max()
            
            # Filter primary and secondary monitors by this datetime range
            filtered_NM_pair = self.nm_pair.filter_by_datetime(start_time, end_time)
        else:
            # Filter all monitors by threshold
            filtered_NM_pair = self.nm_pair.filter_by_threshold(
                threshold_percentage, use_corrected
            )
            
        return full_NM_set(filtered_NM_pair.primary, filtered_NM_pair.secondary, filtered_norm)
    
    def filter_by_datetime(self, start_time=None, end_time=None):
        """
        Filter all monitors (primary, secondary, and normalization) to only include 
        data points within the specified datetime range.
        
        Parameters:
        -----------
        start_time : datetime or str
            The start time of the filter range. If string, should be in format 'YYYY-MM-DD HH:MM:SS'
        end_time : datetime or str
            The end time of the filter range. If string, should be in format 'YYYY-MM-DD HH:MM:SS'
            
        Returns:
        --------
        full_NM_set
            A new full_NM_set object containing the filtered data
        """
        # Convert string dates to datetime if needed
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        
        # Filter primary and secondary monitors through the NM_pair
        filtered_NM_pair = self.nm_pair.filter_by_datetime(start_time, end_time)
        
        # Filter normalization monitor
        if start_time is not None and end_time is not None:
            filtered_norm = self.normalisation[(self.normalisation.timestamp >= start_time) & 
                                              (self.normalisation.timestamp <= end_time)]
        elif start_time is not None:
            filtered_norm = self.normalisation[self.normalisation.timestamp >= start_time]
        elif end_time is not None:
            filtered_norm = self.normalisation[self.normalisation.timestamp <= end_time]
        else:
            filtered_norm = self.normalisation.copy()
            
        return full_NM_set(filtered_NM_pair.primary, filtered_NM_pair.secondary, filtered_norm)
    
    def get_event_start_data(self, threshold_percentage=8.0, use_corrected=True):
        """
        Get the first data points from each monitor that exceed the threshold percentage increase.
        This is useful for identifying the start of an event.
        
        Parameters:
        -----------
        threshold_percentage : float
            The minimum percentage increase to consider as the start of an event
        use_corrected : bool
            Whether to use corrected_percentage_increase (if available) or percentage_increase
            
        Returns:
        --------
        dict
            A dictionary containing the first data points for each monitor that exceed the threshold
        """
        filtered_primary, filtered_secondary, filtered_norm = self.filter_by_threshold(
            threshold_percentage, use_corrected
        )
        
        # Get the first row from each filtered dataset if available
        result = {}
        if not filtered_primary.empty:
            result['primary'] = filtered_primary.iloc[0]
        if not filtered_secondary.empty:
            result['secondary'] = filtered_secondary.iloc[0]
        if not filtered_norm.empty:
            result['normalisation'] = filtered_norm.iloc[0]
            
        return result
    
    def get_event_start_datetime(self, threshold_percentage=8.0, use_corrected=True):
        """
        Get the datetime when the event first exceeds the threshold percentage increase
        using the normalisation monitor.
        
        Parameters:
        -----------
        threshold_percentage : float
            The minimum percentage increase to consider as the start of an event
        use_corrected : bool
            Whether to use corrected_percentage_increase (if available) or percentage_increase
            
        Returns:
        --------
        pd.Timestamp or None
            The datetime when the event first exceeds the threshold, or None if no data exceeds the threshold
        """
        # Get the column to check based on use_corrected flag
        column_name = 'corrected_percentage_increase' if use_corrected and 'corrected_percentage_increase' in self.normalisation.columns else 'percentage_increase'
        
        # Check if the column exists in the normalisation monitor
        if column_name not in self.normalisation.columns:
            return None
            
        # Find the first point where it exceeds the threshold
        above_threshold = self.normalisation[self.normalisation[column_name] >= threshold_percentage]
        
        # Return the earliest datetime if any were found
        return above_threshold.iloc[0]["timestamp"] if not above_threshold.empty else None
    
    def get_event_end_datetime(self, threshold_percentage=8.0, use_corrected=True, min_duration_minutes=30):
        """
        Get the datetime when the event falls below the threshold percentage increase after having exceeded it,
        using the normalisation monitor.
        
        Parameters:
        -----------
        threshold_percentage : float
            The percentage increase threshold
        use_corrected : bool
            Whether to use corrected_percentage_increase (if available) or percentage_increase
        min_duration_minutes : int
            Minimum duration in minutes to consider before looking for the end of the event
            
        Returns:
        --------
        pd.Timestamp or None
            The datetime when the event ends (falls below threshold), or None if no end is found
        """
        # First get the start time
        start_datetime = self.get_event_start_datetime(threshold_percentage, use_corrected)
        if start_datetime is None:
            return None
            
        # Calculate the minimum end time based on the minimum duration
        min_end_time = start_datetime + pd.Timedelta(minutes=min_duration_minutes)
        
        # Get the column to check based on use_corrected flag
        column_name = 'corrected_percentage_increase' if use_corrected and 'corrected_percentage_increase' in self.normalisation.columns else 'percentage_increase'
        
        # Check if the column exists in the normalisation monitor
        if column_name not in self.normalisation.columns:
            return None
            
        # Get data after the start time and minimum duration
        data_after_start = self.normalisation[self.normalisation["timestamp"] > min_end_time]
        
        if data_after_start.empty:
            return None
            
        # Find the first point where it falls below threshold after min_duration
        below_threshold = data_after_start[data_after_start[column_name] < threshold_percentage]
        
        # Return the earliest datetime when it falls below threshold
        return below_threshold.iloc[0]["timestamp"] if not below_threshold.empty else None
    
valid_NM_pairs = [
    ["TXBY", "YKTK"],
    ["TXBY", "MGDN"],
    ["APTY", "MOSC"],
    ["NAIN", "NWRK"],
    ["OULU", "KIEL"],
    ["OULU", "DRBS"],
    ["TERA", "KGSN"],
    ["MCMD", "KGSN"],
    ["TXBY", "MGDN"],
    ["SOPO", "KERG"],
]

class MAIREPLUS_event(BaseAniMAIREEvent):
    """Main class for handling MAIREPLUS-spectrum-based events."""
    def __init__(
        self,
        neutron_monitor_1_percentage_increase: Union[float, Sequence[float]],
        neutron_monitor_2_percentage_increase: Union[float, Sequence[float]],
        normalisation_monitor_percentage_increase: Union[float, Sequence[float]],
        OULU_gcr_count_rate_in_seconds: Union[float, Sequence[float]],
        datetime: Union[dt.datetime, Sequence[dt.datetime]],
        kp_index: Optional[Union[int, Sequence[int]]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        # --- INPUT PREPARATION ------------------------------------------------
        # Map original inputs into a dict for uniform handling
        raw_inputs = {
            'neutron_monitor_1_percentage_increase': neutron_monitor_1_percentage_increase,
            'neutron_monitor_2_percentage_increase': neutron_monitor_2_percentage_increase,
            'normalisation_monitor_percentage_increase': normalisation_monitor_percentage_increase,
            'OULU_gcr_count_rate_in_seconds': OULU_gcr_count_rate_in_seconds,
            'datetime': datetime,
            'kp_index': kp_index,
        }
        # Determine how many events: any sequence input length >1 sets count
        seq_lengths = [
            len(v) for v in raw_inputs.values()
            if isinstance(v, (list, tuple, np.ndarray, pd.Series))
        ]
        if seq_lengths:
            num_events = seq_lengths[0]
            if any(l != num_events for l in seq_lengths):
                raise ValueError('All sequence inputs must have the same number of elements.')
        else:
            num_events = 1

        # Broadcast inputs into lists of length 'num_events'
        broadcasted_inputs: Dict[str, list] = {}
        for name, val in raw_inputs.items():
            # Convert pandas Series or numpy array to list
            if isinstance(val, pd.Series):
                items = val.tolist()
            elif isinstance(val, np.ndarray):
                items = list(val)
            elif isinstance(val, (list, tuple)):
                items = list(val)
            else:
                items = [val]
            # Expand single-value lists to match num_events
            if len(items) == 1:
                broadcasted_inputs[name] = items * num_events
            elif len(items) == num_events:
                broadcasted_inputs[name] = items
            else:
                raise ValueError(f"Parameter '{name}' has length {len(items)} but expected 1 or {num_events}.")

        # Store count and prepared parameters
        self._count = num_events
        self.params = broadcasted_inputs
        self.run_kwargs = kwargs
        # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Return enhanced string representation showing neutron monitor details.
        
        Returns:
            str: Detailed representation of the MAIREPLUS_event with monitor information
        """
        # Get the base representation from parent class
        base_repr = super().__repr__()
        
        # Add MAIREPLUS-specific information
        enhancement_lines = []
        
        # Add event count information
        event_count = getattr(self, '_count', 1)
        enhancement_lines.append(f"Event count: {event_count}")
        
        # Add detailed neutron monitor information
        monitor_details = self._get_detailed_monitor_info()
        if monitor_details:
            enhancement_lines.extend(monitor_details)
        
        # Add percentage increase ranges if available
        if hasattr(self, 'params'):
            for param in ['neutron_monitor_1_percentage_increase', 'neutron_monitor_2_percentage_increase', 'normalisation_monitor_percentage_increase']:
                if param in self.params:
                    values = self.params[param]
                    if isinstance(values, list) and len(values) > 0:
                        if len(values) == 1:
                            enhancement_lines.append(f"{param.replace('_', ' ').title()}: {values[0]:.2f}%")
                        else:
                            min_val, max_val = min(values), max(values)
                            enhancement_lines.append(f"{param.replace('_', ' ').title()}: {min_val:.2f}% to {max_val:.2f}%")
        
        # Add OULU baseline information
        if hasattr(self, 'params') and 'OULU_gcr_count_rate_in_seconds' in self.params:
            baseline = self.params['OULU_gcr_count_rate_in_seconds']
            if isinstance(baseline, list) and len(baseline) > 0:
                baseline_val = baseline[0]
                enhancement_lines.append(f"OULU baseline rate: {baseline_val:.2f} counts/sec")
        
        # Add spectral information
        spectral_details = self._get_spectral_details()
        if spectral_details:
            enhancement_lines.extend(spectral_details)
        
        # Combine base representation with enhancements
        if enhancement_lines:
            enhancement_text = "\n".join(f"  • {line}" for line in enhancement_lines)
            return f"{base_repr}\n\nMAIREPLUS Event Details:\n{enhancement_text}"
        else:
            return base_repr

    def _repr_html_(self) -> str:
        """
        Return enhanced HTML representation for Jupyter notebook display.
        
        Returns:
            str: HTML representation with detailed neutron monitor information
        """
        # Get basic event information
        class_name = self.__class__.__name__
        n_timestamps = len(self.dose_rates) if hasattr(self, 'dose_rates') and self.dose_rates else 0
        event_count = getattr(self, '_count', 1)
        
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
        
        # Start building HTML
        html = f"""
        <div style="background-color:#f8f9fa;padding:15px;border-radius:6px;border:1px solid #ddd;margin:10px 0;">
            <h3 style="margin-top:0;color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:5px;">
                🚀 {class_name}
            </h3>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
                <div>
                    <h4 style="color:#2980b9;margin-bottom:8px;">📊 Event Overview</h4>
                    <table style="border-collapse:collapse;width:100%;">
                        <tr><td style="padding:4px;border:1px solid #ddd;background:#ecf0f1;"><b>Event count</b></td><td style="padding:4px;border:1px solid #ddd;">{event_count}</td></tr>
                        <tr><td style="padding:4px;border:1px solid #ddd;background:#ecf0f1;"><b>Timestamps</b></td><td style="padding:4px;border:1px solid #ddd;">{n_timestamps}</td></tr>
                        <tr><td style="padding:4px;border:1px solid #ddd;background:#ecf0f1;"><b>Time range</b></td><td style="padding:4px;border:1px solid #ddd;">{timestamp_range}</td></tr>
                    </table>
                </div>
                <div>
                    <h4 style="color:#2980b9;margin-bottom:8px;">🛰️ Neutron Monitor Setup</h4>
        """
        
        # Add detailed monitor information
        monitor_details = self._get_detailed_monitor_info_for_html()
        if monitor_details:
            html += monitor_details
        else:
            html += "<p style='font-style:italic;color:#7f8c8d;'>No monitor details available</p>"
        
        html += """
                </div>
            </div>
        """
        
        # Add percentage increase information if available
        if hasattr(self, 'params'):
            html += """
            <div style="margin-top:15px;">
                <h4 style="color:#2980b9;margin-bottom:8px;">📈 Percentage Increases</h4>
                <table style="border-collapse:collapse;width:100%;">
            """
            
            monitor_labels = {
                'neutron_monitor_1_percentage_increase': 'Primary Monitor',
                'neutron_monitor_2_percentage_increase': 'Secondary Monitor', 
                'normalisation_monitor_percentage_increase': 'Normalisation Monitor'
            }
            
            for param, label in monitor_labels.items():
                if param in self.params:
                    values = self.params[param]
                    if isinstance(values, list) and len(values) > 0:
                        if len(values) == 1:
                            value_display = f"{values[0]:.2f}%"
                        else:
                            min_val, max_val = min(values), max(values)
                            value_display = f"{min_val:.2f}% to {max_val:.2f}%"
                        html += f"<tr><td style='padding:4px;border:1px solid #ddd;background:#ecf0f1;width:40%;'><b>{label}</b></td><td style='padding:4px;border:1px solid #ddd;'>{value_display}</td></tr>"
            
            # Add OULU baseline
            if 'OULU_gcr_count_rate_in_seconds' in self.params:
                baseline = self.params['OULU_gcr_count_rate_in_seconds']
                if isinstance(baseline, list) and len(baseline) > 0:
                    baseline_val = baseline[0]
                    html += f"<tr><td style='padding:4px;border:1px solid #ddd;background:#ecf0f1;'><b>OULU Baseline Rate</b></td><td style='padding:4px;border:1px solid #ddd;'>{baseline_val:.2f} counts/sec</td></tr>"
            
            html += "</table></div>"
        
        # Add spectral information section
        spectral_html = self._get_spectral_details_for_html()
        if spectral_html:
            html += spectral_html
        
        # Add available data section
        html += """
            <div style="margin-top:15px;">
                <h4 style="color:#2980b9;margin-bottom:8px;">📋 Available Data</h4>
        """
        
        if altitudes:
            html += "<p><b>Altitudes:</b> " + ", ".join([f"{alt:.2f} km" for alt in altitudes]) + "</p>"
            
        if dose_types:
            html += "<p><b>Dose types:</b> " + ", ".join(dose_types) + "</p>"
        
        # Add help text
        html += """
                <p style="margin-top:10px;font-size:0.9em;color:#7f8c8d;">
                    <i>💡 Available methods: run_AniMAIRE(), plot_map_at_time(), create_gle_map_animation(), etc.</i>
                </p>
            </div>
        </div>
        """
        return html

    def _get_detailed_monitor_info(self) -> List[str]:
        """
        Get detailed neutron monitor information for text representation.
        
        Returns:
            List[str]: List of formatted monitor information strings
        """
        details = []
        
        # Try to get information from nm_set attribute (if created from files)
        if hasattr(self, 'nm_set'):
            try:
                nm_set = self.nm_set
                primary = nm_set.get_primary()
                secondary = nm_set.get_secondary()
                normalisation = nm_set.get_normalisation()
                
                details.append(f"Primary Monitor: {primary.get_station_name()} at {primary.get_location()}")
                details.append(f"Secondary Monitor: {secondary.get_station_name()} at {secondary.get_location()}")
                details.append(f"Normalisation Monitor: {normalisation.get_station_name()} at {normalisation.get_location()}")
                return details
            except Exception:
                pass
        
        # Fallback to location information from parameters
        if hasattr(self, 'params') or hasattr(self, 'run_kwargs'):
            params = getattr(self, 'params', {})
            run_kwargs = getattr(self, 'run_kwargs', {})
            
            monitor_info = [
                ('Primary', 'neutron_monitor_1_location'),
                ('Secondary', 'neutron_monitor_2_location'),
                ('Normalisation', 'normalisation_monitor_location')
            ]
            
            for label, key in monitor_info:
                location = None
                if key in params:
                    location = params[key][0] if isinstance(params[key], list) else params[key]
                elif key in run_kwargs:
                    location = run_kwargs[key]
                
                if location:
                    if isinstance(location, (list, tuple)) and len(location) >= 2:
                        details.append(f"{label} Monitor: Location ({location[0]:.2f}°, {location[1]:.2f}°)")
                    else:
                        details.append(f"{label} Monitor: {location}")
        
        return details

    def _get_detailed_monitor_info_for_html(self) -> str:
        """
        Get detailed neutron monitor information formatted as HTML.
        
        Returns:
            str: HTML formatted monitor information
        """
        # Try to get information from nm_set attribute (if created from files)
        if hasattr(self, 'nm_set'):
            try:
                nm_set = self.nm_set
                primary = nm_set.get_primary()
                secondary = nm_set.get_secondary()
                normalisation = nm_set.get_normalisation()
                
                html = """
                <table style="border-collapse:collapse;width:100%;">
                    <tr><td style="padding:4px;border:1px solid #ddd;background:#ecf0f1;"><b>Primary</b></td><td style="padding:4px;border:1px solid #ddd;">{} at ({:.2f}°, {:.2f}°)</td></tr>
                    <tr><td style="padding:4px;border:1px solid #ddd;background:#ecf0f1;"><b>Secondary</b></td><td style="padding:4px;border:1px solid #ddd;">{} at ({:.2f}°, {:.2f}°)</td></tr>
                    <tr><td style="padding:4px;border:1px solid #ddd;background:#ecf0f1;"><b>Normalisation</b></td><td style="padding:4px;border:1px solid #ddd;">{} at ({:.2f}°, {:.2f}°)</td></tr>
                </table>
                """.format(
                    primary.get_station_name(), primary.get_location()[0], primary.get_location()[1],
                    secondary.get_station_name(), secondary.get_location()[0], secondary.get_location()[1],
                    normalisation.get_station_name(), normalisation.get_location()[0], normalisation.get_location()[1]
                )
                return html
            except Exception:
                pass
        
        # Fallback to location information from parameters
        if hasattr(self, 'params') or hasattr(self, 'run_kwargs'):
            params = getattr(self, 'params', {})
            run_kwargs = getattr(self, 'run_kwargs', {})
            
            html = "<table style='border-collapse:collapse;width:100%;'>"
            
            monitor_info = [
                ('Primary', 'neutron_monitor_1_location'),
                ('Secondary', 'neutron_monitor_2_location'),
                ('Normalisation', 'normalisation_monitor_location')
            ]
            
            for label, key in monitor_info:
                location = None
                if key in params:
                    location = params[key][0] if isinstance(params[key], list) else params[key]
                elif key in run_kwargs:
                    location = run_kwargs[key]
                
                if location:
                    if isinstance(location, (list, tuple)) and len(location) >= 2:
                        html += f"<tr><td style='padding:4px;border:1px solid #ddd;background:#ecf0f1;'><b>{label}</b></td><td style='padding:4px;border:1px solid #ddd;'>({location[0]:.2f}°, {location[1]:.2f}°)</td></tr>"
                    else:
                        html += f"<tr><td style='padding:4px;border:1px solid #ddd;background:#ecf0f1;'><b>{label}</b></td><td style='padding:4px;border:1px solid #ddd;'>{location}</td></tr>"
            
            html += "</table>"
            return html
        
        return ""

    def _get_spectral_details(self) -> List[str]:
        """
        Get spectral information for text representation.
        
        Returns:
            List[str]: List of formatted spectral information strings
        """
        details = []
        
        # Try to get spectral summary from parent class
        try:
            spectra_summary = self.summarize_spectra()
            if spectra_summary:
                details.append(f"Spectral model: {spectra_summary.get('spectrum_type', 'MAIREPLUS')}")
                
                if 'particle_types' in spectra_summary:
                    particle_info = spectra_summary['particle_types']
                    if isinstance(particle_info, dict):
                        particles = ", ".join([f"{k} (Z={v})" for k, v in particle_info.items()])
                        details.append(f"Particle types: {particles}")
                    elif isinstance(particle_info, list):
                        details.append(f"Particle types: {', '.join(map(str, particle_info))}")
                
                if 'rigidity_range' in spectra_summary:
                    rig_range = spectra_summary['rigidity_range']
                    if isinstance(rig_range, (list, tuple)) and len(rig_range) >= 2:
                        details.append(f"Rigidity range: {rig_range[0]:.2f} to {rig_range[1]:.2f} GV")
                
                if 'flux_range' in spectra_summary:
                    flux_range = spectra_summary['flux_range']
                    if isinstance(flux_range, (list, tuple)) and len(flux_range) >= 2:
                        details.append(f"Flux range: {flux_range[0]:.2e} to {flux_range[1]:.2e} particles/m²/sr/s/GV")
        except Exception:
            # If summarize_spectra fails, try to extract basic info
            pass
        
        # Add information about spectrum generation method
        if hasattr(self, 'dose_rates') and self.dose_rates:
            first_frame = next(iter(self.dose_rates.values()))
            if hasattr(first_frame, 'particle_distributions'):
                n_particles = len(first_frame.particle_distributions) if first_frame.particle_distributions else 0
                if n_particles > 0:
                    details.append(f"Particle distributions: {n_particles}")
                    
                    # Try to get spectrum type info
                    try:
                        first_dist = first_frame.particle_distributions[0]
                        if hasattr(first_dist, 'momentum_distribution') and \
                           hasattr(first_dist.momentum_distribution, 'rigidity_spectrum'):
                            spectrum_obj = first_dist.momentum_distribution.rigidity_spectrum
                            spectrum_class = spectrum_obj.__class__.__name__
                            details.append(f"Spectrum type: {spectrum_class}")
                    except Exception:
                        pass
        
        return details

    def _get_spectral_details_for_html(self) -> str:
        """
        Get spectral information formatted as HTML.
        
        Returns:
            str: HTML formatted spectral information
        """
        try:
            spectra_summary = self.summarize_spectra()
            if not spectra_summary:
                return ""
        except Exception:
            return ""
        
        html = """
        <div style="margin-top:15px;">
            <h4 style="color:#2980b9;margin-bottom:8px;">🌟 Spectral Information</h4>
            <table style="border-collapse:collapse;width:100%;">
        """
        
        # Add spectrum type
        spectrum_type = spectra_summary.get('spectrum_type', 'MAIREPLUS')
        html += f"<tr><td style='padding:4px;border:1px solid #ddd;background:#ecf0f1;width:40%;'><b>Spectrum Model</b></td><td style='padding:4px;border:1px solid #ddd;'>{spectrum_type}</td></tr>"
        
        # Add particle types
        if 'particle_types' in spectra_summary:
            particle_info = spectra_summary['particle_types']
            if isinstance(particle_info, dict):
                particles = ", ".join([f"{k} (Z={v})" for k, v in particle_info.items()])
                html += f"<tr><td style='padding:4px;border:1px solid #ddd;background:#ecf0f1;'><b>Particle Types</b></td><td style='padding:4px;border:1px solid #ddd;'>{particles}</td></tr>"
            elif isinstance(particle_info, list):
                particles_str = ', '.join(map(str, particle_info))
                html += f"<tr><td style='padding:4px;border:1px solid #ddd;background:#ecf0f1;'><b>Particle Types</b></td><td style='padding:4px;border:1px solid #ddd;'>{particles_str}</td></tr>"
        
        # Add rigidity range
        if 'rigidity_range' in spectra_summary:
            rig_range = spectra_summary['rigidity_range']
            if isinstance(rig_range, (list, tuple)) and len(rig_range) >= 2:
                html += f"<tr><td style='padding:4px;border:1px solid #ddd;background:#ecf0f1;'><b>Rigidity Range</b></td><td style='padding:4px;border:1px solid #ddd;'>{rig_range[0]:.2f} to {rig_range[1]:.2f} GV</td></tr>"
        
        # Add flux range
        if 'flux_range' in spectra_summary:
            flux_range = spectra_summary['flux_range']
            if isinstance(flux_range, (list, tuple)) and len(flux_range) >= 2:
                html += f"<tr><td style='padding:4px;border:1px solid #ddd;background:#ecf0f1;'><b>Flux Range</b></td><td style='padding:4px;border:1px solid #ddd;'>{flux_range[0]:.2e} to {flux_range[1]:.2e} particles/m²/sr/s/GV</td></tr>"
        
        # Add number of particle distributions
        if hasattr(self, 'dose_rates') and self.dose_rates:
            first_frame = next(iter(self.dose_rates.values()))
            if hasattr(first_frame, 'particle_distributions'):
                n_particles = len(first_frame.particle_distributions) if first_frame.particle_distributions else 0
                if n_particles > 0:
                    html += f"<tr><td style='padding:4px;border:1px solid #ddd;background:#ecf0f1;'><b>Particle Distributions</b></td><td style='padding:4px;border:1px solid #ddd;'>{n_particles}</td></tr>"
                    
                    # Try to get spectrum type info
                    try:
                        first_dist = first_frame.particle_distributions[0]
                        if hasattr(first_dist, 'momentum_distribution') and \
                           hasattr(first_dist.momentum_distribution, 'rigidity_spectrum'):
                            spectrum_obj = first_dist.momentum_distribution.rigidity_spectrum
                            spectrum_class = spectrum_obj.__class__.__name__
                            html += f"<tr><td style='padding:4px;border:1px solid #ddd;background:#ecf0f1;'><b>Spectrum Class</b></td><td style='padding:4px;border:1px solid #ddd;'>{spectrum_class}</td></tr>"
                    except Exception:
                        pass
        
        html += "</table></div>"
        return html

    def run_AniMAIRE(self, use_cache: bool = True, **kwargs: Any) -> Dict[dt.datetime, DoseRateFrame]:
        """Run MAIREPLUS-based AniMAIRE events (vectorized) with optional caching."""
        # Clear any prior results and prepare the component list
        self.dose_rates = {}
        self.dose_rate_components.setdefault('maireplus', [])

        # Loop over each set of input parameters
        num_events = self._count
        for event_index in range(num_events):
            # Prepare parameters for this individual run
            event_parameters = {key: self.params[key][event_index] for key in self.params}
            # Incorporate stored defaults and any overrides
            event_parameters.update(self.run_kwargs)
            event_parameters.update(kwargs)

            # Execute the spectrum calculation (cached or direct)
            if use_cache:
                dose_rate_frame = run_maireplus_cached(**event_parameters)
            else:
                dose_rate_frame = run_maireplus_spectrum(**event_parameters)

            # Extract run timestamp for indexing
            timestamp = getattr(dose_rate_frame, 'timestamp', None)

            # Store the frame in both results and components
            self.dose_rates[timestamp] = dose_rate_frame
            self.dose_rate_components['maireplus'].append(dose_rate_frame)

        return self.dose_rates

    @classmethod
    def from_files(
        cls,
        primary_monitor_file: str,
        secondary_monitor_file: str,
        normalisation_monitor_file: str,
        threshold_percentage: float = 8.0,
        use_corrected: bool = True,
        kp_index: Optional[int] = None,
        OULU_baseline: Optional[Union[str, float]] = None,
        head_n: Optional[int] = None,
        **kwargs: Any
    ) -> 'MAIREPLUS_event':
        """
        Create a MAIREPLUS_event from neutron monitor data files.

        Parameters
        ----------
        primary_monitor_file : str
            Path to the primary neutron monitor data file
        secondary_monitor_file : str
            Path to the secondary neutron monitor data file
        normalisation_monitor_file : str
            Path to the normalisation monitor data file
        threshold_percentage : float, default=8.0
            Percentage threshold for filtering monitor data
        use_corrected : bool, default=True
            Whether to use corrected percentage increases
        kp_index : Optional[int], default=None
            Kp index for the event. If None, will be determined from datetime.
        OULU_baseline : Optional[Union[str, float]], default=None
            Path to the OULU baseline file or baseline value. If None, will use the default method.
        head_n : Optional[int], default=None
            Number of rows to extract from each filtered DataFrame
        **kwargs : Any
            Additional arguments passed to MAIREPLUS_event constructor

        Returns
        -------
        MAIREPLUS_event
            A new event instance configured from the monitor data
        """
        # Load monitor data
        primary_data = NeutronMonitorData.from_file(primary_monitor_file)
        secondary_data = NeutronMonitorData.from_file(secondary_monitor_file)
        normalisation_data = NeutronMonitorData.from_file(normalisation_monitor_file)

        # Create full NM set and filter by threshold
        nm_set = full_NM_set(primary_data, secondary_data, normalisation_data)
        filtered_set = nm_set.filter_by_threshold(threshold_percentage, use_corrected)

        # Extract rows from filtered data for each monitor
        if head_n is not None:
            primary_filtered = filtered_set.get_primary().head(head_n)
            secondary_filtered = filtered_set.get_secondary().head(head_n)
            normalisation_filtered = filtered_set.get_normalisation().head(head_n)
        else:
            primary_filtered = filtered_set.get_primary()
            secondary_filtered = filtered_set.get_secondary()
            normalisation_filtered = filtered_set.get_normalisation()

        # Determine OULU baseline value
        if OULU_baseline is None:
            OULU_gcr_count_rate = filtered_set.get_primary().get_baseline_rate()
        elif isinstance(OULU_baseline, (int, float)):
            OULU_gcr_count_rate = OULU_baseline
        elif isinstance(OULU_baseline, str):
            # Load from file (assume same format as NM data)
            oulu_data = NeutronMonitorData.from_file(OULU_baseline)
            OULU_gcr_count_rate = oulu_data.get_baseline_rate()
        else:
            raise ValueError("OULU_baseline must be None, a number, or a file path.")

        # Create event instance
        event = cls(
            # neutron_monitor_1_percentage_increase=primary_filtered['percentage_increase'].values,
            # neutron_monitor_2_percentage_increase=secondary_filtered['percentage_increase'].values,
            # normalisation_monitor_percentage_increase=normalisation_filtered['percentage_increase'].values,
            neutron_monitor_1_percentage_increase=primary_filtered['corrected_percentage_increase'].values,
            neutron_monitor_2_percentage_increase=secondary_filtered['corrected_percentage_increase'].values,
            normalisation_monitor_percentage_increase=normalisation_filtered['corrected_percentage_increase'].values,
            OULU_gcr_count_rate_in_seconds=OULU_gcr_count_rate,
            datetime=primary_filtered['timestamp'].values,
            kp_index=kp_index,
            neutron_monitor_1_location=filtered_set.get_primary().get_location(),
            neutron_monitor_2_location=filtered_set.get_secondary().get_location(),
            normalisation_monitor_location=filtered_set.get_normalisation().get_location(),
            **kwargs
        )
        
        # Save the full NM set to the event object
        event.nm_set = nm_set
        
        return event