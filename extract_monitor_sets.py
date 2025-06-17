"""
extract_monitor_sets.py

This module provides utilities for discovering, validating, and assembling neutron monitor sets
from a directory of monitor data files for use in AniMAIRE event analysis. It includes logic to:
- Scan a directory for available neutron monitor data files
- Validate and pair monitors according to a list of valid pairs
- Find the nearest valid monitor pair for each available monitor
- Assemble full monitor set objects for use in anisotropic event processing

Typical usage:
    monitor_sets = get_monitor_sets_from_directory('/path/to/monitor/data')

This will return a list of full_NM_set objects, each containing primary, secondary, and normalisation
monitor data, ready for use in event analysis.
"""

import os
import warnings
import numpy as np
import pandas as pd
import datetime as dt
import re
from geopy.distance import geodesic
from AniMAIRE.MAIREPLUS_event import NeutronMonitorData, NM_pair, full_NM_set
from typing import Dict, List, Tuple, Any, Optional

# --- List of valid neutron monitor pairs for anisotropic analysis ---
valid_NM_pairs: List[List[str]] = [
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

# --- Utility functions ---
def check_monitors_in_dir_and_dict(directory_path: str) -> Dict[str, NeutronMonitorData]:
    """
    Scan a directory for neutron monitor data files and load them into NeutronMonitorData objects.
    """
    found_monitors_in_dir: Dict[str, NeutronMonitorData] = {}
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found - {directory_path}")
        return {}
    try:
        for filename in os.listdir(directory_path):
            # Only consider files that look like monitor data files (start with 'C', end with '.DAT')
            if filename.startswith("C") and filename.endswith(".DAT"):
                parts = filename[1:].split('.')
                if parts:
                    # Extract station code (remove digits and trailing '_cor')
                    station_code = ''.join([c for c in parts[0] if not c.isdigit()])
                    if station_code and not station_code.endswith('_cor'):
                        file_path = os.path.join(directory_path, filename)
                        # Load the monitor data from file using the NeutronMonitorData class
                        found_monitors_in_dir[station_code] = NeutronMonitorData.from_file(file_path)
    except Exception as e:
        print(f"Error accessing directory {directory_path}: {e}")
        return {}
    return found_monitors_in_dir

def get_valid_monitor_pairs(
    available_monitors: Dict[str, NeutronMonitorData],
    valid_pairs: List[List[str]]
) -> List[NM_pair]:
    """
    Filter the list of valid monitor pairs to those for which both monitors are available and have data.
    """
    valid_available_pairs: List[NM_pair] = []
    for pair in valid_pairs:
        primary, secondary = pair[0], pair[1]
        # Check both monitors are available and have data
        if primary in available_monitors and secondary in available_monitors:
            if available_monitors[primary].empty or available_monitors[secondary].empty:
                # Skip pairs where either monitor has no data
                warnings.warn(f"Skipping pair {primary}-{secondary} because one or both monitors have empty data")
                continue
            # Create NM_pair object for this valid pair
            nm_pair = NM_pair(available_monitors[primary], available_monitors[secondary])
            valid_available_pairs.append(nm_pair)
    return valid_available_pairs

def find_nearest_location_for_monitors(
    monitors: Dict[str, NeutronMonitorData],
    locations: List[Tuple[float, float]]
) -> Dict[str, Dict[str, Any]]:
    """
    For each monitor, find the nearest location from a list of locations (typically midpoints of monitor pairs).
    """
    result: Dict[str, Dict[str, Any]] = {}
    for monitor_code, monitor_data in monitors.items():
        if monitor_data.empty:
            print(f"Warning: Monitor {monitor_code} has empty data")
            continue
        try:
            # Get the latitude and longitude of the monitor
            monitor_lat, monitor_lon, monitor_alt = monitor_data.get_location()
        except (KeyError, IndexError):
            print(f"Warning: Could not find location data for monitor {monitor_code}")
            continue
        min_distance = float('inf')
        nearest_location: Optional[Tuple[float, float]] = None
        for location in locations:
            lat, lon = location[0], location[1]
            # Compute geodesic distance between monitor and location
            distance = geodesic((monitor_lat, monitor_lon), (lat, lon)).kilometers
            if distance < min_distance:
                min_distance = distance
                nearest_location = location
        # Store the nearest location and distance for this monitor
        result[monitor_code] = {
            'nearest_location': nearest_location,
            'distance_km': min_distance
        }
    return result

def get_list_of_monitor_sets_to_run_across(
    list_of_available_monitors: Dict[str, NeutronMonitorData],
    valid_available_pairs: List[NM_pair],
    nearest_locations: Dict[str, Dict[str, Any]]
) -> List[full_NM_set]:
    """
    Assemble full_NM_set objects for each available monitor, using the nearest valid monitor pair as primary/secondary.
    """
    monitor_to_pair_mapping: Dict[str, NM_pair] = {}
    # For each monitor, find the NM_pair whose midpoint is nearest
    for monitor, nearest_info in nearest_locations.items():
        nearest_midpoint = nearest_info['nearest_location']
        for pair in valid_available_pairs:
            # Compare the midpoint of the pair to the nearest location for this monitor
            if pair.midpoint_location == nearest_midpoint:
                monitor_to_pair_mapping[monitor] = pair
                break  # Only map to the first matching pair
    if not valid_available_pairs:
        # If no valid pairs, raise an error and print debug info
        print("No valid monitor pairs found where both monitors are available.")
        print("Available monitors:", list(list_of_available_monitors.keys()))
        print("Valid pairs:", valid_NM_pairs)
        raise Exception("No valid monitor pairs found for anisotropic processing. Cannot continue.")
    list_of_monitor_sets_objects: List[full_NM_set] = []
    for monitor, pair in monitor_to_pair_mapping.items():
        monitor_data = list_of_available_monitors.get(monitor)
        if not monitor_data.empty:
            try:
                # Get the location for debugging or future use (not used directly here)
                monitor_location = list(monitor_data.get_location())
                # Create a full_NM_set with this monitor as normalisation, and the pair as primary/secondary
                monitor_set = full_NM_set(
                    primary=pair.primary,
                    secondary=pair.secondary,
                    normalisation=monitor_data,
                )
                list_of_monitor_sets_objects.append(monitor_set)
            except (KeyError, IndexError):
                # If location extraction fails, skip this monitor
                print(f"Warning: Could not extract location for monitor {monitor}")
        else:
            print(f"Warning: No data available for monitor {monitor}")
    return list_of_monitor_sets_objects

# --- Main user-facing function ---
def get_monitor_sets_from_directory(directory_path: str) -> List[full_NM_set]:
    """
    Discover and assemble all available monitor sets from a directory for use in event analysis.
    """
    # Step 1: Find all available monitors in the directory
    available_monitors: Dict[str, NeutronMonitorData] = check_monitors_in_dir_and_dict(directory_path)
    # Step 2: Find all valid pairs among available monitors
    valid_pairs: List[NM_pair] = get_valid_monitor_pairs(available_monitors, valid_NM_pairs)
    # Step 3: Get the midpoints of all valid pairs
    midpoints: List[Tuple[float, float]] = [pair.midpoint_location for pair in valid_pairs]
    # Step 4: For each monitor, find the nearest valid pair midpoint
    nearest_locations: Dict[str, Dict[str, Any]] = find_nearest_location_for_monitors(available_monitors, midpoints)
    # Step 5: Assemble full_NM_set objects for each monitor
    list_of_monitor_sets_objects: List[full_NM_set] = get_list_of_monitor_sets_to_run_across(available_monitors, valid_pairs, nearest_locations)
    return list_of_monitor_sets_objects 