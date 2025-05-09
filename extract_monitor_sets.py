import os
import warnings
import numpy as np
import pandas as pd
import datetime as dt
import re
from geopy.distance import geodesic
from AniMAIRE.MAIREPLUS_event import NeutronMonitorData, NM_pair, full_NM_set

# --- valid_NM_pairs ---
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

# --- Utility functions ---
def check_monitors_in_dir_and_dict(directory_path):
    found_monitors_in_dir = {}
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found - {directory_path}")
        return {}
    try:
        for filename in os.listdir(directory_path):
            if filename.startswith("C") and filename.endswith(".DAT"):
                parts = filename[1:].split('.')
                if parts:
                    station_code = ''.join([c for c in parts[0] if not c.isdigit()])
                    if station_code and not station_code.endswith('_cor'):
                        file_path = os.path.join(directory_path, filename)
                        found_monitors_in_dir[station_code] = NeutronMonitorData.from_file(file_path)
    except Exception as e:
        print(f"Error accessing directory {directory_path}: {e}")
        return {}
    return found_monitors_in_dir

def get_valid_monitor_pairs(available_monitors, valid_pairs):
    valid_available_pairs = []
    for pair in valid_pairs:
        primary, secondary = pair[0], pair[1]
        if primary in available_monitors and secondary in available_monitors:
            if available_monitors[primary].empty or available_monitors[secondary].empty:
                warnings.warn(f"Skipping pair {primary}-{secondary} because one or both monitors have empty data")
                continue
            nm_pair = NM_pair(available_monitors[primary], available_monitors[secondary])
            valid_available_pairs.append(nm_pair)
    return valid_available_pairs

def find_nearest_location_for_monitors(monitors, locations):
    result = {}
    for monitor_code, monitor_data in monitors.items():
        if monitor_data.empty:
            print(f"Warning: Monitor {monitor_code} has empty data")
            continue
        try:
            monitor_lat, monitor_lon, monitor_alt = monitor_data.get_location()
        except (KeyError, IndexError):
            print(f"Warning: Could not find location data for monitor {monitor_code}")
            continue
        min_distance = float('inf')
        nearest_location = None
        for location in locations:
            lat, lon = location[0], location[1]
            distance = geodesic((monitor_lat, monitor_lon), (lat, lon)).kilometers
            if distance < min_distance:
                min_distance = distance
                nearest_location = location
        result[monitor_code] = {
            'nearest_location': nearest_location,
            'distance_km': min_distance
        }
    return result

def get_list_of_monitor_sets_to_run_across(list_of_available_monitors, valid_available_pairs, nearest_locations):
    monitor_to_pair_mapping = {}
    for monitor, nearest_info in nearest_locations.items():
        nearest_midpoint = nearest_info['nearest_location']
        for pair in valid_available_pairs:
            if pair.midpoint_location == nearest_midpoint:
                monitor_to_pair_mapping[monitor] = pair
                break
    if not valid_available_pairs:
        print("No valid monitor pairs found where both monitors are available.")
        print("Available monitors:", list(list_of_available_monitors.keys()))
        print("Valid pairs:", valid_NM_pairs)
        raise Exception("No valid monitor pairs found for anisotropic processing. Cannot continue.")
    list_of_monitor_sets_objects = []
    for monitor, pair in monitor_to_pair_mapping.items():
        monitor_data = list_of_available_monitors.get(monitor)
        if not monitor_data.empty:
            try:
                monitor_location = list(monitor_data.get_location())
                monitor_set = full_NM_set(
                    primary=pair.primary,
                    secondary=pair.secondary,
                    normalisation=monitor_data,
                )
                list_of_monitor_sets_objects.append(monitor_set)
            except (KeyError, IndexError):
                print(f"Warning: Could not extract location for monitor {monitor}")
        else:
            print(f"Warning: No data available for monitor {monitor}")
    return list_of_monitor_sets_objects

# --- Main user-facing function ---
def get_monitor_sets_from_directory(directory_path):
    """
    Given a directory path, return the list_of_monitor_sets_objects for all available monitors.
    """
    available_monitors = check_monitors_in_dir_and_dict(directory_path)
    valid_pairs = get_valid_monitor_pairs(available_monitors, valid_NM_pairs)
    midpoints = [pair.midpoint_location for pair in valid_pairs]
    nearest_locations = find_nearest_location_for_monitors(available_monitors, midpoints)
    list_of_monitor_sets_objects = get_list_of_monitor_sets_to_run_across(available_monitors, valid_pairs, nearest_locations)
    return list_of_monitor_sets_objects 