"""
MAIREPLUS spectral index calculation tool.

This module provides functionality for calculating the MAIREPLUS spectral index
based on neutron monitor readings.
"""

import datetime as dt
from typing import Tuple

from .neutron_monitor import NeutronMonitor
from .calculate_spectral_index import calculate_spectral_index_for_target_ratio


def calculate_MAIREPLUS_spectral_index(
    neutron_monitor_1_location: Tuple[float, float, float],
    neutron_monitor_1_percentage_increase: float,
    neutron_monitor_2_location: Tuple[float, float, float],
    neutron_monitor_2_percentage_increase: float,
    datetime: dt.datetime,
    kp_index: float
) -> float:
    """
    Calculate the MAIREPLUS spectral index based on neutron monitor locations and percentage increases.

    This method uses two neutron monitors to calculate a spectral index based on the 
    ratio of their percentage increases. It automatically calculates the cutoff rigidities
    for the neutron monitors based on their locations, the provided datetime, and the Kp index.

    Parameters
    ----------
    neutron_monitor_1_location : Tuple[float, float, float]
        Location of the first neutron monitor as (latitude, longitude, altitude_in_km).
    neutron_monitor_1_percentage_increase : float
        Percentage increase observed by the first neutron monitor.
    neutron_monitor_2_location : Tuple[float, float, float]
        Location of the second neutron monitor as (latitude, longitude, altitude_in_km).
    neutron_monitor_2_percentage_increase : float
        Percentage increase observed by the second neutron monitor.
    datetime : dt.datetime
        Date and time for the calculation.
    kp_index : float
        Geomagnetic Kp index at the time of the calculation.

    Returns
    -------
    float
        The calculated MAIREPLUS spectral index.
    """
    target_ratio = neutron_monitor_1_percentage_increase / neutron_monitor_2_percentage_increase
    
    # Create neutron monitor objects
    neutron_monitor_1 = NeutronMonitor(
        latitude=neutron_monitor_1_location[0],
        longitude=neutron_monitor_1_location[1],
        altitude_in_km=neutron_monitor_1_location[2]
    )
    
    neutron_monitor_2 = NeutronMonitor(
        latitude=neutron_monitor_2_location[0],
        longitude=neutron_monitor_2_location[1],
        altitude_in_km=neutron_monitor_2_location[2]
    )
    
    # Calculate the cutoff rigidities
    neutron_monitor_1_cutoff_rigidity = neutron_monitor_1.calculate_vertical_cutoff_rigidity(
        datetime=datetime, 
        kp_index=kp_index
    )
    
    neutron_monitor_2_cutoff_rigidity = neutron_monitor_2.calculate_vertical_cutoff_rigidity(
        datetime=datetime, 
        kp_index=kp_index
    )
    
    # Calculate the spectral index
    return calculate_spectral_index_for_target_ratio(
        target_ratio=target_ratio,
        cut_off_rigidity_1=neutron_monitor_1_cutoff_rigidity,
        altitude_in_km_1=neutron_monitor_1.altitude_in_km,
        cut_off_rigidity_2=neutron_monitor_2_cutoff_rigidity,
        altitude_in_km_2=neutron_monitor_2.altitude_in_km
    ) 