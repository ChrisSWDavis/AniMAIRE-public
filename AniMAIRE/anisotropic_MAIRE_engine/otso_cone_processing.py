#!/bin/python3
import numpy as np
import pandas as pd
import datetime as dt
from typing import Optional
import OTSO
import logging
import psutil
# NOTE: This file is now deprecated but could be reused in the future.
# It contains functionality for processing OTSO cone output into asymptotic direction dataframes.
# The new functionality is in otso_planet_processing.py and uses the OTSO.planet() function.

def convert_to_asymp_df(cone_output: list[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert the OTSO.cone() output to a DataFrame with asymptotic directions.
    
    Parameters:
    -----------
    cone_output : list
        The output from OTSO.cone() containing asymptotic directions
        
    Returns:
    --------
    pd.DataFrame
        A DataFrame with columns:
        - initialLatitude: Float (starting latitude)
        - initialLongitude: Float (starting longitude)
        - Rigidity: Float (in GV)
        - Lat: Float (asymptotic direction latitude in degrees)
        - Long: Float (asymptotic direction longitude in degrees)
        - Filter: Integer (1 if valid, 0 if not)
    """
    # Extract the data frame from the cone output
    df_raw = cone_output[0]
    
    # Get the rigidity column (first column)
    rigidity_col = df_raw.columns[0]
    
    # Get all location columns (all columns except the first)
    location_cols = df_raw.columns[1:]
    
    # Create an empty list to store the rows
    rows = []
    
    # Iterate through each row in the dataframe
    for _, row in df_raw.iterrows():
        rigidity = row[rigidity_col]
        
        # Skip header rows that might contain asterisks
        if isinstance(rigidity, str) and '*' in rigidity:
            logging.warning(f"Skipping row with invalid rigidity value containing asterisk: {rigidity}. This likely indicates a rigidity value that was too large for Fortran's numeric format. Check your input rigidity range and consider using smaller values.")
            continue
        
        # Convert rigidity to float
        try:
            rigidity = float(rigidity)
        except (ValueError, TypeError):
            continue
        
        # Process each location column
        for loc_col in location_cols:
            # Extract the initial latitude and longitude from the column name
            # Format is typically "lat_long"
            if '_' in loc_col:
                try:
                    lat_str, long_str = loc_col.split('_')
                    init_lat = float(lat_str)
                    init_long = float(long_str)
                except (ValueError, TypeError):
                    # Skip if we can't parse the column name
                    continue
                
                # Parse the asymptotic direction string (format: "1;lat;long")
                asymp_dir = row[loc_col]
                if isinstance(asymp_dir, str) and ';' in asymp_dir:
                    try:
                        filter_val, asymp_lat, asymp_long = asymp_dir.split(';')
                        filter_val = int(filter_val)
                        asymp_lat = float(asymp_lat)
                        asymp_long = float(asymp_long)
                        
                        # Add the row to our list
                        rows.append({
                            'initialLatitude': init_lat,
                            'initialLongitude': init_long,
                            'Rigidity': rigidity,
                            'Lat': asymp_lat,
                            'Long': asymp_long,
                            'Filter': filter_val
                        })
                    except (ValueError, TypeError):
                        # Skip if we can't parse the asymptotic direction
                        continue
    
    # Create a DataFrame from the rows
    result_df = pd.DataFrame(rows)

    # Convert longitudes greater than 180 degrees to the range [-180, 0]
    # This is done by applying the transformation: if long > 180, then long = long - 360
    mask = result_df['Long'] > 180
    result_df.loc[mask, 'Long'] = result_df.loc[mask, 'Long'] - 360
    
    return result_df

def create_and_convert_cone(array_of_lats_and_longs:list[list[float,float]],
                            KpIndex:int,
                            dateAndTime:dt.datetime,
                            cache:bool,
                            full_output:bool,
                            array_of_zeniths_and_azimuths=[[0.0,0.0]],
                            max_rigidity=1010, 
                            min_rigidity=20, 
                            rigidity_step=16, 
                            corenum=7, 
                           **kwargs):
    """
    Create a cone of asymptotic directions using OTSO.cone() and convert it to a DataFrame format.
    
    Parameters:
    -----------
    max_rigidity : float, optional
        Maximum rigidity value, default 1010
    min_rigidity : float, optional
        Minimum rigidity value, default 20
    rigidity_step : float, optional
        Step size for rigidity, default 16
    corenum : int, optional
        Number of cores to use for calculation, default 7
    array_of_lats_and_lons : list or array, optional
        List of (lat, long) coordinates to use
        If None, a default world map grid will be generated with lat range (-90, 90),
        long range (0, 360), and steps of 5 degrees
    **kwargs : dict
        Additional parameters to pass to OTSO.cone()
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the asymptotic directions with columns:
        initialLatitude, initialLongitude, Rigidity, Lat, Long, Filter
    """
    
    # Generate grid points based on provided coordinates or default world map
    grid_points = []
    
    if array_of_lats_and_longs is not None:
        # Use the provided array of coordinates
        for lat, long in array_of_lats_and_longs:
            grid_points.append((f"{lat}_{long}", lat, long))
    else:
        # Generate default world map grid points
        lat_range = (-90, 90)
        long_range = (0, 360)
        lat_step = 5
        long_step = 5
        
        for lat in range(lat_range[0], lat_range[1] + 1, lat_step):
            for long in range(long_range[0], long_range[1] + 1, long_step):
                grid_points.append((f"{lat}_{long}", lat, long))

    print(grid_points)
    
    list_of_cone_results = []
    list_of_rigidity_results = []
    list_of_run_information = []
    for zenith, azimuth in array_of_zeniths_and_azimuths:
        # Calculate the cone of asymptotic directions
        cone_result, rigidity_result, run_information = OTSO.cone(
            customlocations=grid_points,
            corenum=corenum,
            startrigidity=max_rigidity,
            endrigidity=min_rigidity,
            rigiditystep=rigidity_step,
            kp=KpIndex,
            year=dateAndTime.year,
            month=dateAndTime.month,
            day=dateAndTime.day,
            hour=dateAndTime.hour,
            minute=dateAndTime.minute,
            second=dateAndTime.second,
            zenith=zenith,
            azimuth=azimuth,
            **kwargs
        )

        list_of_cone_results.append(cone_result)
        list_of_rigidity_results.append(rigidity_result)
        list_of_run_information.append(run_information)

    full_cone_result = pd.concat(list_of_cone_results)
    full_rigidity_result = pd.concat(list_of_rigidity_results)
    
    # Convert the result to a DataFrame
    converted_to_magcos_format_DF = convert_to_asymp_df([full_cone_result, full_rigidity_result,list_of_run_information])

    # Set filter value to 0 for all rows where filter is -1
    converted_to_magcos_format_DF.loc[converted_to_magcos_format_DF['Filter'] == -1, 'Filter'] = 0

    return converted_to_magcos_format_DF




def create_and_convert_full_cone(array_of_lats_and_longs:list[list[float,float]],
                            KpIndex:int,
                            dateAndTime:dt.datetime,
                            cache:bool,
                            full_output:bool,
                            array_of_zeniths_and_azimuths=[[0.0,0.0]],
                            highestMaxRigValue = 1010,
                            maxRigValue = 20,
                            minRigValue = 0.1,
                            nIncrements_high = 60,
                            nIncrements_low = 200,
                            corenum=psutil.cpu_count(logical=False) - 2, 
                           **kwargs):
    
    high_rigidity_step = (highestMaxRigValue - maxRigValue) / (nIncrements_high - 1)

    high_rigidity_cone_results = create_and_convert_cone(array_of_lats_and_longs, 
                                                         KpIndex, 
                                                         dateAndTime, 
                                                         cache, 
                                                         full_output, 
                                                         array_of_zeniths_and_azimuths, 
                                                         highestMaxRigValue, 
                                                         maxRigValue, 
                                                         high_rigidity_step, 
                                                         corenum, 
                                                         **kwargs)
    
    low_rigidity_step = (maxRigValue - minRigValue) / (nIncrements_low - 1)

    low_rigidity_cone_results = create_and_convert_cone(array_of_lats_and_longs, 
                                                        KpIndex, 
                                                        dateAndTime, 
                                                        cache, 
                                                        full_output, 
                                                        array_of_zeniths_and_azimuths, 
                                                        maxRigValue, 
                                                        minRigValue + low_rigidity_step, 
                                                        low_rigidity_step, 
                                                        corenum, 
                                                        **kwargs)

    return pd.concat([high_rigidity_cone_results, low_rigidity_cone_results]) 