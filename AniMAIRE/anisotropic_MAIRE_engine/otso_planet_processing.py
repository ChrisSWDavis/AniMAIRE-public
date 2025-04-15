#!/bin/python3
import numpy as np
import pandas as pd
import datetime as dt
from typing import Optional
import OTSO
import logging
import ParticleRigidityCalculationTools as PRCT
from joblib import Memory
# Set up caching for Magnetocosmics run data
OTSOcachedir = 'cachedOTSOData'
OTSOmemory = Memory(OTSOcachedir, verbose=0)

def convert_planet_df_to_asymp_format(planet_df):
    """
    Convert the OTSO planet dataframe to the same format as the MagCos asymptotic directions dataframe.
    
    Parameters:
    -----------
    planet_df : tuple
        The output from OTSO.planet() with asymptotic directions
        
    Returns:
    --------
    pd.DataFrame
        A dataframe in the same format as magcos_asymp_dirs_DF
    """
    import pandas as pd
    
    # Extract the dataframe from the tuple
    df = planet_df[0]
    
    # Get all columns that contain energy values (asymptotic directions)
    energy_columns = [col for col in df.columns if '[GeV]' in str(col)]
    
    # Create an empty list to store the rows
    rows = []
    
    # Iterate through each row in the dataframe
    for _, row in df.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        
        # Process each energy column
        for energy_col in energy_columns:
            # Extract the energy value from the column name
            energy_value = float(energy_col.split(' ')[0])
            
            # Parse the asymptotic direction string (format: "1;lat;long")
            asymp_dir = row[energy_col]
            if isinstance(asymp_dir, str) and ';' in asymp_dir:
                filter_val, asymp_lat, asymp_long = asymp_dir.split(';')
                
                # Create a new row
                new_row = {
                    'initialLatitude': lat,
                    'initialLongitude': lon,
                    'Energy': energy_value,  # This is actually energy, will need conversion if rigidity is needed
                    'Lat': float(asymp_lat),
                    'Long': float(asymp_long),
                    'Filter': int(filter_val)
                }
                rows.append(new_row)
    
    # Create a dataframe from the rows
    result_df = pd.DataFrame(rows)
    result_df["Rigidity"] = PRCT.convertParticleEnergyToRigidity(result_df["Energy"]*1000.0,particleMassAU = 1,particleChargeAU = 1)
    
    # Convert energy to rigidity if needed
    # Uncomment and modify the following lines if conversion is needed
    # import ParticleRigidityCalculationTools as PRCT
    # result_df['Rigidity'] = PRCT.convertParticleEnergyToRigidity(
    #     result_df['Rigidity'] * 1000.0,  # Convert back to MeV
    #     particleMassAU=1,
    #     particleChargeAU=1
    # )
    
    return result_df.sort_values(by=["initialLatitude","initialLongitude"]).reset_index(drop=True)

def create_and_convert_planet(array_of_lats_and_longs:list[list[float,float]],
                            kpIndex:int,
                            dateAndTime:dt.datetime,
                            array_of_zeniths_and_azimuths=[[0.0,0.0]],
                            max_rigidity=1010, 
                            min_rigidity=20, 
                            rigidity_step=16, 
                            corenum=7, 
                            **kwargs):
    """
    Create asymptotic directions using OTSO.planet() and convert to a DataFrame format.
    
    Parameters:
    -----------
    array_of_lats_and_longs : list[list[float,float]]
        List of [latitude, longitude] coordinates to use
        If None, a default world map grid will be generated with lat range (-90, 90),
        long range (0, 360), and steps of 5 degrees
    kpIndex : int
        Kp index value for the magnetic field model
    dateAndTime : dt.datetime
        Date and time for the calculation
    array_of_zeniths_and_azimuths : list, optional
        List of [zenith, azimuth] pairs, default [[0.0, 0.0]]
    max_rigidity : float, optional
        Maximum rigidity value in GV, default 1010
    min_rigidity : float, optional
        Minimum rigidity value in GV, default 20
    rigidity_step : float, optional
        Step size for rigidity in GV, default 16
    corenum : int, optional
        Number of cores to use for calculation, default 7
    **kwargs : dict
        Additional parameters to pass to OTSO.planet()
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the asymptotic directions with columns:
        initialLatitude, initialLongitude, Rigidity, Lat, Long, Filter, zenith, azimuth
    """
    
    # Create rigidity levels for asymptotic directions
    rigidity_levels = []
    current_rigidity = max_rigidity
    while current_rigidity >= min_rigidity:
        rigidity_levels.append(current_rigidity)
        current_rigidity -= rigidity_step
    
    all_results = []
    
    # Loop over all zenith and azimuth pairs
    for zenith_azimuth in array_of_zeniths_and_azimuths:
        zenith, azimuth = zenith_azimuth
        
        # Calculate asymptotic directions using OTSO.planet
        planet_result = OTSO.planet(
            array_of_lats_and_longs=array_of_lats_and_longs,
            corenum=corenum,
            asymptotic="YES",
            asymlevels=rigidity_levels,
            kp=kpIndex,
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
        
        # Convert the result to a DataFrame
        result_df = convert_planet_df_to_asymp_format(planet_result)
        
        # Add zenith and azimuth columns to identify the viewing direction
        result_df['zenith'] = zenith
        result_df['azimuth'] = azimuth
        
        all_results.append(result_df)
    
    # Combine all results into a single DataFrame
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()  # Return empty DataFrame if no results

def create_and_convert_full_planet(array_of_lats_and_longs:list[list[float,float]],
                            KpIndex:int,
                            dateAndTime:dt.datetime,
                            cache:bool,
                            full_output=False,
                            array_of_zeniths_and_azimuths=[[0.0,0.0]],
                            highestMaxRigValue = 1010,
                            maxRigValue = 20,
                            minRigValue = 0.1,
                            nIncrements_high = 60,
                            nIncrements_low = 200,
                            corenum=7, 
                           **kwargs):
    
    high_rigidity_step = (highestMaxRigValue - maxRigValue) / (nIncrements_high - 1)
    low_rigidity_step = (maxRigValue - minRigValue) / (nIncrements_low - 1)

    # Function to call with or without caching based on cache parameter
    create_convert_func = OTSOmemory.cache(create_and_convert_planet) if cache else create_and_convert_planet
    
    high_rigidity_planet_results = create_convert_func(array_of_lats_and_longs, 
                                                      KpIndex, 
                                                      dateAndTime,
                                                      array_of_zeniths_and_azimuths, 
                                                      highestMaxRigValue, 
                                                      maxRigValue, 
                                                      high_rigidity_step, 
                                                      corenum, 
                                                      **kwargs)
    
    low_rigidity_planet_results = create_convert_func(array_of_lats_and_longs, 
                                                     KpIndex, 
                                                     dateAndTime,
                                                     array_of_zeniths_and_azimuths, 
                                                     maxRigValue, 
                                                     minRigValue + low_rigidity_step, 
                                                     low_rigidity_step, 
                                                     corenum, 
                                                     **kwargs)

    return pd.concat([high_rigidity_planet_results, low_rigidity_planet_results]) 