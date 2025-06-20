import numpy as np
from typing import Optional, Tuple, Dict, Any
import OTSO
import datetime as dt


class NeutronMonitor:
    """
    A class representing a neutron monitor with its geographical location and methods
    to calculate vertical cut-off rigidity.
    """
    
    def __init__(
        self, 
        latitude: float, 
        longitude: float, 
        altitude_in_km: float,
        name: Optional[str] = "Unnamed_Neutron_Monitor",
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a neutron monitor with its geographical coordinates.
        
        Parameters:
        -----------
        latitude : float
            The latitude of the neutron monitor in degrees (-90 to 90).
        longitude : float
            The longitude east of the neutron monitor in degrees (0 to 360).
        altitude_in_km : float
            The altitude of the neutron monitor in kilometers.
        name : str, optional
            The name or identifier of the neutron monitor.
        additional_info : dict, optional
            Additional information about the neutron monitor.
        """
        # Validate inputs
        if not -90 <= latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90 degrees")
        if not 0 <= longitude <= 360:
            raise ValueError("Longitude east must be between 0 and 360 degrees")
        if altitude_in_km < 0:
            raise ValueError("Altitude must be non-negative")
        
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude_in_km = altitude_in_km
        self.additional_info = additional_info or {}
        
    def get_location(self) -> Tuple[float, float, float]:
        """
        Get the geographical coordinates of the neutron monitor.
        
        Returns:
        --------
        tuple
            A tuple containing (latitude, longitude, altitude_in_km).
        """
        return (self.latitude, self.longitude, self.altitude_in_km)
    
    def calculate_vertical_cutoff_rigidity(self, datetime: dt.datetime, kp_index: float, cutoff_type: str = "Rc") -> float:
        """
        Calculate the vertical cutoff rigidity for the neutron monitor location.
        
        Parameters:
        -----------
        datetime : datetime
            The datetime for the calculation.
        cutoff_type : str, optional
            The type of cutoff rigidity to calculate. Must be one of:
            - "Ru": Upper cutoff rigidity
            - "Rc": Effective cutoff rigidity
            - "Rl": Lower cutoff rigidity
            Default is "Re" (effective cutoff rigidity).
        kp_index : float
            The geomagnetic Kp index value. Used in the cutoff
            rigidity calculation to account for geomagnetic activity.
            
        Returns:
        --------
        float
            The calculated vertical cutoff rigidity in GV.
            
        Raises:
        -------
        ValueError
            If cutoff_type is not one of "Ru", "Rc", or "Rl".
        """
        if cutoff_type not in ["Ru", "Rc", "Rl"]:
            raise ValueError("cutoff_type must be one of 'Ru', 'Rc', or 'Rl'")
        
        cutoff_rigidities = OTSO.cutoff(
            Stations=[],
            customlocations=[[self.name, self.latitude, self.longitude]],
            year=datetime.year, 
            month=datetime.month, 
            day=datetime.day, 
            hour=datetime.hour, 
            minute=datetime.minute,
            kp=kp_index
        )[0]
        cutoff_rigidity = cutoff_rigidities.loc[cutoff_type].values[0]
        
        return cutoff_rigidity
    
    def __str__(self) -> str:
        """
        Return a string representation of the neutron monitor.
        
        Returns:
        --------
        str
            A string describing the neutron monitor.
        """
        name_str = f"name='{self.name}', " if self.name else ""
        return (f"NeutronMonitor({name_str}"
                f"latitude={self.latitude}°, longitude={self.longitude}°, "
                f"altitude={self.altitude_in_km} km)")
    
    def __repr__(self) -> str:
        """
        Return a string representation of the neutron monitor for debugging.
        
        Returns:
        --------
        str
            A string representation of the neutron monitor.
        """
        name_str = f"name='{self.name}', " if self.name else ""
        return (f"NeutronMonitor({name_str}"
                f"latitude={self.latitude}, longitude={self.longitude}, "
                f"altitude_in_km={self.altitude_in_km}, "
                f"additional_info={self.additional_info})")
