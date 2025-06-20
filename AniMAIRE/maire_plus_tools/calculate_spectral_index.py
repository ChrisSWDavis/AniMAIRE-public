import numpy as np
from scipy.optimize import curve_fit
from pynverse import inversefunc
from CosRayModifiedISO import CosRayModifiedISO
from scipy.interpolate import interp1d
from atmosphericRadiationDoseAndFlux import doseAndFluxCalculator as DAFcalc
import pandas as pd

def calculate_GCR_total_doses(vertical_cut_off_rigidity: float, altitude_in_km: float | list[float], solar_modulation: float = 100.0, dose_label_to_use: str = "tn3") -> pd.DataFrame:
    """
    Calculate the total Galactic Cosmic Ray (GCR) dose rates at a given altitude and vertical cut-off rigidity.

    Parameters:
    vertical_cut_off_rigidity (float): The vertical cut-off rigidity in GV.
    altitude_in_km (float or list of floats): The altitude(s) in kilometers where the dose rates are to be calculated.
    solar_modulation (float, optional): The solar modulation parameter. Default is 100.0.
    dose_label_to_use (str, optional): The dose label to use for the output DataFrame. Default is "tn3".

    Returns:
    pandas.DataFrame: A DataFrame containing the total dose rates for protons and alpha particles.
    """
    # Get proton spectrum and interpolate
    proton_spectrum = CosRayModifiedISO.getSpectrumUsingSolarModulation(solar_modulation, atomicNumber=1)
    proton_spectrum_interpolated = interp1d(
        proton_spectrum['Rigidity (GV/n)'], 
        proton_spectrum['d_Flux / d_R (cm-2 s-1 sr-1 (GV/n)-1)'], 
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )
    proton_spectrum_interpolated_vcutoff = lambda x: proton_spectrum_interpolated(x) if x >= vertical_cut_off_rigidity else 0.0

    # Get alpha particle spectrum and interpolate
    alpha_spectrum = CosRayModifiedISO.getSpectrumUsingSolarModulation(solar_modulation, atomicNumber=2)
    alpha_spectrum_interpolated = interp1d(
        alpha_spectrum['Rigidity (GV/n)'], 
        alpha_spectrum['d_Flux / d_R (cm-2 s-1 sr-1 (GV/n)-1)'], 
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )
    alpha_spectrum_interpolated_vcutoff = lambda x: alpha_spectrum_interpolated(x) if x >= vertical_cut_off_rigidity / 4.0 else 0.0

    # Calculate dose rates for protons
    proton_doses = DAFcalc.calculate_from_rigidity_spec(
        proton_spectrum_interpolated_vcutoff, 
        particleName="proton",
        altitudesInkm=altitude_in_km
    )

    # Calculate dose rates for alpha particles
    alpha_doses = DAFcalc.calculate_from_rigidity_spec(
        alpha_spectrum_interpolated_vcutoff, 
        particleName="alpha",
        altitudesInkm=altitude_in_km
    )

    # Add all dose rates together
    total_doses = proton_doses.copy()
    for column in ['edose', 'adose', 'dosee', 'tn1', 'tn2', 'tn3', 'SEU', 'SEL']:
        total_doses[column] += alpha_doses[column]
    
    total_doses.attrs['proton_doses'] = proton_doses
    total_doses.attrs['alpha_doses'] = alpha_doses

    return total_doses

@np.vectorize
def determine_single_tn(spectral_index: float, vertical_cut_off_rigidity: float, altitude_in_km: float, dose_label_to_use: str = "tn3") -> float:
    """
    Determines the tn3 dose for a given spectral index, vertical cut-off rigidity, and altitude.

    Parameters:
    spectral_index (float): The spectral index used in the calculations.
    vertical_cut_off_rigidity (float): The vertical cut-off rigidity.
    altitude_in_km (float): The altitude in kilometers.
    dose_label_to_use (str, optional): The dose label to use for the output DataFrame. Default is "tn3".

    Returns:
    float: The tn3 dose value.
    """
    A = 1
    rigiditySpectrum = lambda x: A * (x ** (-spectral_index)) if x >= vertical_cut_off_rigidity else 0.0

    output_doses = DAFcalc.calculate_from_rigidity_spec(
        rigiditySpectrum, 
        particleName="proton",
        altitudesInkm=altitude_in_km
    )
    
    output_doses["spectral_index"] = spectral_index
    output_doses["vertical_cut_off_rigidity"] = vertical_cut_off_rigidity

    return output_doses[dose_label_to_use].values[0]

def determine_percentage_increase_ratio(
    cut_off_rigidity_1: float, 
    altitude_in_km_1: float, 
    cut_off_rigidity_2: float, 
    altitude_in_km_2: float, 
    spectral_index: float,
    dose_label_to_use: str = "tn3"
) -> float:
    """
    Determines the ratio of percentage increases in total doses for two different sets of cut-off rigidity and altitude values.

    Parameters:
    cut_off_rigidity_1 (float): The cut-off rigidity for the first set of conditions.
    altitude_in_km_1 (float): The altitude in kilometers for the first set of conditions.
    cut_off_rigidity_2 (float): The cut-off rigidity for the second set of conditions.
    altitude_in_km_2 (float): The altitude in kilometers for the second set of conditions.
    spectral_index (float): The spectral index used in the calculations.
    dose_label_to_use (str, optional): The dose label to use for the output DataFrame. Default is "tn3".

    Returns:
    float: The ratio of the percentage increase in total doses between the two sets of conditions.
    """
    tn3_1 = determine_single_tn(spectral_index, cut_off_rigidity_1, altitude_in_km_1, dose_label_to_use)
    GCR_tn3_1 = calculate_GCR_total_doses(cut_off_rigidity_1, altitude_in_km_1, dose_label_to_use=dose_label_to_use)
    percent_increase_1 = tn3_1 / GCR_tn3_1[dose_label_to_use].values[0]

    tn3_2 = determine_single_tn(spectral_index, cut_off_rigidity_2, altitude_in_km_2, dose_label_to_use)
    GCR_tn3_2 = calculate_GCR_total_doses(cut_off_rigidity_2, altitude_in_km_2, dose_label_to_use=dose_label_to_use)
    percent_increase_2 = tn3_2 / GCR_tn3_2[dose_label_to_use].values[0]
    
    return percent_increase_1 / percent_increase_2

def calculate_spectral_index_for_target_ratio(
    target_ratio: float, 
    cut_off_rigidity_1: float, 
    altitude_in_km_1: float, 
    cut_off_rigidity_2: float, 
    altitude_in_km_2: float,
    dose_label_to_use: str = "tn3"
) -> float:
    """
    Calculate the spectral index for a target ratio of percentage increases in total doses.

    Parameters:
    target_ratio (float): The target ratio of percentage increases in total doses.
    cut_off_rigidity_1 (float): The cut-off rigidity for the first neutron monitor.
    altitude_in_km_1 (float): The altitude in kilometers for the first neutron monitor.
    cut_off_rigidity_2 (float): The cut-off rigidity for the second neutron monitor.
    altitude_in_km_2 (float): The altitude in kilometers for the second neutron monitor.

    Returns:
    float: The spectral index that achieves the target ratio.
    """
    percentage_increase_function = lambda spec_index: determine_percentage_increase_ratio(
        cut_off_rigidity_1,
        altitude_in_km_1,
        cut_off_rigidity_2,
        altitude_in_km_2,
        spec_index,
        dose_label_to_use
    )
    inverse_percentage_increase_function = inversefunc(percentage_increase_function)
    
    return float(inverse_percentage_increase_function(target_ratio))
