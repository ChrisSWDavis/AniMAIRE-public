import numpy as np
import datetime as dt
from typing import Callable, Tuple

from .maire_plus_tools.calculate_MAIREPLUS_spectral_index import calculate_MAIREPLUS_spectral_index

from .anisotropic_MAIRE_engine.spectralCalculations.rigiditySpectrum import PowerLawSpectrum
# Import directly to avoid circular imports
from .anisotropic_MAIRE_engine.spectralCalculations.pitchAngleDistribution import IsotropicPitchAngleDistribution

# Forward declaration to avoid circular imports
run_from_DLR_cosmic_ray_model = None
run_from_spectra = None

def _set_function_references(DLR_func, spectra_func):
    """Set function references to avoid circular imports."""
    global run_from_DLR_cosmic_ray_model, run_from_spectra
    run_from_DLR_cosmic_ray_model = DLR_func
    run_from_spectra = spectra_func

def calculate_MAIREPLUS_normalisation(spectral_index: float,
                                      neutron_monitor_location: Tuple[float, float, float],
                                      neutron_monitor_percentage_increase: float,
                                      OULU_gcr_count_rate_in_seconds: float,
                                      datetime: dt.datetime, 
                                      kp_index: float,
                                      dose_type_to_normalise_to: str = "tn3",
                                      test_factor: float = 1.0):
  
  global run_from_DLR_cosmic_ray_model, run_from_spectra
  if run_from_DLR_cosmic_ray_model is None or run_from_spectra is None:
      raise RuntimeError("Function references not set. Call _set_function_references first.")
  
  gcr_dose_rate_at_neutron_monitor = run_from_DLR_cosmic_ray_model(
        OULU_count_rate_in_seconds=OULU_gcr_count_rate_in_seconds,
        date_and_time=datetime,
        array_of_lats_and_longs=[[neutron_monitor_location[0],neutron_monitor_location[1]]],
        altitudes_in_km=[neutron_monitor_location[2]],
        Kp_index=kp_index,
        proton_pitch_angle_distribution=IsotropicPitchAngleDistribution(use_fast_calculation=True),
        alpha_pitch_angle_distribution=IsotropicPitchAngleDistribution(use_fast_calculation=True),
      )[dose_type_to_normalise_to].values[0]
    
  predicted_dose_rate_at_neutron_monitor = run_from_spectra(
        proton_rigidity_spectrum=lambda x:test_factor*(x**-spectral_index),
        Kp_index=kp_index,
        date_and_time=datetime,
        array_of_lats_and_longs=[[neutron_monitor_location[0],neutron_monitor_location[1]]],
        altitudes_in_km=[neutron_monitor_location[2]],
        proton_pitch_angle_distribution=IsotropicPitchAngleDistribution(use_fast_calculation=True),
        alpha_pitch_angle_distribution=IsotropicPitchAngleDistribution(use_fast_calculation=True),
  )[dose_type_to_normalise_to].values[0]

  predicted_dose_rate_percentage_increase = 100 * predicted_dose_rate_at_neutron_monitor / gcr_dose_rate_at_neutron_monitor

  return neutron_monitor_percentage_increase / predicted_dose_rate_percentage_increase

class MAIREPLUS_spectrum(PowerLawSpectrum):
  def __init__(self, neutron_monitor_1_location: Tuple[float, float, float],
               neutron_monitor_1_percentage_increase: float,
               neutron_monitor_2_location: Tuple[float, float, float],
               neutron_monitor_2_percentage_increase: float,
               normalisation_monitor_location: Tuple[float, float, float],
               normalisation_monitor_percentage_increase: float,
               OULU_gcr_count_rate_in_seconds: float,
               datetime: dt.datetime,
               kp_index: float):

    self.MAIREPLUS_spectral_index = calculate_MAIREPLUS_spectral_index(neutron_monitor_1_location=neutron_monitor_1_location,
                                                        neutron_monitor_1_percentage_increase=neutron_monitor_1_percentage_increase,
                                                        neutron_monitor_2_location=neutron_monitor_2_location,
                                                        neutron_monitor_2_percentage_increase=neutron_monitor_2_percentage_increase,
                                                        datetime=datetime,
                                                        kp_index=kp_index)

    self.normalisation_factor = calculate_MAIREPLUS_normalisation(spectral_index=self.MAIREPLUS_spectral_index,
                                                              neutron_monitor_location=normalisation_monitor_location,
                                                              neutron_monitor_percentage_increase=normalisation_monitor_percentage_increase,
                                                              OULU_gcr_count_rate_in_seconds=OULU_gcr_count_rate_in_seconds,
                                                              datetime=datetime,
                                                              kp_index=kp_index)

    super().__init__(normalisationFactor=self.normalisation_factor, 
                     spectralIndex=-self.MAIREPLUS_spectral_index)
    
  def plot(self, ax=None, **kwargs):
    """
    Plot the rigidity spectrum and annotate with normalisation factor and spectral index.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.
    **kwargs : dict
        Additional keyword arguments to pass to the parent plot method.
        
    Returns:
    --------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    # Call the parent class plot method
    ax = super().plot(ax=ax, **kwargs)
    
    # Add annotation with normalisation factor and spectral index
    annotation_text = (f"Normalisation Factor: {self.normalisation_factor:.3e}\n"
                       f"Spectral Index: {self.MAIREPLUS_spectral_index:.3f}")
    
    # Position the annotation at the bottom of the plot
    ax.annotate(annotation_text, 
                xy=(0.5, 0.01), 
                xycoords='figure fraction',
                ha='center', 
                va='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7))
    
    return ax
