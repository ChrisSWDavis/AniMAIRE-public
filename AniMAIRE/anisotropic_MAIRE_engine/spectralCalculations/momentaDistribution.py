import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .pitchAngleDistribution import pitchAngleDistribution
from .rigiditySpectrum import rigiditySpectrum

def jacobian_function_to_use(pitch_angle_in_radians: float) -> float:
    try:
        output_Jacobian_factor = 1 / np.sin(2.0 * pitch_angle_in_radians)
    except ZeroDivisionError:
        output_Jacobian_factor = 0.0
    return output_Jacobian_factor

class momentaDistribution():
    """
    Class representing a momenta distribution.
    """

    def __init__(self, rigidity_spectrum: rigiditySpectrum, pitch_angle_distribution: pitchAngleDistribution):
        """
        Initialize the momenta distribution.

        Parameters:
        - rigidity_spectrum: rigiditySpectrum
            The rigidity spectrum.
        - pitch_angle_distribution: pitchAngleDistribution
            The pitch angle distribution.
        """
        self.setRigiditySpectrum(rigidity_spectrum)
        self.setPitchAngleDistribution(pitch_angle_distribution)

    def plot_spectrum_and_pad(self, figsize=(12, 5), 
                          min_rigidity=0.1, max_rigidity=20, 
                          reference_rigidity=1.0):
        """
        Plot the rigidity spectrum and pitch angle distribution.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches (default: (12, 5))
        min_rigidity : float, optional
            Minimum rigidity in GV for spectrum plot (default: 0.1)
        max_rigidity : float, optional
            Maximum rigidity in GV for spectrum plot (default: 20)
        reference_rigidity : float, optional
            Reference rigidity in GV for pitch angle distribution (default: 1.0)
        
        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the plots
        """
        
        # Create a figure with two subplots
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, figure=fig)
        
        # Plot the spectrum using the dedicated function
        ax1 = fig.add_subplot(gs[0, 0])
        self.rigidity_spectrum.plot(ax=ax1, min_rigidity=min_rigidity, max_rigidity=max_rigidity)
        
        # Plot the pitch angle distribution using the dedicated function
        ax2 = fig.add_subplot(gs[0, 1])
        self.pitch_angle_distribution.plot(ax=ax2, reference_rigidity=reference_rigidity)
        
        plt.tight_layout()
        return fig

    # setter methods

    def setPitchAngleDistribution(self, pitch_angle_distribution: pitchAngleDistribution):
        """
        Set the pitch angle distribution.

        Parameters:
        - pitch_angle_distribution: pitchAngleDistribution
            The pitch angle distribution.
        """
        self.pitch_angle_distribution = pitch_angle_distribution

    def setRigiditySpectrum(self, rigidity_spectrum: callable):
        """
        Set the rigidity spectrum.

        Parameters:
        - rigidity_spectrum: rigiditySpectrum
            The rigidity spectrum.
        """
        # Check if rigidity_spectrum is a rigiditySpectrum instance or inherits from it
        if not (isinstance(rigidity_spectrum, rigiditySpectrum) or 
                (isinstance(rigidity_spectrum, type) and issubclass(rigidity_spectrum, rigiditySpectrum))):
            # If it's a callable but not a rigiditySpectrum instance, wrap it
            rigidity_spectrum = rigiditySpectrum(rigidity_spectrum)

    
        self.rigidity_spectrum = rigidity_spectrum

    # getter methods

    def getPitchAngleDistribution(self) -> pitchAngleDistribution:
        """
        Get the pitch angle distribution.

        Returns:
        - pitchAngleDistribution
            The pitch angle distribution.
        """
        return self.pitch_angle_distribution

    def getRigiditySpectrum(self) -> rigiditySpectrum:
        """
        Get the rigidity spectrum.

        Returns:
        - rigiditySpectrum
            The rigidity spectrum.
        """
        return self.rigidity_spectrum

    # other Methods

    def __call__(self, pitchAngle: float, rigidity: float) -> float:
        """
        Evaluate the momenta distribution.

        Parameters:
        - pitchAngle: float
            The pitch angle.
        - rigidity: float
            The rigidity.

        Returns:
        - float
            The value of the momenta distribution.
        """
        pitch_angle_weighting_factor = self.pitch_angle_distribution(pitchAngle, rigidity) #* jacobian_function_to_use(pitchAngle)
        rigidity_weighting_factor = self.rigidity_spectrum(rigidity)

        full_rigidity_pitch_weighting_factor = pitch_angle_weighting_factor * rigidity_weighting_factor

        return full_rigidity_pitch_weighting_factor