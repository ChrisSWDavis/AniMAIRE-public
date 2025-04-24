import numpy as np
from spacepy.coordinates import Coords as spaceCoords
import copy
import matplotlib.pyplot as plt
from typing import Optional, Callable, Any, Tuple
from .utils import Distribution, SummedFunction, ScaledFunction

class PitchAngleDistribution(Distribution[float]):
    """
    Base class for pitch angle distributions.
    """

    def __init__(self, 
                 pitch_angle_distribution: Optional[Callable[[float, float], float]] = None, 
                 reference_latitude_in_GSM: float = 0.0, 
                 reference_longitude_in_GSM: float = 0.0):
        """
        Initialize the pitch angle distribution.

        Parameters:
        - pitch_angle_distribution: callable, optional
            Function describing the pitch angle distribution.
        - reference_latitude_in_GSM: float
            Reference latitude in GSM coordinates.
        - reference_longitude_in_GSM: float
            Reference longitude in GSM coordinates.
        """
        self.pitchAngleDistFunction = pitch_angle_distribution or self.evaluate

        self.interplanetary_mag_field = spaceCoords([100.0,
                                                     reference_latitude_in_GSM, 
                                                     reference_longitude_in_GSM],
                                                     "GSM","sph")
    
    def evaluate(self, pitchAngle: float, rigidity: float) -> float:
        """
        Default evaluate method, should be overridden by subclasses.
        
        Args:
            pitchAngle: The pitch angle in radians
            rigidity: The rigidity in GV
            
        Returns:
            float: The value of the pitch angle distribution
        """
        raise NotImplementedError("Subclasses must implement evaluate method")

    def __call__(self, pitchAngle: float, rigidity: float) -> float:
        """
        Evaluate the pitch angle distribution at a given pitch angle and rigidity.

        Parameters:
        - pitchAngle: float
            The pitch angle.
        - rigidity: float
            The rigidity.

        Returns:
        - float
            The value of the pitch angle distribution.
        """
        return self.pitchAngleDistFunction(pitchAngle, rigidity)
    
    def __add__(self, right: 'PitchAngleDistribution') -> 'PitchAngleDistribution':
        """
        Add two pitch angle distributions.

        Parameters:
        - right: PitchAngleDistribution
            The other pitch angle distribution.

        Returns:
        - PitchAngleDistribution
            The sum of the two pitch angle distributions.
        """
        summed_dist = copy.deepcopy(self)
        summed_dist.pitchAngleDistFunction = SummedFunction(self.pitchAngleDistFunction, right.pitchAngleDistFunction)
        return summed_dist
    
    def plot(self, title: Optional[str] = None, reference_rigidity: float = 1.0, 
             ax: Optional[plt.Axes] = None, **kwargs: Any) -> plt.Axes:
        """
        Plot the pitch angle distribution.
        
        Parameters:
        -----------
        title : str, optional
            Title for the plot. If None, a default title is used.
        reference_rigidity : float, optional
            Reference rigidity in GV for pitch angle distribution (default: 1.0)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        
        Returns:
        --------
        matplotlib.axes.Axes
            The axes containing the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        
        alpha_range = np.linspace(0, np.pi, 100)  # radians
        pad_values = [self.pitchAngleDistFunction(a, reference_rigidity) for a in alpha_range]
        ax.plot(alpha_range, pad_values, **kwargs)
        ax.set_xlabel('Pitch Angle (radians)')
        ax.set_ylabel('Relative Intensity')
        ax.set_title('Pitch Angle Distribution' if title is None else title)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(0, np.pi)
        
        return ax
    
    def __mul__(self, right: float) -> 'PitchAngleDistribution':
        """
        Multiply the pitch angle distribution by a scalar.

        Parameters:
        - right: float
            The scalar.

        Returns:
        - PitchAngleDistribution
            The scaled pitch angle distribution.
        """
        multiplied_dist = copy.deepcopy(self)
        multiplied_dist.pitchAngleDistFunction = ScaledFunction(self.pitchAngleDistFunction, right)
        return multiplied_dist

    __rmul__ = __mul__


class CosinePitchAngleDistribution(PitchAngleDistribution):
    """
    Cosine pitch angle distribution.
    """

    def __init__(self, reference_latitude_in_GSM: float = 0.0, reference_longitude_in_GSM: float = 0.0):
        """
        Initialize the cosine pitch angle distribution.
        
        Parameters:
        - reference_latitude_in_GSM: float
            Reference latitude in GSM coordinates.
        - reference_longitude_in_GSM: float
            Reference longitude in GSM coordinates.
        """
        super().__init__(None, reference_latitude_in_GSM, reference_longitude_in_GSM)
    
    def evaluate(self, pitchAngle: float, rigidity: float) -> float:
        """
        Evaluate the cosine pitch angle distribution.
        
        Args:
            pitchAngle: The pitch angle in radians
            rigidity: The rigidity in GV
            
        Returns:
            float: The value of the cosine pitch angle distribution
        """
        return np.abs(0.5 * np.sin(2 * pitchAngle))


class IsotropicPitchAngleDistribution(PitchAngleDistribution):
    """
    Isotropic pitch angle distribution.
    """

    def __init__(self, reference_latitude_in_GSM: float = 0.0, reference_longitude_in_GSM: float = 0.0):
        """
        Initialize the isotropic pitch angle distribution.
        
        Parameters:
        - reference_latitude_in_GSM: float
            Reference latitude in GSM coordinates.
        - reference_longitude_in_GSM: float
            Reference longitude in GSM coordinates.
        """
        super().__init__(None, reference_latitude_in_GSM, reference_longitude_in_GSM)
    
    def evaluate(self, pitchAngle: float, rigidity: float) -> float:
        """
        Evaluate the isotropic pitch angle distribution.
        
        Args:
            pitchAngle: The pitch angle in radians
            rigidity: The rigidity in GV
            
        Returns:
            float: The value of the isotropic pitch angle distribution (always 1)
        """
        return 1


class GaussianPitchAngleDistribution(PitchAngleDistribution):
    """
    Gaussian pitch angle distribution.
    """

    def __init__(self, normFactor: float, sigma: float, alpha: float = 0.0, 
                 reference_latitude_in_GSM: float = 0.0, reference_longitude_in_GSM: float = 0.0):
        """
        Initialize the Gaussian pitch angle distribution.

        Parameters:
        - normFactor: float
            The normalization factor.
        - sigma: float
            The standard deviation of the Gaussian distribution.
        - alpha: float, optional
            The mean of the Gaussian distribution.
        - reference_latitude_in_GSM: float
            Reference latitude in GSM coordinates.
        - reference_longitude_in_GSM: float
            Reference longitude in GSM coordinates.
        """
        super().__init__(None, reference_latitude_in_GSM, reference_longitude_in_GSM)
        self.normFactor = normFactor
        self.sigma = sigma
        self.alpha = alpha
    
    def evaluate(self, pitchAngle: float, rigidity: float) -> float:
        """
        Evaluate the Gaussian pitch angle distribution.
        
        Args:
            pitchAngle: The pitch angle in radians
            rigidity: The rigidity in GV
            
        Returns:
            float: The value of the Gaussian pitch angle distribution
        """
        return self.normFactor * np.exp(-(pitchAngle - self.alpha)**2 / (self.sigma**2))


class GaussianBeeckPitchAngleDistribution(PitchAngleDistribution):
    """
    Gaussian Beeck pitch angle distribution.
    """

    def __init__(self, normFactor: float, A: float, B: float,
                 reference_latitude_in_GSM: float = 0.0, reference_longitude_in_GSM: float = 0.0):
        """
        Initialize the Gaussian Beeck pitch angle distribution.

        Parameters:
        - normFactor: float
            The normalization factor.
        - A: float
            Parameter A for the distribution.
        - B: float
            Parameter B for the distribution.
        - reference_latitude_in_GSM: float
            Reference latitude in GSM coordinates.
        - reference_longitude_in_GSM: float
            Reference longitude in GSM coordinates.
        """
        super().__init__(None, reference_latitude_in_GSM, reference_longitude_in_GSM)
        self.normFactor = normFactor
        self.A = A
        self.B = B
    
    def evaluate(self, pitchAngle_radians: float, rigidity: float) -> float:
        """
        Evaluate the Gaussian Beeck pitch angle distribution.
        
        Args:
            pitchAngle_radians: The pitch angle in radians
            rigidity: The rigidity in GV
            
        Returns:
            float: The value of the Gaussian Beeck pitch angle distribution
        """
        return self.normFactor * np.exp((-0.5 * (pitchAngle_radians - (np.sin(pitchAngle_radians) * np.cos(pitchAngle_radians)))) / \
                                        (self.A - (0.5 * (self.A - self.B) * (1 - np.cos(pitchAngle_radians)))))

# For backward compatibility
pitchAngleDistribution = PitchAngleDistribution
cosinePitchAngleDistribution = CosinePitchAngleDistribution
isotropicPitchAngleDistribution = IsotropicPitchAngleDistribution
gaussianPitchAngleDistribution = GaussianPitchAngleDistribution
gaussianBeeckPitchAngleDistribution = GaussianBeeckPitchAngleDistribution