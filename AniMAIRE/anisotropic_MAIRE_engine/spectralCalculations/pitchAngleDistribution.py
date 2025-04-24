import numpy as np
from spacepy.coordinates import Coords as spaceCoords
import copy
import matplotlib.pyplot as plt

# Define picklable callable classes for function composition
class SummedFunction:
    """Callable class that combines two functions by addition."""
    def __init__(self, func1, func2):
        self.func1 = func1
        self.func2 = func2
        
    def __call__(self, *args, **kwargs):
        return self.func1(*args, **kwargs) + self.func2(*args, **kwargs)
        
class ScaledFunction:
    """Callable class that scales a function by a factor."""
    def __init__(self, func, scale):
        self.func = func
        self.scale = scale
        
    def __call__(self, *args, **kwargs):
        return self.scale * self.func(*args, **kwargs)

class pitchAngleDistribution():
    """
    Base class for pitch angle distributions.
    """

    def __init__(self, 
                 pitch_angle_distribution: callable = None, 
                 reference_latitude_in_GSM: float = 0.0, 
                 reference_longitude_in_GSM: float = 0.0):
        """
        Initialize the pitch angle distribution.

        Parameters:
        - pitch_angle_distribution: callable
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
    
    def __add__(self, right: 'pitchAngleDistribution') -> 'pitchAngleDistribution':
        """
        Add two pitch angle distributions.

        Parameters:
        - right: pitchAngleDistribution
            The other pitch angle distribution.

        Returns:
        - pitchAngleDistribution
            The sum of the two pitch angle distributions.
        """
        summed_dist = copy.deepcopy(self)
        summed_dist.pitchAngleDistFunction = SummedFunction(self.pitchAngleDistFunction, right.pitchAngleDistFunction)
        return summed_dist
    
    def plot(self, title=None, reference_rigidity=1.0, ax=None, **kwargs):
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
    
    def __mul__(self, right: float) -> 'pitchAngleDistribution':
        """
        Multiply the pitch angle distribution by a scalar.

        Parameters:
        - right: float
            The scalar.

        Returns:
        - pitchAngleDistribution
            The scaled pitch angle distribution.
        """
        multiplied_dist = copy.deepcopy(self)
        multiplied_dist.pitchAngleDistFunction = ScaledFunction(self.pitchAngleDistFunction, right)
        return multiplied_dist

    __rmul__ = __mul__

class cosinePitchAngleDistribution(pitchAngleDistribution):
    """
    Cosine pitch angle distribution.
    """

    def __init__(self, reference_latitude_in_GSM=0.0, reference_longitude_in_GSM=0.0):
        """
        Initialize the cosine pitch angle distribution.
        """
        super().__init__(None, reference_latitude_in_GSM, reference_longitude_in_GSM)
    
    def evaluate(self, pitchAngle, rigidity):
        """
        Evaluate the cosine pitch angle distribution.
        """
        return np.abs(0.5 * np.sin(2 * pitchAngle))

class isotropicPitchAngleDistribution(pitchAngleDistribution):
    """
    Isotropic pitch angle distribution.
    """

    def __init__(self, reference_latitude_in_GSM=0.0, reference_longitude_in_GSM=0.0):
        """
        Initialize the isotropic pitch angle distribution.
        """
        super().__init__(None, reference_latitude_in_GSM, reference_longitude_in_GSM)
    
    def evaluate(self, pitchAngle, rigidity):
        """
        Evaluate the isotropic pitch angle distribution.
        """
        return 1

class gaussianPitchAngleDistribution(pitchAngleDistribution):
    """
    Gaussian pitch angle distribution.
    """

    def __init__(self, normFactor: float, sigma: float, alpha: float = 0.0, 
                 reference_latitude_in_GSM=0.0, reference_longitude_in_GSM=0.0):
        """
        Initialize the Gaussian pitch angle distribution.

        Parameters:
        - normFactor: float
            The normalization factor.
        - sigma: float
            The standard deviation of the Gaussian distribution.
        - alpha: float, optional
            The mean of the Gaussian distribution.
        """
        super().__init__(None, reference_latitude_in_GSM, reference_longitude_in_GSM)
        self.normFactor = normFactor
        self.sigma = sigma
        self.alpha = alpha
    
    def evaluate(self, pitchAngle, rigidity):
        """
        Evaluate the Gaussian pitch angle distribution.
        """
        return self.normFactor * np.exp(-(pitchAngle - self.alpha)**2 / (self.sigma**2))

class gaussianBeeckPitchAngleDistribution(pitchAngleDistribution):
    """
    Gaussian Beeck pitch angle distribution.
    """

    def __init__(self, normFactor: float, A: float, B: float,
                 reference_latitude_in_GSM=0.0, reference_longitude_in_GSM=0.0):
        """
        Initialize the Gaussian Beeck pitch angle distribution.

        Parameters:
        - normFactor: float
            The normalization factor.
        - A: float
            Parameter A for the distribution.
        - B: float
            Parameter B for the distribution.
        """
        super().__init__(None, reference_latitude_in_GSM, reference_longitude_in_GSM)
        self.normFactor = normFactor
        self.A = A
        self.B = B
    
    def evaluate(self, pitchAngle_radians, rigidity):
        """
        Evaluate the Gaussian Beeck pitch angle distribution.
        """
        return self.normFactor * np.exp((-0.5 * (pitchAngle_radians - (np.sin(pitchAngle_radians) * np.cos(pitchAngle_radians)))) / \
                                        (self.A - (0.5 * (self.A - self.B) * (1 - np.cos(pitchAngle_radians)))))