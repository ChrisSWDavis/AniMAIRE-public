from scipy.interpolate import interp1d
import pandas as pd
import scipy
import numpy as np
import datetime
import matplotlib.pyplot as plt
from CosRayModifiedISO import CosRayModifiedISO

print("WARNING: currently unknown whether the reported spectral weighting factor should be in terms of energy or rigidity")

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

class rigiditySpectrum():
    """
    Base class for rigidity spectra.
    """

    def __init__(self, rigiditySpec: callable = None):
        """
        Initialize the rigidity spectrum.
        """
        self.rigiditySpec = rigiditySpec or self.evaluate

    def evaluate(self, x: float) -> float:
        """
        Default evaluate method, should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement evaluate method")

    def __call__(self, x: float) -> float:
        """
        Evaluate the rigidity spectrum at a given rigidity.

        Parameters:
        - x: float
            The rigidity.

        Returns:
        - float
            The value of the rigidity spectrum.
        """
        return self.rigiditySpec(x)
    
    def __add__(self, right: 'rigiditySpectrum') -> 'rigiditySpectrum':
        """
        Add two rigidity spectra.

        Parameters:
        - right: rigiditySpectrum
            The other rigidity spectrum.

        Returns:
        - rigiditySpectrum
            The sum of the two rigidity spectra.
        """
        summed_spectrum = rigiditySpectrum()
        summed_spectrum.rigiditySpec = SummedFunction(self.rigiditySpec, right.rigiditySpec)
        return summed_spectrum
    
    def plot(self, title=None, ax=None, min_rigidity=0.1, max_rigidity=20, **kwargs):
        """
        Plot the spectrum for this spectrum object.
        
        Parameters:
        -----------
        title : str, optional
            Title for the plot. If None, a default title is used.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        min_rigidity : float, optional
            Minimum rigidity in GV for spectrum plot (default: 0.1)
        max_rigidity : float, optional
            Maximum rigidity in GV for spectrum plot (default: 20)
        
        Returns:
        --------
        matplotlib.axes.Axes
            The axes containing the plot
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        
        rigidity_range = np.logspace(np.log10(min_rigidity), np.log10(max_rigidity), 100)  # GV
        flux_values = [self.rigiditySpec(r) for r in rigidity_range]
        ax.loglog(rigidity_range, flux_values, **kwargs)
        ax.set_xlabel('Rigidity (GV)')
        ax.set_ylabel('Flux (particles/m²/sr/s/GV)')
        ax.set_title('Rigidity Spectrum' if title is None else title)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.set_xlim(min_rigidity, max_rigidity)
    
        return ax

class powerLawSpectrum(rigiditySpectrum):
    """
    Power law rigidity spectrum.
    """

    def __init__(self, normalisationFactor: float, spectralIndex: float):
        """
        Initialize the power law spectrum.

        Parameters:
        - normalisationFactor: float
            The normalization factor.
        - spectralIndex: float
            The spectral index.
        """
        super().__init__(None)
        self.normalisationFactor = normalisationFactor
        self.spectralIndex = spectralIndex
    
    def evaluate(self, x: float) -> float:
        """
        Evaluate the power law spectrum.
        """
        return self.normalisationFactor * (x ** self.spectralIndex)

class interpolatedInputFileSpectrum(rigiditySpectrum):
    """
    Interpolated rigidity spectrum from an input file.
    """

    def __init__(self, inputFileName: str):
        """
        Initialize the interpolated spectrum.

        Parameters:
        - inputFileName: str
            The path to the input file.
        """
        self.inputFilename = inputFileName
        interp_func = self.readSpecFromCSV(self.inputFilename)
        super().__init__(interp_func)

    def readSpecFromCSV(self, inputFileName: str) -> callable:
        """
        Read the spectrum from a CSV file.

        Parameters:
        - inputFileName: str
            The name of the input file.

        Returns:
        - callable
            The interpolated spectrum.
        """
        inputDF = pd.read_csv(inputFileName, header=None)
        rigidityList = inputDF[0]  # GV
        fluxList = inputDF[1]  # p/m2/sr/s/GV
        fluxListcm2 = fluxList / (100 ** 2)

        rigiditySpec = scipy.interpolate.interp1d(rigidityList, fluxListcm2, kind="linear",
                                                  fill_value=0.0, bounds_error=False)
        
        return rigiditySpec

class DLRmodelSpectrum(rigiditySpectrum):
    """
    DLR model rigidity spectrum.
    """

    def __init__(self, atomicNumber: int, date_and_time: 'datetime' = None, OULUcountRateInSeconds: float = None, W_parameter: float = None):
        """
        Initialize the DLR model spectrum.

        Parameters:
        - atomicNumber: int
            The atomic number of the particle.
        - date_and_time: datetime, optional
            The date and time for the spectrum.
        - OULUcountRateInSeconds: float, optional
            The OULU count rate in seconds.
        - W_parameter: float, optional
            The W parameter for the DLR model.
        """
        if not sum([(date_and_time is not None), (OULUcountRateInSeconds is not None), (W_parameter is not None)]) == 1:
            print("Error: exactly one supplied input out of the date and time, OULU count rate per second or the W parameter, to the DLR model spectrum must be given!")
            raise Exception

        if date_and_time is not None:
            self._generatedSpectrumDF = CosRayModifiedISO.getSpectrumUsingTimestamp(timestamp=date_and_time, atomicNumber=atomicNumber)

        if W_parameter is not None:
            self._generatedSpectrumDF = CosRayModifiedISO.getSpectrumUsingSolarModulation(solarModulationWparameter=W_parameter, atomicNumber=atomicNumber)

        if OULUcountRateInSeconds is not None:
            self._generatedSpectrumDF =CosRayModifiedISO.getSpectrumUsingOULUcountRate(OULUcountRatePerSecond=OULUcountRateInSeconds, atomicNumber=atomicNumber)

        interp_func = interp1d(x=self._generatedSpectrumDF["Rigidity (GV/n)"],
                             y=self._generatedSpectrumDF["d_Flux / d_R (cm-2 s-1 sr-1 (GV/n)-1)"],
                             kind="linear",
                             bounds_error=False,
                             fill_value=(0.0, 0.0))
        
        super().__init__(interp_func)

class CommonModifiedPowerLawSpectrum(rigiditySpectrum):
    """
    Common modified power law rigidity spectrum.
    """

    def __init__(self, J0: float, gamma: float, deltaGamma: float, lowerLimit: float = -np.inf, upperLimit: float = np.inf):
        """
        Initialize the common modified power law spectrum.

        Parameters:
        - J0: float
            The normalization factor.
        - gamma: float
            The spectral index.
        - deltaGamma: float
            The modification factor for the spectral index.
        - lowerLimit: float, optional
            The lower limit for the rigidity.
        - upperLimit: float, optional
            The upper limit for the rigidity.
        """
        super().__init__(None)
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit
        self.J0 = J0  # m-2 s-1 sr-1 GV-1
        self.gamma = gamma
        self.deltaGamma = deltaGamma

    def specIndexModification(self, P: float) -> float:
        """
        Calculate the spectral index modification.
        """
        return self.deltaGamma * (P - 1)
    
    def step_function(self, rigidity: float, lowerLimit: float, upperLimit: float) -> float:
        """
        Step function for the rigidity spectrum.

        Parameters:
        - rigidity: float
            The rigidity.
        - lowerLimit: float
            The lower limit for the rigidity.
        - upperLimit: float
            The upper limit for the rigidity.

        Returns:
        - float
            The value of the step function.
        """
        if (rigidity >= lowerLimit) and (rigidity <= upperLimit):
            return 1.0
        else:
            return 0.0
    
    def evaluate(self, P: float) -> float:
        """
        Evaluate the common modified power law spectrum.
        """
        return self.J0 * self.step_function(P, self.lowerLimit, self.upperLimit) * (P ** (-(self.gamma + self.specIndexModification(P)))) / (100 ** 2)  # cm-2 s-1 sr-1 GV-1 : converted from m-2 to cm-2

class CommonModifiedPowerLawSpectrumSplit(rigiditySpectrum):
    """
    Common modified power law rigidity spectrum with split spectral index modification.
    """

    def __init__(self, J0: float, gamma: float, deltaGamma: float):
        """
        Initialize the common modified power law spectrum with split spectral index modification.

        Parameters:
        - J0: float
            The normalization factor.
        - gamma: float
            The spectral index.
        - deltaGamma: float
            The modification factor for the spectral index.
        """
        super().__init__(None)
        self.J0 = J0  # m-2 s-1 sr-1 GV-1
        self.gamma = gamma
        self.deltaGamma = deltaGamma
    
    def specIndexModification_high(self, P: float) -> float:
        """
        Calculate the spectral index modification for high rigidity.
        """
        return self.deltaGamma * (P - 1)
    
    def specIndexModification_low(self, P: float) -> float:
        """
        Calculate the spectral index modification for low rigidity.
        """
        return self.deltaGamma * (P)
    
    def specIndexModification(self, P: float) -> float:
        """
        Calculate the spectral index modification.
        """
        return self.specIndexModification_high(P) if P > 1.0 else self.specIndexModification_low(P)
    
    def evaluate(self, P: float) -> float:
        """
        Evaluate the common modified power law spectrum with split spectral index modification.
        """
        return self.J0 * (P ** (-(self.gamma + self.specIndexModification(P)))) / (100 ** 2)  # cm-2 s-1 sr-1 GV-1 : converted from m-2 to cm-2
