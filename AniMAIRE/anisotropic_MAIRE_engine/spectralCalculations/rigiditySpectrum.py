from scipy.interpolate import interp1d
import pandas as pd
import scipy
import numpy as np
import datetime
import matplotlib.pyplot as plt
from typing import Optional, Callable, Any, List, Union
from CosRayModifiedISO import CosRayModifiedISO
from .utils import Distribution, SummedFunction, ScaledFunction
import numba

print("WARNING: currently unknown whether the reported spectral weighting factor should be in terms of energy or rigidity")

# Numba-optimized standalone functions that will be used by the classes
@numba.njit(cache=True)
def power_law_evaluate(normalisationFactor: float, spectralIndex: float, x: float) -> float:
    """Optimized power law calculation"""
    return normalisationFactor * (x ** spectralIndex)

@numba.njit(cache=True)
def step_function(rigidity: float, lowerLimit: float, upperLimit: float) -> float:
    """Optimized step function calculation"""
    if (rigidity >= lowerLimit) and (rigidity <= upperLimit):
        return 1.0
    else:
        return 0.0

@numba.njit(cache=True)
def spec_index_modification(deltaGamma: float, P: float) -> float:
    """Optimized spectral index modification calculation"""
    return deltaGamma * (P - 1)

@numba.njit(cache=True)
def common_modified_power_law_evaluate(J0: float, gamma: float, deltaGamma: float, 
                                      P: float, lowerLimit: float, upperLimit: float) -> float:
    """Optimized common modified power law calculation"""
    mod = spec_index_modification(deltaGamma, P)
    step = step_function(P, lowerLimit, upperLimit)
    return J0 * step * (P ** (-(gamma + mod))) / (100 ** 2)

@numba.njit(cache=True)
def spec_index_modification_high(deltaGamma: float, P: float) -> float:
    """Optimized high rigidity spectral index modification"""
    return deltaGamma * (P - 1)

@numba.njit(cache=True)
def spec_index_modification_low(deltaGamma: float, P: float) -> float:
    """Optimized low rigidity spectral index modification"""
    return deltaGamma * P

@numba.njit(cache=True)
def spec_index_modification_split(deltaGamma: float, P: float) -> float:
    """Optimized spectral index modification with split"""
    if P > 1.0:
        return spec_index_modification_high(deltaGamma, P)
    else:
        return spec_index_modification_low(deltaGamma, P)

@numba.njit(cache=True)
def common_modified_power_law_split_evaluate(J0: float, gamma: float, deltaGamma: float, P: float) -> float:
    """Optimized common modified power law with split calculation"""
    mod = spec_index_modification_split(deltaGamma, P)
    return J0 * (P ** (-(gamma + mod))) / (100 ** 2)

class RigiditySpectrum(Distribution):
    """
    Base class for rigidity spectra.
    """

    def __init__(self, rigiditySpec: Optional[Callable[[float], float]] = None):
        """
        Initialize the rigidity spectrum.
        
        Args:
            rigiditySpec: Callable function for evaluating the spectrum, if None,
                          the subclass evaluate method will be used
        """
        self.rigiditySpec = rigiditySpec or self.evaluate

    def evaluate(self, x: float) -> float:
        """
        Default evaluate method, should be overridden by subclasses.
        
        Args:
            x: The rigidity in GV
            
        Returns:
            float: The value of the rigidity spectrum
        """
        raise NotImplementedError("Subclasses must implement evaluate method")

    def __call__(self, x: float) -> float:
        """
        Evaluate the rigidity spectrum at a given rigidity.

        Args:
            x: The rigidity in GV

        Returns:
            float: The value of the rigidity spectrum
        """
        return self.rigiditySpec(x)
    
    def __add__(self, right: 'RigiditySpectrum') -> 'RigiditySpectrum':
        """
        Add two rigidity spectra.

        Args:
            right: The other rigidity spectrum

        Returns:
            A new rigidity spectrum representing the sum
        """
        summed_spectrum = RigiditySpectrum()
        summed_spectrum.rigiditySpec = SummedFunction(self.rigiditySpec, right.rigiditySpec)
        return summed_spectrum
    
    def __mul__(self, scalar: float) -> 'RigiditySpectrum':
        """
        Multiply the spectrum by a scalar.
        
        Args:
            scalar: The scaling factor
            
        Returns:
            A new scaled rigidity spectrum
        """
        scaled_spectrum = RigiditySpectrum()
        scaled_spectrum.rigiditySpec = ScaledFunction(self.rigiditySpec, scalar)
        return scaled_spectrum
        
    __rmul__ = __mul__
    
    def plot(self, title: Optional[str] = None, ax: Optional[plt.Axes] = None, 
             min_rigidity: float = 0.1, max_rigidity: float = 20, **kwargs: Any) -> plt.Axes:
        """
        Plot the spectrum for this spectrum object.
        
        Args:
            title: Title for the plot. If None, a default title is used
            ax: Axes to plot on. If None, a new figure is created
            min_rigidity: Minimum rigidity in GV for spectrum plot (default: 0.1)
            max_rigidity: Maximum rigidity in GV for spectrum plot (default: 20)
            **kwargs: Additional arguments passed to plot function
        
        Returns:
            The axes containing the plot
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        
        rigidity_range = np.logspace(np.log10(min_rigidity), np.log10(max_rigidity), 100)  # GV
        # Optimize by converting list comprehension to vectorized numpy operation if possible
        flux_values = np.array([self.rigiditySpec(r) for r in rigidity_range])
        ax.loglog(rigidity_range, flux_values, **kwargs)
        ax.set_xlabel('Rigidity (GV)')
        ax.set_ylabel('Flux (particles/m²/sr/s/GV)')
        ax.set_title('Rigidity Spectrum' if title is None else title)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.set_xlim(min_rigidity, max_rigidity)
    
        return ax

class PowerLawSpectrum(RigiditySpectrum):
    """
    Power law rigidity spectrum.
    """

    def __init__(self, normalisationFactor: float, spectralIndex: float):
        """
        Initialize the power law spectrum.

        Args:
            normalisationFactor: The normalization factor
            spectralIndex: The spectral index
        """
        super().__init__(None)
        self.normalisationFactor = normalisationFactor
        self.spectralIndex = spectralIndex
    
    def evaluate(self, x: float) -> float:
        """
        Evaluate the power law spectrum.
        
        Args:
            x: The rigidity in GV
            
        Returns:
            float: The value of the power law spectrum
        """
        return power_law_evaluate(self.normalisationFactor, self.spectralIndex, x)

class InterpolatedInputFileSpectrum(RigiditySpectrum):
    """
    Interpolated rigidity spectrum from an input file.
    """

    def __init__(self, inputFileName: str):
        """
        Initialize the interpolated spectrum.

        Args:
            inputFileName: The path to the input file
        """
        self.inputFilename = inputFileName
        interp_func = self.readSpecFromCSV(self.inputFilename)
        super().__init__(interp_func)

    def readSpecFromCSV(self, inputFileName: str) -> Callable[[float], float]:
        """
        Read the spectrum from a CSV file.

        Args:
            inputFileName: The name of the input file

        Returns:
            The interpolated spectrum function
        """
        inputDF = pd.read_csv(inputFileName, header=None)
        rigidityList = inputDF[0].to_numpy()  # GV - convert to numpy for better performance
        fluxList = inputDF[1].to_numpy()  # p/m2/sr/s/GV
        fluxListcm2 = fluxList / (100 ** 2)

        rigiditySpec = scipy.interpolate.interp1d(rigidityList, fluxListcm2, kind="linear",
                                                  fill_value=0.0, bounds_error=False)
        
        return rigiditySpec

class DLRmodelSpectrum(RigiditySpectrum):
    """
    DLR model rigidity spectrum.
    """

    def __init__(self, atomicNumber: int, date_and_time: Optional['datetime.datetime'] = None, 
                 OULUcountRateInSeconds: Optional[float] = None, W_parameter: Optional[float] = None):
        """
        Initialize the DLR model spectrum.

        Args:
            atomicNumber: The atomic number of the particle
            date_and_time: The date and time for the spectrum
            OULUcountRateInSeconds: The OULU count rate in seconds
            W_parameter: The W parameter for the DLR model
            
        Note: Exactly one of date_and_time, OULUcountRateInSeconds or W_parameter must be provided
        """
        if not sum([(date_and_time is not None), (OULUcountRateInSeconds is not None), (W_parameter is not None)]) == 1:
            print("Error: exactly one supplied input out of the date and time, OULU count rate per second or the W parameter, to the DLR model spectrum must be given!")
            raise ValueError("Exactly one parameter must be provided")

        if date_and_time is not None:
            self._generatedSpectrumDF = CosRayModifiedISO.getSpectrumUsingTimestamp(timestamp=date_and_time, atomicNumber=atomicNumber)

        if W_parameter is not None:
            self._generatedSpectrumDF = CosRayModifiedISO.getSpectrumUsingSolarModulation(solarModulationWparameter=W_parameter, atomicNumber=atomicNumber)

        if OULUcountRateInSeconds is not None:
            self._generatedSpectrumDF = CosRayModifiedISO.getSpectrumUsingOULUcountRate(OULUcountRatePerSecond=OULUcountRateInSeconds, atomicNumber=atomicNumber)

        # Extract numpy arrays for better performance with numba
        x_values = self._generatedSpectrumDF["Rigidity (GV/n)"].to_numpy()
        y_values = self._generatedSpectrumDF["d_Flux / d_R (cm-2 s-1 sr-1 (GV/n)-1)"].to_numpy()
        
        interp_func = interp1d(x=x_values, y=y_values,
                              kind="linear",
                              bounds_error=False,
                              fill_value=(0.0, 0.0))
        
        super().__init__(interp_func)

class CommonModifiedPowerLawSpectrum(RigiditySpectrum):
    """
    Common modified power law rigidity spectrum.
    """

    def __init__(self, J0: float, gamma: float, deltaGamma: float, 
                 lowerLimit: float = -np.inf, upperLimit: float = np.inf):
        """
        Initialize the common modified power law spectrum.

        Args:
            J0: The normalization factor (m-2 s-1 sr-1 GV-1)
            gamma: The spectral index
            deltaGamma: The modification factor for the spectral index
            lowerLimit: The lower limit for the rigidity (default: -inf)
            upperLimit: The upper limit for the rigidity (default: inf)
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
        
        Args:
            P: The rigidity in GV
            
        Returns:
            The spectral index modification
        """
        return spec_index_modification(self.deltaGamma, P)
    
    def step_function(self, rigidity: float, lowerLimit: float, upperLimit: float) -> float:
        """
        Step function for the rigidity spectrum.

        Args:
            rigidity: The rigidity in GV
            lowerLimit: The lower limit for the rigidity
            upperLimit: The upper limit for the rigidity

        Returns:
            1.0 if rigidity is within limits, 0.0 otherwise
        """
        return step_function(rigidity, lowerLimit, upperLimit)
    
    def evaluate(self, P: float) -> float:
        """
        Evaluate the common modified power law spectrum.
        
        Args:
            P: The rigidity in GV
            
        Returns:
            The spectrum value at the given rigidity
        """
        return common_modified_power_law_evaluate(
            self.J0, self.gamma, self.deltaGamma, P, self.lowerLimit, self.upperLimit
        )

class CommonModifiedPowerLawSpectrumSplit(RigiditySpectrum):
    """
    Common modified power law rigidity spectrum with split spectral index modification.
    """

    def __init__(self, J0: float, gamma: float, deltaGamma: float):
        """
        Initialize the common modified power law spectrum with split spectral index modification.

        Args:
            J0: The normalization factor (m-2 s-1 sr-1 GV-1)
            gamma: The spectral index
            deltaGamma: The modification factor for the spectral index
        """
        super().__init__(None)
        self.J0 = J0  # m-2 s-1 sr-1 GV-1
        self.gamma = gamma
        self.deltaGamma = deltaGamma
    
    def specIndexModification_high(self, P: float) -> float:
        """
        Calculate the spectral index modification for high rigidity.
        
        Args:
            P: The rigidity in GV
            
        Returns:
            The high rigidity spectral index modification
        """
        return spec_index_modification_high(self.deltaGamma, P)
    
    def specIndexModification_low(self, P: float) -> float:
        """
        Calculate the spectral index modification for low rigidity.
        
        Args:
            P: The rigidity in GV
            
        Returns:
            The low rigidity spectral index modification
        """
        return spec_index_modification_low(self.deltaGamma, P)
    
    def specIndexModification(self, P: float) -> float:
        """
        Calculate the spectral index modification.
        
        Args:
            P: The rigidity in GV
            
        Returns:
            The appropriate spectral index modification based on rigidity
        """
        return spec_index_modification_split(self.deltaGamma, P)
    
    def evaluate(self, P: float) -> float:
        """
        Evaluate the common modified power law spectrum with split spectral index modification.
        
        Args:
            P: The rigidity in GV
            
        Returns:
            The spectrum value at the given rigidity
        """
        return common_modified_power_law_split_evaluate(self.J0, self.gamma, self.deltaGamma, P)

# For backward compatibility
rigiditySpectrum = RigiditySpectrum
powerLawSpectrum = PowerLawSpectrum
interpolatedInputFileSpectrum = InterpolatedInputFileSpectrum
DLRmodelSpectrum = DLRmodelSpectrum
CommonModifiedPowerLawSpectrum = CommonModifiedPowerLawSpectrum
CommonModifiedPowerLawSpectrumSplit = CommonModifiedPowerLawSpectrumSplit
