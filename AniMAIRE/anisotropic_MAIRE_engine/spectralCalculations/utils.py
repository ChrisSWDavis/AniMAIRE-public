import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Generic, Any

T = TypeVar('T')
U = TypeVar('U')

# Callable composition classes
class SummedFunction:
    """Callable class that combines two functions by addition."""
    def __init__(self, func1: Callable, func2: Callable):
        self.func1 = func1
        self.func2 = func2
        
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func1(*args, **kwargs) + self.func2(*args, **kwargs)
        
class ScaledFunction:
    """Callable class that scales a function by a factor."""
    def __init__(self, func: Callable, scale: float):
        self.func = func
        self.scale = scale
        
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.scale * self.func(*args, **kwargs)

# Base distribution interface
class Distribution(ABC, Generic[T]):
    """
    Abstract base class for distributions.
    
    This class defines the interface for all distribution types,
    including rigidity spectra and pitch angle distributions.
    """
    
    @abstractmethod
    def evaluate(self, x: T, *args: Any) -> float:
        """
        Evaluate the distribution at a given point.
        
        Args:
            x: The input value
            *args: Additional arguments needed for evaluation
            
        Returns:
            float: The distribution value at the given point
        """
        pass
    
    @abstractmethod
    def __call__(self, x: T, *args: Any) -> float:
        """Make the distribution callable"""
        pass
    
    @abstractmethod
    def __add__(self, other: 'Distribution[T]') -> 'Distribution[T]':
        """Add two distributions"""
        pass
    
    @abstractmethod
    def __mul__(self, scalar: float) -> 'Distribution[T]':
        """Scale the distribution by a factor"""
        pass
    
    @abstractmethod
    def plot(self, **kwargs: Any) -> Any:
        """Plot the distribution"""
        pass

# For backward compatibility
Spectrum = Distribution 