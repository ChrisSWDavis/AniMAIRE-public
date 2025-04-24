from AniMAIRE.anisotropic_MAIRE_engine.spectralCalculations.momentaDistribution import momentaDistribution
from AniMAIRE.anisotropic_MAIRE_engine.spectralCalculations.pitchAngleDistribution import pitchAngleDistribution
from AniMAIRE.anisotropic_MAIRE_engine.spectralCalculations.rigiditySpectrum import rigiditySpectrum
from AniMAIRE.dose_plotting import plot_dose_map, plot_on_spherical_globe


import matplotlib.pyplot as plt
import pandas as pd


import logging

class DoseRateFrame(pd.DataFrame):
    """
    A class for containing and manipulating a single dose rate dataframe from a specific timestamp.

    This class extends pandas.DataFrame and provides methods for data filtering, visualization,
    and analysis of dose rate data at a specific point in time during a GLE event.
    """

    # Required for pandas subclassing
    _metadata = ['timestamp', 'particle_distributions', 'run_parameters']

    @property
    def _constructor(self):
        return DoseRateFrame

    def __init__(self, data=None, timestamp=None, particle_distributions=None, run_parameters=None, *args, **kwargs):
        """
        Initialize a DoseRateFrame with data and optional timestamp.

        Parameters:
        -----------
        data : various formats accepted by pandas.DataFrame
            The dose rate data
        timestamp : datetime, optional
            The timestamp associated with this dose rate data
        particle_distributions : list, optional
            List of particle distributions used to generate this dose rate data.
            Expected format: [(atomic_number, charge_number, spectrum_callable, pad_object), ...]
        run_parameters : dict, optional
            Dictionary containing the input parameters used in run_from_spectra
        *args, **kwargs : additional arguments passed to pandas.DataFrame constructor
        """
        super().__init__(data, *args, **kwargs)
        self.timestamp = timestamp
        self.particle_distributions = particle_distributions if particle_distributions is not None else []
        self.run_parameters = run_parameters

    def get_altitudes(self):
        """
        Get the unique altitude values available in this DoseRateFrame.

        Returns:
        --------
        numpy.ndarray
            Sorted array of unique altitude values in km
        """
        if self.empty or 'altitude (km)' not in self.columns:
            return []
        return sorted(self.loc[:, 'altitude (km)'].unique())

    def at_altitude(self, altitude):
        """
        Filter data to a specific altitude.

        Parameters:
        -----------
        altitude : float
            The altitude in km to filter by

        Returns:
        --------
        DoseRateFrame
            Filtered dataframe containing only rows at the specified altitude
        """
        result = self.query(f"`altitude (km)` == {altitude}")
        result.timestamp = self.timestamp
        result.particle_distributions = self.particle_distributions
        result.run_parameters = self.run_parameters
        return result

    def plot_dose_map(self, altitude=12.192, title=None,**kwargs):
        """
        Plot a 2D dose rate map at a specific altitude using AniMAIRE plotting tools.

        Parameters:
        -----------
        altitude : float, optional
            The altitude in km for which to plot the dose map. Default is 12.192 km.
        title : str, optional
            Title for the plot. If None, uses the timestamp if available.
        ax : matplotlib.axes.Axes, optional
            Axes on which to draw the plot. If None, creates a new figure.
        **kwargs : additional keyword arguments
            Additional arguments passed to dose_plotting.plot_dose_map

        Returns:
        --------
        tuple
            (ax, colorbar) - The matplotlib axes and colorbar objects
        """
        # Filter data to the specified altitude
        data_at_alt = self.at_altitude(altitude)

        # Set custom title if provided
        if title:
            pass
        elif self.timestamp:
            title = f'Dose Rate Map at {self.timestamp.strftime("%Y-%m-%d %H:%M:%S")} (Altitude: {altitude} km)'

        # Use AniMAIRE's plotting function
        output_plot = plot_dose_map(data_at_alt,plot_title=title,**kwargs)

        return plt.gca()

    def plot_on_globe(self, altitude=12.192, **kwargs):
        """
        Plot dose rate data on a 3D globe using AniMAIRE plotting tools.

        Parameters:
        -----------
        altitude : float, optional
            The altitude in km for which to plot the data. Default is 12.192 km.
        **kwargs : additional keyword arguments
            Additional arguments passed to dose_plotting.plot_on_spherical_globe

        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the 3D globe plot
        """
        data_at_alt = self.at_altitude(altitude)

        # Set title based on timestamp if available
        if 'plot_title' not in kwargs and self.timestamp:
            kwargs['plot_title'] = f'Dose Rate at {self.timestamp.strftime("%Y-%m-%d %H:%M:%S")} (Altitude: {altitude} km)'

        return plot_on_spherical_globe(data_at_alt, **kwargs)

    def get_max_dose(self, altitude=None):
        """
        Get the maximum dose rate in the dataset, optionally at a specific altitude.

        Parameters:
        -----------
        altitude : float, optional
            The altitude in km to filter by. If None, considers all altitudes.

        Returns:
        --------
        float
            Maximum effective dose rate value
        """
        if altitude is not None:
            data = self.at_altitude(altitude)
        else:
            data = self

        return data['edose'].max() if not data.empty else 0

    def get_mean_dose(self, altitude=None):
        """
        Get the mean dose rate in the dataset, optionally at a specific altitude.

        Parameters:
        -----------
        altitude : float, optional
            The altitude in km to filter by. If None, considers all altitudes.

        Returns:
        --------
        float
            Mean effective dose rate value
        """
        if altitude is not None:
            data = self.at_altitude(altitude)
        else:
            data = self

        return data['edose'].mean() if not data.empty else 0

    def __repr__(self):
        """String representation of the DoseRateFrame."""
        timestamp_str = f" at {self.timestamp}" if self.timestamp else ""
        return f"DoseRateFrame{timestamp_str} with {len(self)} data points"

    def plot_spectra(self, ax=None, min_rigidity=0.1, max_rigidity=20, **kwargs):
        """
        Plot the rigidity spectra for all particle types in this DoseRateFrame.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        min_rigidity : float, optional
            Minimum rigidity in GV for spectrum plot (default: 0.1)
        max_rigidity : float, optional
            Maximum rigidity in GV for spectrum plot (default: 20)
        **kwargs : additional keyword arguments
            Passed to the underlying rigiditySpectrum.plot() method.

        Returns:
        --------
        matplotlib.axes.Axes
            The axes containing the plot(s).
        """
        if not self.particle_distributions:
            logging.warning("No particle distributions found in this DoseRateFrame.")
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        for particle_distribution in self.particle_distributions:
            label = f'Z={particle_distribution.particle_species.atomicNumber}' 
            particle_distribution.momentum_distribution.rigidity_spectrum.plot(ax=ax, min_rigidity=min_rigidity, max_rigidity=max_rigidity, title=None, **kwargs) # Plot on the same axes
            # Add label manually as plot might not support it directly
            ax.lines[-1].set_label(label)

        ax.set_title(f'Input Rigidity Spectra at {self.timestamp.strftime("%Y-%m-%d %H:%M")}')
        ax.legend()
        return ax

    def plot_pitch_angle_distributions(self, ax=None, reference_rigidity=1.0, **kwargs):
        """
        Plot the pitch angle distributions for all particle types in this DoseRateFrame.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        reference_rigidity : float, optional
            Reference rigidity in GV for pitch angle distribution plot (default: 1.0)
        **kwargs : additional keyword arguments
            Passed to the underlying pitchAngleDistribution.plot() method.

        Returns:
        --------
        matplotlib.axes.Axes
            The axes containing the plot(s).
        """
        if not self.particle_distributions:
            logging.warning("No particle distributions found in this DoseRateFrame.")
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        for particle_distribution in self.particle_distributions:
            if isinstance(particle_distribution.momentum_distribution.pitch_angle_distribution, pitchAngleDistribution):
                label = f'Z={particle_distribution.particle_species.atomicNumber}'
                particle_distribution.momentum_distribution.pitch_angle_distribution.plot(ax=ax, reference_rigidity=reference_rigidity, title=None, **kwargs) # Plot on the same axes
                # Add label manually as plot might not support it directly
                ax.lines[-1].set_label(label)
            else:
                logging.warning(f"Particle distribution does not have a valid pitchAngleDistribution object.")

        ax.set_title(f'Input Pitch Angle Distributions at {self.timestamp.strftime("%Y-%m-%d %H:%M")} (R={reference_rigidity} GV)')
        ax.legend()
        return ax

    def plot_combined_distributions(self, figsize=(12, 5), min_rigidity=0.1, max_rigidity=20, reference_rigidity=1.0, **kwargs):
        """
        Plot the combined rigidity spectrum and pitch angle distribution for each particle type.
        Creates a separate figure for each particle distribution found.

        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches for each plot (default: (12, 5))
        min_rigidity : float, optional
            Minimum rigidity in GV for spectrum plot (default: 0.1)
        max_rigidity : float, optional
            Maximum rigidity in GV for spectrum plot (default: 20)
        reference_rigidity : float, optional
            Reference rigidity in GV for pitch angle distribution plot (default: 1.0)
        **kwargs : additional keyword arguments
            Passed to the underlying momentaDistribution.plot_spectrum_and_pad() method.

        Returns:
        --------
        list[matplotlib.figure.Figure]
            A list of figures, one for each particle distribution. Returns empty list if no distributions.
        """
        if not self.particle_distributions:
            logging.warning("No particle distributions found in this DoseRateFrame.")
            return []

        figures = []
        for particle_distribution in self.particle_distributions:
            # Get the particle species information
            particle_species = particle_distribution.particle_species
            
            # Get the rigidity spectrum
            momenta_dist = particle_distribution.momentum_distribution
            
            if isinstance(momenta_dist, momentaDistribution):
                fig = momenta_dist.plot_spectrum_and_pad(figsize=figsize,
                                                         min_rigidity=min_rigidity,
                                                         max_rigidity=max_rigidity,
                                                         reference_rigidity=reference_rigidity,
                                                         **kwargs)
                fig.suptitle(f'Input Distribution for Z={particle_species.atomicNumber} at {self.timestamp.strftime("%Y-%m-%d %H:%M")}', y=1.02)
                figures.append(fig)
            else:
                logging.warning(f"Cannot plot combined distribution for particle {particle_species.atomicNumber}: Invalid momenta object.")

        return figures