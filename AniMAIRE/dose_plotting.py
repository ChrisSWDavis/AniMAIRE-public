import numpy as np
import plotly.express as px
from scipy.interpolate import NearestNDInterpolator, griddata
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas
import pandas as pd
import cartopy.crs as ccrs
from matplotlib import animation
from IPython.display import HTML
import os
import imageio


pd.options.mode.chained_assignment = None


def plot_on_spherical_globe(data_df, color_column="adose", 
                           title=None, 
                           cmap="Spectral_r", 
                           colorbar_label=None,
                           central_latitude=0,
                           central_longitude=0):
    """
    Plot data on a 3D spherical globe using Cartopy.
    
    Parameters:
    -----------
    data_df : pandas.DataFrame
        DataFrame containing latitude, longitude, and data values
    color_column : str
        Column name in data_df to use for coloring points
    marker_size : int
        Size of markers
    title : str
        Plot title
    cmap : str
        Matplotlib colormap name
    colorbar_label : str
        Label for the colorbar
    central_latitude : float
        Central latitude of the plot
    central_longitude : float
        Central longitude of the plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    
    # Create figure with orthographic projection (3D globe view)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(central_longitude, central_latitude))
    
    # Add coastlines and features
    ax.coastlines(linewidth=0.5)
    ax.gridlines(linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # Create a heatmap using pcolormesh
    # First, we need to reshape the data into a grid
    
    # Create a regular grid for the heatmap
    # lon_grid = np.linspace(data_df["longitude"].min(), data_df["longitude"].max(), 100)
    # lat_grid = np.linspace(data_df["latitude"].min(), data_df["latitude"].max(), 100)
    lon_mesh, lat_mesh = np.meshgrid(data_df["longitude"].unique(), data_df["latitude"].unique())
    
    # Interpolate the data onto the grid
    values = data_df[color_column].values
    grid_values = griddata((data_df["longitude"], data_df["latitude"]), values, 
                           (lon_mesh, lat_mesh), method='linear', fill_value=np.nan)
    
    # Plot the heatmap
    scatter = ax.pcolormesh(lon_mesh, lat_mesh, grid_values, 
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, 
                           alpha=0.8,
                           shading='auto')
    
    # Add colorbar
    if colorbar_label is None:
        colorbar_label = color_column
    cbar = plt.colorbar(scatter, shrink=0.7, pad=0.1)
    cbar.set_label(colorbar_label)
    
    # Set title
    if title:
        plt.title(title)
    
    return fig

def plot_dose_map_contours(dose_map_to_plot, dose_type="edose", levels=3, **kwargs):

    dose_map_to_plot_sorted = dose_map_to_plot.sort_values(by=["longitudeTranslated","latitude"])

    (contour_longs, contour_lats) = np.meshgrid(dose_map_to_plot_sorted["longitudeTranslated"].unique(),
            dose_map_to_plot_sorted["latitude"].unique())

    interp = NearestNDInterpolator(list(zip(dose_map_to_plot_sorted["longitudeTranslated"], dose_map_to_plot_sorted["latitude"])),
                               dose_map_to_plot_sorted[dose_type])

    contours = plt.contour(contour_longs,contour_lats,interp(contour_longs, contour_lats),
            levels=levels,linestyles="dashed",colors="black",zorder=1000,**kwargs)
    plt.clabel(contours, inline=True) #,fmt={"fontweight":"bold"})

def create_single_dose_map_plot_plt(heatmap_DF_to_Plot,
                                    hue_range = None, #(0,100), 
                                    heatmap_s = 63,
                                    edgecolor=None,
                                    dose_type = "edose",
                                    legend_label=r"Effective dose ($\mu Sv / hr$)",
                                    palette="Spectral_r",
                                    plot_longitude_east=False,
                                    plot_colorbar=True):

    if not (heatmap_DF_to_Plot["altitude (km)"].nunique() == 1):
        print()
        print("\033[1mWARNING: multiple altitudes were supplied in the input dataframe, therefore only the map for the maximum altitude will be plotted!\033[0m")
        print()
        heatmap_DF_to_Plot = heatmap_DF_to_Plot.query(f"`altitude (km)` == {heatmap_DF_to_Plot['altitude (km)'].max()}")

    if hue_range is None:
        hue_range = (0,heatmap_DF_to_Plot[dose_type].max())

    ############################ creating background world map and dose image
    currentFigure = plt.gcf()

    currentFigure.set_figheight(10)
    currentFigure.set_figwidth(10)

    # Store original dose type request
    original_dose_type = dose_type
    # Set default legend label if not provided
    effective_legend_label = legend_label

    # Conditionally calculate SEU/SEL rates and update dose_type for plotting
    if dose_type == 'SEU':
        if 'SEU' in heatmap_DF_to_Plot.columns:
            calculated_col_name = "SEU (Upsets/hr/Gb)"
            heatmap_DF_to_Plot[calculated_col_name] = heatmap_DF_to_Plot["SEU"] * (60.0 * 60.0) * 1e9
            dose_type = calculated_col_name # Use the calculated column for plotting
            # Update legend label if it was the default
            if legend_label == r"Effective dose ($\mu Sv / hr$)":
                 effective_legend_label = "SEU (Upsets/hr/Gb)"
        else:
            print(f"Warning: dose_type='SEU' requested, but 'SEU' column not found in DataFrame.")
            # Fallback or error handling needed? For now, proceed with original dose_type if it exists
            if original_dose_type not in heatmap_DF_to_Plot.columns:
                 print(f"Error: Neither 'SEU' nor original dose_type '{original_dose_type}' found.")
                 return None, None # Cannot plot
            dose_type = original_dose_type # Revert to original if SEU base column missing
            
    elif dose_type == 'SEL':
        if 'SEL' in heatmap_DF_to_Plot.columns:
            calculated_col_name = "SEL (Latch-ups/hr/device)"
            heatmap_DF_to_Plot[calculated_col_name] = heatmap_DF_to_Plot["SEL"] * (60.0 * 60.0)
            dose_type = calculated_col_name # Use the calculated column for plotting
             # Update legend label if it was the default
            if legend_label == r"Effective dose ($\mu Sv / hr$)":
                 effective_legend_label = "SEL (Latch-ups/hr/device)"
        else:
            print(f"Warning: dose_type='SEL' requested, but 'SEL' column not found in DataFrame.")
            if original_dose_type not in heatmap_DF_to_Plot.columns:
                 print(f"Error: Neither 'SEL' nor original dose_type '{original_dose_type}' found.")
                 return None, None # Cannot plot
            dose_type = original_dose_type # Revert to original if SEL base column missing

    # Check if the final dose_type column exists before proceeding
    if dose_type not in heatmap_DF_to_Plot.columns:
         print(f"Error: Final dose_type column '{dose_type}' not found in DataFrame. Cannot plot.")
         return None, None

    # Update hue_range based on the final dose_type if it wasn't provided
    if hue_range is None:
        # Calculate max safely, handling potential NaNs or empty slices
        max_val = heatmap_DF_to_Plot[dose_type].max()
        hue_range = (0, max_val if pd.notna(max_val) else 1) # Default max to 1 if max is NaN/None

    #heatmap_DF_to_Plot = pd.read_csv(file_path_to_read, delimiter=',')
    # REMOVED Unconditional calculation:
    # heatmap_DF_to_Plot["SEU (Upsets/hr/Gb)"] = heatmap_DF_to_Plot["SEU"] * (60.0 * 60.0) * 1e9
    # heatmap_DF_to_Plot["SEL (Latch-ups/hr/device)"] = heatmap_DF_to_Plot["SEL"] * (60.0 * 60.0)
    if plot_longitude_east is False:
        heatmap_DF_to_Plot["longitudeTranslated"] = heatmap_DF_to_Plot["longitude"].apply(lambda x:x-360.0 if x > 180.0 else x)
    else:
        heatmap_DF_to_Plot["longitudeTranslated"] = heatmap_DF_to_Plot["longitude"]

    scatterPlotAxis = sns.scatterplot(data=heatmap_DF_to_Plot,x="longitudeTranslated",y="latitude",
                    hue=dose_type, hue_norm=hue_range, palette=palette,
                    zorder=10,
                    marker="s",s=heatmap_s,edgecolor=edgecolor,
                    legend=False,
                    )#ax=axToPlotOn)

    if plot_colorbar is True:
        norm = plt.Normalize(hue_range[0], hue_range[1])
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])

        # Remove the legend and add a colorbar
        #scatterPlotAxis.get_legend().remove()
        #colorbar = scatterPlotAxis.figure.colorbar(sm,label=legend_label,shrink=0.4)
        # Use the potentially updated legend label
        colorbar = scatterPlotAxis.figure.colorbar(sm,label=effective_legend_label,orientation="horizontal",ax=plt.gca()) 
    else:
        colorbar = None

    plt.ylim([-90,90])
    plt.xlim([-175,180])
    plt.grid(True)
    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")

    try:
        world = geopandas.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
        world.plot(color="None",edgecolor="black",lw=0.35,ax=scatterPlotAxis,zorder=20)
        if plot_longitude_east is True:
            world['geometry'] = world['geometry'].translate(xoff=360)
            world.plot(color="None",edgecolor="black",lw=0.35,ax=scatterPlotAxis,zorder=20)
            plt.xlim([0,355])
    except:
        print("ERROR: was not able to plot world map - note that with the current geopandas implementation you need internet access to access the world map file. Plotting will continue anyways, but without the outline of Earth.")

    ####################################################################

    #plt.legend(title=dose_type,loc="center left",bbox_to_anchor=(1.1,0.5))

    return scatterPlotAxis, colorbar

def plot_on_spherical_globe(heatmap_DF_to_Plot, 
                            dose_type="edose", 
                           plot_title=None, 
                           palette="Spectral_r", 
                           hue_range=None,
                           legend_label=r"Effective dose ($\mu Sv / hr$)",
                           central_latitude=40.0,
                           central_longitude=0.0):
    """
    Plot data on a 3D spherical globe using Cartopy.
    
    Parameters:
    -----------
    heatmap_DF_to_Plot : pandas.DataFrame
        DataFrame containing latitude, longitude, and dose values
    dose_type : str, default="edose"
        Column name in the DataFrame to use for coloring points
    plot_title : str, optional
        Plot title
    palette : str, default="Spectral_r"
        Matplotlib colormap name
    hue_range : tuple, optional
        Range of values for the colormap (min, max)
    legend_label : str, default=r"Effective dose ($\mu Sv / hr$)"
        Label for the colorbar
    central_latitude : float, default=0
        Latitude for the center of the orthographic projection
    central_longitude : float, default=0
        Longitude for the center of the orthographic projection
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    
    # Create figure with orthographic projection (3D globe view)
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, 30))
    ax = plt.gcf().add_subplot(1, 1, 1, projection=ccrs.Orthographic(central_longitude, central_latitude))
    
    # Add coastlines and features
    ax.coastlines(linewidth=0.5)
    ax.gridlines(linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # Create a heatmap using pcolormesh
    # First, we need to reshape the data into a grid
    
    # Create a regular grid for the heatmap
    # lon_grid = np.linspace(data_df["longitude"].min(), data_df["longitude"].max(), 100)
    # lat_grid = np.linspace(data_df["latitude"].min(), data_df["latitude"].max(), 100)
    lon_mesh, lat_mesh = np.meshgrid(heatmap_DF_to_Plot["longitude"].unique(), heatmap_DF_to_Plot["latitude"].unique())
    
    # Interpolate the data onto the grid
    values = heatmap_DF_to_Plot[dose_type].values
    grid_values = griddata((heatmap_DF_to_Plot["longitude"], heatmap_DF_to_Plot["latitude"]), values, 
                           (lon_mesh, lat_mesh), method='linear', fill_value=np.nan)
    
    # Plot the heatmap
    scatter = ax.pcolormesh(lon_mesh, lat_mesh, grid_values, 
                           transform=ccrs.PlateCarree(),
                           cmap=palette, 
                           alpha=0.8,
                           shading='auto')

    if hue_range is None:
        hue_range = (0,heatmap_DF_to_Plot[dose_type].max())
    norm = plt.Normalize(hue_range[0], hue_range[1])
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    #scatterPlotAxis.get_legend().remove()
    #colorbar = scatterPlotAxis.figure.colorbar(sm,label=legend_label,shrink=0.4)
    colorbar = scatter.figure.colorbar(sm,label=legend_label,orientation="horizontal",ax=plt.gca())
    
    # Set title
    if plot_title:
        plt.title(plot_title)
    
    return plt.gcf()

def plot_dose_map(map_to_plot,
                  plot_title=None,
                  plot_contours=True,
                  levels=3,
                    **kwargs):

    #altitude_to_plot_in_km = altitude_to_plot_in_kft * 0.3048

    axis_no_colorbar, colorbar = create_single_dose_map_plot_plt(map_to_plot,
                                                     **kwargs)

    plt.title(plot_title)

    if plot_contours is True:
        plot_dose_map_contours(map_to_plot,levels=levels,**kwargs)

    return axis_no_colorbar, colorbar

def add_colorbar_to_plot(hue_range, palette, legend_label, scatterPlotAxis=None):
    norm = plt.Normalize(hue_range[0], hue_range[1])
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    #scatterPlotAxis.get_legend().remove()
    if scatterPlotAxis is None:
        colorbar = plt.colorbar(sm,label=legend_label,shrink=0.4)
    else:
        colorbar = scatterPlotAxis.figure.colorbar(sm,label=legend_label,shrink=0.4)
    return colorbar

def create_single_dose_map_plotly(DF_to_use,
                              selected_altitude_in_km,
                              **kwargs):

    if selected_altitude_in_km is not None:
        DF_to_use = DF_to_use[round(DF_to_use["altitude (km)"],4) == selected_altitude_in_km]

    if len(DF_to_use) == 0:
        raise Exception("Error: specified altitude in kilometers did not match any of the altitudes in kilometers in the inputted DataFrame.")

    doseRateMap = px.scatter(DF_to_use, x="longitude",y="latitude",color="adose",
                            symbol_sequence=["square"],
                            range_y=[-90,90],
                            range_x=[0,360],
                            **kwargs)

    doseRateMap.update_traces(marker={'size': 10})
    doseRateMap.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1, range=[-90,90]))
    doseRateMap.update_xaxes(range=[0,360])
    doseRateMap.update_yaxes(range=[-90,90])
    doseRateMap.update_layout(xaxis_scaleanchor="y")

    doseRateMap.update_layout(autosize=False,
                            width=800,
                            height=600)

    doseRateMap.show()

    return doseRateMap

def create_gle_map_animation(results, altitude=12.192, save_gif=False, save_mp4=False):
    """
    Create animations for GLE event data.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing timestamps as keys and dataframes as values.
    altitude : float, optional
        The altitude in km for which to create the animation. Default is 12.192 km.
    save_gif : bool, optional
        Whether to save the animation as a GIF file. Default is False.
    save_mp4 : bool, optional
        Whether to save the animation as an MP4 file. Default is False.
    """
    
    # Find the maximum dose value across all timestamps
    max_dose = 0
    for timestamp, df in results.items():
        data_df = df.query(f'`altitude (km)` == {altitude}')
        if not data_df.empty and 'edose' in data_df.columns:
            current_max = data_df['edose'].max()
            max_dose = max(max_dose, current_max)
    
    # Round up to the nearest 5 or 10 for a cleaner colorbar
    if max_dose > 100:
        max_dose = np.ceil(max_dose / 10) * 10
    else:
        max_dose = np.ceil(max_dose / 5) * 5
    
    # Create figure for the animation
    fig = plt.figure(figsize=(10, 6))
    
    # Get sorted timestamps for consistent animation order
    timestamps = sorted(results.keys())
    
    # Function to update the animation frame
    def update(timestamp_idx):
        plt.clf()
        timestamp = timestamps[timestamp_idx]
        df = results[timestamp]
        
        # Extract data for the specified altitude
        data_df = df.query(f'`altitude (km)` == {altitude}')
        
        from AniMAIRE.dose_plotting import plot_dose_map
        outputted_figure = plot_dose_map(data_df,
                            plot_title=f'GLE74 Effective Dose Rate at {timestamp} (Altitude: {altitude} km)',
                            #colorbar_label=f"Effective Dose Rate (μSv/hr)",
                            #central_latitude=50.0,
                            hue_range=(0, max_dose))
        
        return [plt.gca()]

    # Create the animation - make sure to import animation from matplotlib
    ani = animation.FuncAnimation(fig, update, frames=len(timestamps), 
                                 interval=500, blit=False)

    # Save the animation to HTML5 video
    video = ani.to_jshtml()
    
    if save_gif:
        # Save the animation to an animated GIF file
        gif_filename = f"GLE_animation_{altitude}km.gif"
        
        # Create a temporary directory if it doesn't exist
        if not os.path.exists('animation_temp'):
            os.makedirs('animation_temp')
        
        # Save individual frames for GIF creation
        frames = []
        for i in range(len(timestamps)):
            # Update the figure for this frame
            update(i)
            
            # Save the frame
            frame_filename = f'animation_temp/frame_{i:03d}.png'
            plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
            frames.append(frame_filename)
        
        # Use imageio to create the GIF
        with imageio.get_writer(gif_filename, mode='I', duration=0.5) as writer:
            for frame_filename in frames:
                image = imageio.imread(frame_filename)
                writer.append_data(image)
        
        print(f"Animation saved as {gif_filename}")
        
        # Clean up temporary files
        for frame_filename in frames:
            os.remove(frame_filename)

    if save_mp4:
        # Save the animation as an MP4 video
        mp4_filename = f"GLE_animation_{altitude}km.mp4"
        
        # Set up the writer with desired parameters
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=2, metadata=dict(artist='AniMAIRE'), bitrate=1800)
        
        try:
            # Save the animation to MP4
            ani.save(mp4_filename, writer=writer)
            print(f"Animation saved as {mp4_filename}")
        except Exception as e:
            print(f"Error saving MP4: {e}")
            print("If ffmpeg is not installed, you may need to install it with: pip install ffmpeg-python")

    # Display the animation
    print(f"GLE Animation at {altitude} km (max dose: {max_dose} μSv/hr):")
    return HTML(video)  # Return HTML object instead of displaying it directly

def create_gle_globe_animation(results, altitude=12.192, save_gif=False, save_mp4=False):
    """
    Create animations for GLE event data.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing timestamps as keys and dataframes as values.
    altitude : float, optional
        The altitude in km for which to create the animation. Default is 12.192 km.
    save_gif : bool, optional
        Whether to save the animation as a GIF file. Default is False.
    save_mp4 : bool, optional
        Whether to save the animation as an MP4 file. Default is False.
    """
    
    # Find the maximum dose value across all timestamps
    max_dose = 0
    for timestamp, df in results.items():
        data_df = df.query(f'`altitude (km)` == {altitude}')
        if not data_df.empty and 'edose' in data_df.columns:
            current_max = data_df['edose'].max()
            max_dose = max(max_dose, current_max)
    
    # Round up to the nearest 5 or 10 for a cleaner colorbar
    if max_dose > 100:
        max_dose = np.ceil(max_dose / 10) * 10
    else:
        max_dose = np.ceil(max_dose / 5) * 5
    
    # Create figure for the animation
    fig = plt.figure(figsize=(10, 6))
    
    # Get sorted timestamps for consistent animation order
    timestamps = sorted(results.keys())
    
    # Function to update the animation frame
    def update(timestamp_idx):
        plt.clf()
        timestamp = timestamps[timestamp_idx]
        df = results[timestamp]
        
        # Extract data for the specified altitude
        data_df = df.query(f'`altitude (km)` == {altitude}')
        
        outputted_figure = plot_on_spherical_globe(data_df,
                            plot_title=f'GLE74 Effective Dose Rate at {timestamp} (Altitude: {altitude} km)',
                            legend_label=f"Effective Dose Rate (μSv/hr)",
                            central_latitude=50.0,
                            hue_range=(0, max_dose))
        
        return [plt.gca()]

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(timestamps), 
                                 interval=500, blit=False)

    # Save the animation to HTML5 video
    video = ani.to_jshtml()
    
    if save_gif:
        # Save the animation to an animated GIF file
        gif_filename = f"GLE_animation_{altitude}km.gif"
        
        # Create a temporary directory if it doesn't exist
        if not os.path.exists('animation_temp'):
            os.makedirs('animation_temp')
        
        # Save individual frames for GIF creation
        frames = []
        for i in range(len(timestamps)):
            # Update the figure for this frame
            update(i)
            
            # Save the frame
            frame_filename = f'animation_temp/frame_{i:03d}.png'
            plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
            frames.append(frame_filename)
        
        # Use imageio to create the GIF
        with imageio.get_writer(gif_filename, mode='I', duration=0.5) as writer:
            for frame_filename in frames:
                image = imageio.imread(frame_filename)
                writer.append_data(image)
        
        print(f"Animation saved as {gif_filename}")
        
        # Clean up temporary files
        for frame_filename in frames:
            os.remove(frame_filename)

    if save_mp4:
        # Save the animation as an MP4 video
        mp4_filename = f"GLE_animation_{altitude}km_globe.mp4"
        
        # Set up the writer with desired parameters
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=2, metadata=dict(artist='AniMAIRE'), bitrate=1800)
        
        try:
            # Save the animation to MP4
            ani.save(mp4_filename, writer=writer)
            print(f"Animation saved as {mp4_filename}")
        except Exception as e:
            print(f"Error saving MP4: {e}")
            print("If ffmpeg is not installed, you may need to install it with: pip install ffmpeg-python")

    # Display the animation
    print(f"GLE Animation at {altitude} km (max dose: {max_dose} μSv/hr):")
    return HTML(video)  # Return HTML object instead of displaying it directly