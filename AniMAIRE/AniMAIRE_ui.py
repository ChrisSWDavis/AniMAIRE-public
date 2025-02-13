import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import messagebox
import datetime as dt
import threading
import pandas as pd
import numpy as np
import math
from PIL import Image, ImageTk
import os
from .AniMAIRE import (
    run_from_DLR_cosmic_ray_model,
    run_from_power_law_gaussian_distribution,
    run_from_double_power_law_gaussian_distribution,
    run_from_power_law_Beeck_gaussian_distribution,
    run_from_spectra
)
from .dose_plotting import create_single_dose_map_plotly
from .dose_plotting import plot_dose_map
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import tempfile
import os
import webbrowser
import geopandas as gpd
import json

def generate_europe_grid():
    """Generate a grid of lat/lon points covering Europe."""
    # Europe approximate bounds
    lat_min, lat_max = 35, 70  # From Southern Greece to Northern Scandinavia
    lon_min, lon_max = -10, 30  # From Western Spain to Eastern Finland
    
    # Create 7x7 grid (49 points)
    lats = np.linspace(lat_min, lat_max, 7)
    lons = np.linspace(lon_min, lon_max, 7)
    
    # Create grid points
    grid_points = []
    for lat in lats:
        for lon in lons:
            grid_points.append((lat, lon))
    
    return grid_points

def generate_custom_grid(lat_min, lat_max, lon_min, lon_max, n_points):
    """Generate a custom grid of lat/lon points."""
    n_per_side = int(np.sqrt(n_points))
    lats = np.linspace(lat_min, lat_max, n_per_side)
    lons = np.linspace(lon_min, lon_max, n_per_side)
    
    grid_points = []
    for lat in lats:
        for lon in lons:
            grid_points.append((lat, lon))
    return grid_points

PRESET_REGIONS = {
    "Europe": {"lat_min": 35, "lat_max": 70, "lon_min": -10, "lon_max": 30},
    "North America": {"lat_min": 25, "lat_max": 70, "lon_min": -130, "lon_max": -60},
    "Asia": {"lat_min": 10, "lat_max": 60, "lon_min": 60, "lon_max": 140},
    "Global": {"lat_min": -90, "lat_max": 90, "lon_min": -180, "lon_max": 180},
}

class LocationSelector(ttk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Location Selection")
        self.geometry("600x500")
        
        # Mode selection
        self.mode_frame = ttk.Labelframe(self, text="Selection Mode", padding=10)
        self.mode_frame.pack(fill="x", padx=10, pady=5)
        
        self.mode = tk.StringVar(value="preset")
        ttk.Radiobutton(self.mode_frame, text="Preset Regions", variable=self.mode, 
                       value="preset", command=self._toggle_mode).pack(side="left", padx=5)
        ttk.Radiobutton(self.mode_frame, text="Custom Region", variable=self.mode,
                       value="custom", command=self._toggle_mode).pack(side="left", padx=5)
        ttk.Radiobutton(self.mode_frame, text="Single Point", variable=self.mode,
                       value="point", command=self._toggle_mode).pack(side="left", padx=5)
        ttk.Radiobutton(self.mode_frame, text="Multiple Points", variable=self.mode,
                       value="points", command=self._toggle_mode).pack(side="left", padx=5)
        
        # Preset selection
        self.preset_frame = ttk.Labelframe(self, text="Preset Region", padding=10)
        self.preset_frame.pack(fill="x", padx=10, pady=5)
        
        self.preset = tk.StringVar(value="Europe")
        for region in PRESET_REGIONS:
            ttk.Radiobutton(self.preset_frame, text=region, variable=self.preset,
                          value=region).pack(side="left", padx=5)
        
        # Custom region frame
        self.custom_frame = ttk.Labelframe(self, text="Custom Region", padding=10)
        self.custom_frame.pack(fill="x", padx=10, pady=5)
        
        # Grid layout for custom region inputs
        ttk.Label(self.custom_frame, text="Latitude Min:").grid(row=0, column=0, padx=5)
        self.lat_min = ttk.Entry(self.custom_frame, width=10)
        self.lat_min.grid(row=0, column=1, padx=5)
        
        ttk.Label(self.custom_frame, text="Latitude Max:").grid(row=0, column=2, padx=5)
        self.lat_max = ttk.Entry(self.custom_frame, width=10)
        self.lat_max.grid(row=0, column=3, padx=5)
        
        ttk.Label(self.custom_frame, text="Longitude Min:").grid(row=1, column=0, padx=5)
        self.lon_min = ttk.Entry(self.custom_frame, width=10)
        self.lon_min.grid(row=1, column=1, padx=5)
        
        ttk.Label(self.custom_frame, text="Longitude Max:").grid(row=1, column=2, padx=5)
        self.lon_max = ttk.Entry(self.custom_frame, width=10)
        self.lon_max.grid(row=1, column=3, padx=5)
        
        ttk.Label(self.custom_frame, text="Number of Points:").grid(row=2, column=0, columnspan=2, padx=5)
        self.n_points = ttk.Entry(self.custom_frame, width=10)
        self.n_points.grid(row=2, column=2, columnspan=2, padx=5)
        self.n_points.insert(0, "49")  # Default to 7x7 grid
        
        # Single point frame
        self.point_frame = ttk.Labelframe(self, text="Single Point", padding=10)
        self.point_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(self.point_frame, text="Latitude:").pack(side="left", padx=5)
        self.point_lat = ttk.Entry(self.point_frame, width=10)
        self.point_lat.pack(side="left", padx=5)
        
        ttk.Label(self.point_frame, text="Longitude:").pack(side="left", padx=5)
        self.point_lon = ttk.Entry(self.point_frame, width=10)
        self.point_lon.pack(side="left", padx=5)
        
        # Multiple points frame
        self.points_frame = ttk.Labelframe(self, text="Multiple Points", padding=10)
        self.points_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.points_text = tk.Text(self.points_frame, height=6)
        self.points_text.pack(fill="both", expand=True, padx=5, pady=5)
        ttk.Label(self.points_frame, 
                 text="Enter points as 'latitude,longitude' one per line").pack()
        
        # Buttons
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(self.button_frame, text="OK", command=self._on_ok).pack(side="right", padx=5)
        ttk.Button(self.button_frame, text="Cancel", command=self.destroy).pack(side="right", padx=5)
        
        self.result = None
        self._toggle_mode()
    
    def _toggle_mode(self):
        mode = self.mode.get()
        for frame in [self.preset_frame, self.custom_frame, self.point_frame, self.points_frame]:
            frame.pack_forget()
        
        if mode == "preset":
            self.preset_frame.pack(fill="x", padx=10, pady=5)
        elif mode == "custom":
            self.custom_frame.pack(fill="x", padx=10, pady=5)
        elif mode == "point":
            self.point_frame.pack(fill="x", padx=10, pady=5)
        elif mode == "points":
            self.points_frame.pack(fill="both", expand=True, padx=10, pady=5)
    
    def _on_ok(self):
        mode = self.mode.get()
        try:
            if mode == "preset":
                region = PRESET_REGIONS[self.preset.get()]
                self.result = generate_custom_grid(
                    region["lat_min"], region["lat_max"],
                    region["lon_min"], region["lon_max"],
                    49  # Default to 7x7 grid
                )
            elif mode == "custom":
                self.result = generate_custom_grid(
                    float(self.lat_min.get()),
                    float(self.lat_max.get()),
                    float(self.lon_min.get()),
                    float(self.lon_max.get()),
                    int(self.n_points.get())
                )
            elif mode == "point":
                self.result = [(float(self.point_lat.get()), float(self.point_lon.get()))]
            elif mode == "points":
                points = []
                for line in self.points_text.get("1.0", tk.END).strip().split("\n"):
                    if line.strip():
                        lat, lon = map(float, line.strip().split(","))
                        points.append((lat, lon))
                self.result = points
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))

class ImprovedAniMAIREUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AniMAIRE UI")
        self.root.geometry("1200x800")
        
        # Create and display logo
        logo_frame = ttk.Frame(root)
        logo_frame.pack(fill="x", padx=20, pady=10)
        
        # Create logo canvas
        logo_size = 120
        self.logo_canvas = tk.Canvas(logo_frame, width=logo_size, height=logo_size, 
                                   highlightthickness=0, bg=self.root.cget('bg'))
        self.logo_canvas.pack(side="left")
        
        # Load and display the logo image
        try:
            # Get the path to the logo relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(os.path.dirname(current_dir), 'AniMAIRE_logo_png.png')
            
            # Load and resize the image
            logo_img = Image.open(logo_path)
            logo_img = logo_img.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            
            # Display the image
            self.logo_canvas.create_image(logo_size//2, logo_size//2, image=self.logo_photo)
        except Exception as e:
            print(f"Failed to load logo: {e}")
            # Fallback to text if image fails to load
            self.logo_canvas.create_text(logo_size//2, logo_size//2, text="AniMAIRE",
                                       font=("Helvetica", 16, "bold"))
        
        # Add title next to logo
        title_frame = ttk.Frame(logo_frame)
        title_frame.pack(side="left", padx=20)
        
        title_label = ttk.Label(title_frame, text="AniMAIRE",
                               font=("Helvetica", 24, "bold"))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, 
                                 text="Anisotropic Model for Atmospheric\nIonizing Radiation Environment",
                                 font=("Helvetica", 12))
        subtitle_label.pack()
        
        # GLE presets
        self.gle_presets = {
            "GLE 42 (29-Sep-1989)": {
                "model": "POWER_LAW",
                "datetime": "1989-09-29 11:45:00",
                "j0": 2.034e4,
                "gamma": 5.2,
                "delta_gamma": 0.2,
                "sigma": 0.4,
                "ref_lat": 90.0,
                "ref_lon": 0.0,
                "description": "One of the largest GLEs in solar cycle 22, associated with X9.8 flare",
                "references": [
                    "Lovell et al. (1998) JGR, 103(A10), 23733-23742",
                    "Miroshnichenko et al. (2000) Space Sci. Rev., 91(3-4), 615-715"
                ]
            },
            "GLE 69 (20-Jan-2005)": {
                "model": "DOUBLE_POWER_LAW",
                "datetime": "2005-01-20 06:49:00",
                "j0": 1.366e4,
                "gamma": 4.5,
                "delta_gamma": 0.1,
                "sigma1": 0.3,
                "sigma2": 0.5,
                "b": 0.2,
                "alpha_prime": 0.785,
                "ref_lat": 80.0,
                "ref_lon": 90.0,
                "description": "Highest intensity GLE of solar cycle 23, extreme relativistic proton acceleration",
                "references": [
                    "Bütikofer et al. (2008) JGR: Space Physics, 113(A8)",
                    "Plainaki et al. (2007) JGR: Space Physics, 112(A4)"
                ]
            },
            "GLE 70 (13-Dec-2006)": {
                "model": "POWER_LAW_BEECK",
                "datetime": "2006-12-13 02:50:00",
                "j0": 8.532e3,
                "gamma": 5.0,
                "delta_gamma": 0.15,
                "beeck_a": 1.2,
                "beeck_b": 2.5,
                "ref_lat": 85.0,
                "ref_lon": -90.0,
                "description": "Notable for its unusual timing in solar minimum and strong anisotropy",
                "references": [
                    "Vashenyuk et al. (2008) Adv. Space Res., 41(6), 926-935",
                    "Moraal et al. (2009) JGR: Space Physics, 114(A5)"
                ]
            },
            "GLE 71 (17-May-2012)": {
                "model": "POWER_LAW",
                "datetime": "2012-05-17 01:50:00",
                "j0": 5.234e3,
                "gamma": 4.8,
                "delta_gamma": 0.12,
                "sigma": 0.35,
                "ref_lat": 87.0,
                "ref_lon": 45.0,
                "description": "First GLE of solar cycle 24, moderate intensity but strong anisotropy",
                "references": [
                    "Mishev et al. (2014) JGR: Space Physics, 119(2), 670-679",
                    "Papaioannou et al. (2014) Sol. Phys., 289(1), 423-436"
                ]
            },
            "GLE 72 (10-Sep-2017)": {
                "model": "DOUBLE_POWER_LAW",
                "datetime": "2017-09-10 16:15:00",
                "j0": 9.845e3,
                "gamma": 4.2,
                "delta_gamma": 0.08,
                "sigma1": 0.25,
                "sigma2": 0.4,
                "b": 0.15,
                "alpha_prime": 0.698,
                "ref_lat": 83.0,
                "ref_lon": 135.0,
                "description": "Second GLE of solar cycle 24, associated with X8.2 flare and very fast CME",
                "references": [
                    "Mishev et al. (2018) Sol. Phys., 293(10), 136",
                    "Kurt et al. (2019) JGR: Space Physics, 124(8), 5578-5586"
                ]
            }
        }
        
        # Theme handling
        self.current_theme = "flatly"
        self.style = ttk.Style(self.current_theme)
        
        # Keyboard shortcuts and bindings
        self.root.bind("<Control-r>", lambda e: self._start_simulation())
        self.root.bind("<Control-l>", lambda e: self._show_location_selector())
        self.root.bind("<Control-d>", lambda e: self._toggle_theme())
        self.root.bind("<F1>", lambda e: self._show_help())
        
        # Create status bar first so we can update it
        self.status_bar = ttk.Label(root, text="Ready", relief="sunken", padding=(5, 2))
        self.status_bar.pack(side="bottom", fill="x")
        
        # Create main toolbar
        self.toolbar = ttk.Frame(root)
        self.toolbar.pack(fill="x", padx=5, pady=2)
        
        # Quick access buttons with reliable Unicode symbols
        ttk.Button(self.toolbar, text="\u25B6 Run (Ctrl+R)", 
                  command=self._start_simulation).pack(side="left", padx=2)
        ttk.Button(self.toolbar, text="\u2295 Location (Ctrl+L)", 
                  command=self._show_location_selector).pack(side="left", padx=2)
        ttk.Button(self.toolbar, text="\u2399 Save Results", 
                  command=self._save_results).pack(side="left", padx=2)
        ttk.Button(self.toolbar, text="\u229E Export Plot", 
                  command=self._export_plot).pack(side="left", padx=2)
        
        # Right-aligned toolbar items
        ttk.Button(self.toolbar, text="\u25D0 Theme (Ctrl+D)", 
                  command=self._toggle_theme).pack(side="right", padx=2)
        ttk.Button(self.toolbar, text="\u2753 Help (F1)", 
                  command=self._show_help).pack(side="right", padx=2)
        
        # Set up the menu bar with refined styling
        self._create_menu()

        # Main content area
        self.main_paned = ttk.PanedWindow(root, orient="horizontal")
        self.main_paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left sidebar for favorites/history
        self.sidebar = ttk.Notebook(self.main_paned)
        self.favorites_frame = ttk.Frame(self.sidebar)
        self.history_frame = ttk.Frame(self.sidebar)
        self.sidebar.add(self.favorites_frame, text="\u2605 Favorites")
        self.sidebar.add(self.history_frame, text="\u231A History")
        self.main_paned.add(self.sidebar, weight=1)
        
        # Main notebook area
        self.notebook_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.notebook_frame, weight=4)
        
        # Use a Notebook to separate Controls and Results
        self.notebook = ttk.Notebook(self.notebook_frame)
        self.notebook.pack(fill="both", expand=True)

        self.settings_frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.settings_frame, text="\u2699 Settings")

        self.results_frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.results_frame, text="\u2295 Results")
        
        # Initialize data
        self.grid_points = generate_europe_grid()
        self.simulation_history = []
        self.favorites = []
        
        # Create widgets
        self._create_settings_widgets()
        self._create_results_widgets()
        self._create_favorites_widgets()
        self._create_history_widgets()
        
        self.result = None
        
        # Add tooltips to all major controls
        self._add_tooltips()
        
        # Update status
        self._update_status("Ready to start simulation")

    def _create_menu(self):
        menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=False)
        file_menu.add_command(label="New Simulation", accelerator="Ctrl+N")
        file_menu.add_command(label="Open...", accelerator="Ctrl+O")
        file_menu.add_command(label="Save Results...", accelerator="Ctrl+S", 
                            command=self._save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Export Plot...", command=self._export_plot)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(menu_bar, tearoff=False)
        edit_menu.add_command(label="Copy Results", accelerator="Ctrl+C")
        edit_menu.add_command(label="Copy Plot", accelerator="Ctrl+Shift+C")
        menu_bar.add_cascade(label="Edit", menu=edit_menu)
        
        # View menu
        view_menu = tk.Menu(menu_bar, tearoff=False)
        view_menu.add_command(label="Toggle Theme", accelerator="Ctrl+D",
                            command=self._toggle_theme)
        view_menu.add_separator()
        view_menu.add_checkbutton(label="Show Sidebar")
        view_menu.add_checkbutton(label="Show Toolbar")
        view_menu.add_checkbutton(label="Show Status Bar")
        menu_bar.add_cascade(label="View", menu=view_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menu_bar, tearoff=False)
        tools_menu.add_command(label="Batch Processing...")
        tools_menu.add_command(label="Compare Simulations...")
        tools_menu.add_command(label="Generate Report...")
        menu_bar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=False)
        help_menu.add_command(label="Quick Start Guide", accelerator="F1",
                            command=self._show_help)
        help_menu.add_command(label="Keyboard Shortcuts...")
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)

    def _create_favorites_widgets(self):
        """Create the favorites list with saved parameter sets"""
        self.favorites_tree = ttk.Treeview(self.favorites_frame, show="tree")
        self.favorites_tree.pack(fill="both", expand=True)
        
        # Add some sample favorites
        self.favorites_tree.insert("", "end", text="GLE72 Peak Settings")
        self.favorites_tree.insert("", "end", text="Standard Flight Level")

    def _create_history_widgets(self):
        """Create the simulation history view"""
        self.history_tree = ttk.Treeview(self.history_frame, 
                                       columns=("date", "params"),
                                       show="headings")
        self.history_tree.heading("date", text="Date/Time")
        self.history_tree.heading("params", text="Parameters")
        self.history_tree.pack(fill="both", expand=True)

    def _toggle_theme(self):
        """Toggle between light and dark themes"""
        if self.current_theme == "flatly":
            self.current_theme = "darkly"
        else:
            self.current_theme = "flatly"
        self.style = ttk.Style(self.current_theme)
        self._update_status(f"Theme changed to {self.current_theme}")

    def _show_help(self):
        """Show the help dialog"""
        help_text = """
        🚀 Quick Start Guide
        
        1. Select altitude using the slider or presets
        2. Choose location points using the Location button
        3. Set simulation parameters
        4. Click Run or press Ctrl+R
        
        ⌨️ Keyboard Shortcuts:
        Ctrl+R: Run Simulation
        Ctrl+L: Select Location
        Ctrl+D: Toggle Dark Mode
        F1: Show This Help
        """
        messagebox.showinfo("Help", help_text)

    def _save_results(self):
        """Save current results to file"""
        if self.result is None:
            messagebox.showwarning("No Results", 
                                 "Run a simulation first to save results.")
            return
        # Add save file dialog and logic here
        self._update_status("Results saved successfully")

    def _export_plot(self):
        """Export current plot to file"""
        if self.result is None:
            messagebox.showwarning("No Plot", 
                                 "Run a simulation first to export plot.")
            return
        # Add export dialog and logic here
        self._update_status("Plot exported successfully")

    def _add_tooltips(self):
        """Add tooltips to all major controls"""
        tooltips = {
            self.altitude_slider: "Drag to adjust altitude (0-60,000 ft)",
            self.run_button: "Start simulation with current parameters (Ctrl+R)",
        }
        # Add tooltip creation logic here

    def _update_status(self, message):
        """Update the status bar message"""
        self.status_bar.config(text=message)
        
    def run_simulation(self):
        try:
            self._update_status("Running simulation...")
            # Simulated progress update
            for i in range(1, 6):
                self.progress_var.set(i * 20)
                self.root.after(300)
            
            # Get common parameters
            kp_index = int(self.kp_index_entry.get()) if self.kp_index_entry.get() else None
            date_and_time = dt.datetime.strptime(self.datetime_entry.get(), "%Y-%m-%d %H:%M:%S") if self.datetime_entry.get() else None
            
            # Run appropriate model
            model = self.model_choice.get()
            if model == "DLR":
                # Get DLR model parameters
                oulu_count_rate = float(self.oulu_count_rate_entry.get()) if self.param_choice.get() == "OULU" else None
                w_parameter = float(self.w_parameter_entry.get()) if self.param_choice.get() == "W" else None
                
                self.result = run_from_DLR_cosmic_ray_model(
                    OULU_count_rate_in_seconds=oulu_count_rate,
                    W_parameter=w_parameter,
                    Kp_index=kp_index,
                    date_and_time=date_and_time,
                    array_of_lats_and_longs=self.grid_points
                )
            
            elif model == "POWER_LAW":
                # Get power law parameters
                j0 = float(self.j0_entry.get())
                gamma = float(self.gamma_entry.get())
                delta_gamma = float(self.delta_gamma_entry.get())
                sigma = float(self.sigma_entry.get())
                ref_lat = float(self.ref_lat_entry.get())
                ref_lon = float(self.ref_lon_entry.get())
                
                self.result = run_from_power_law_gaussian_distribution(
                    J0=j0, gamma=gamma, deltaGamma=delta_gamma, sigma=sigma,
                    reference_pitch_angle_latitude=ref_lat,
                    reference_pitch_angle_longitude=ref_lon,
                    Kp_index=kp_index,
                    date_and_time=date_and_time,
                    array_of_lats_and_longs=self.grid_points
                )
            
            elif model == "DOUBLE_POWER_LAW":
                # Get double power law parameters
                j0 = float(self.j0_entry.get())
                gamma = float(self.gamma_entry.get())
                delta_gamma = float(self.delta_gamma_entry.get())
                sigma1 = float(self.sigma_entry.get())
                sigma2 = float(self.sigma2_entry.get())
                b = float(self.b_entry.get())
                alpha_prime = float(self.alpha_prime_entry.get())
                ref_lat = float(self.ref_lat_entry.get())
                ref_lon = float(self.ref_lon_entry.get())
                
                self.result = run_from_double_power_law_gaussian_distribution(
                    J0=j0, gamma=gamma, deltaGamma=delta_gamma,
                    sigma_1=sigma1, sigma_2=sigma2, B=b, alpha_prime=alpha_prime,
                    reference_pitch_angle_latitude=ref_lat,
                    reference_pitch_angle_longitude=ref_lon,
                    Kp_index=kp_index,
                    date_and_time=date_and_time,
                    array_of_lats_and_longs=self.grid_points
                )
            
            elif model == "POWER_LAW_BEECK":
                # Get Beeck model parameters
                j0 = float(self.j0_entry.get())
                gamma = float(self.gamma_entry.get())
                delta_gamma = float(self.delta_gamma_entry.get())
                a = float(self.beeck_a_entry.get())
                b = float(self.beeck_b_entry.get())
                ref_lat = float(self.ref_lat_entry.get())
                ref_lon = float(self.ref_lon_entry.get())
                
                self.result = run_from_power_law_Beeck_gaussian_distribution(
                    J0=j0, gamma=gamma, deltaGamma=delta_gamma,
                    A=a, B=b,
                    reference_pitch_angle_latitude=ref_lat,
                    reference_pitch_angle_longitude=ref_lon,
                    Kp_index=kp_index,
                    date_and_time=date_and_time,
                    array_of_lats_and_longs=self.grid_points
                )
            
            elif model == "CUSTOM":
                # Get custom spectrum parameters
                formula = self.spectrum_formula.get("1.0", "end-1c")
                var_text = self.variables_text.get("1.0", "end-1c")
                
                # Parse variables
                variables = {}
                for line in var_text.split("\n"):
                    if "=" in line:
                        name, value = line.split("=")
                        variables[name.strip()] = float(value.strip())
                
                # Create the spectrum function
                def custom_spectrum(R):
                    namespace = {**variables, "R": R, "np": np}
                    return eval(formula, {"__builtins__": None}, namespace)
                
                ref_lat = float(self.ref_lat_entry.get())
                ref_lon = float(self.ref_lon_entry.get())
                
                # Run with custom spectrum
                self.result = run_from_spectra(
                    proton_rigidity_spectrum=custom_spectrum,
                    reference_pitch_angle_latitude=ref_lat,
                    reference_pitch_angle_longitude=ref_lon,
                    Kp_index=kp_index,
                    date_and_time=date_and_time,
                    array_of_lats_and_longs=self.grid_points
                )
            
            # Display results
            self.display_result(self.result)
            self.notebook.select(self.results_frame)
            
            # Add to history with model-specific parameters
            history_params = f"Model: {model}, "
            if model == "DLR":
                history_params += f"OULU: {oulu_count_rate}, W: {w_parameter}"
            elif model == "POWER_LAW":
                history_params += f"J₀: {j0}, γ: {gamma}, Δγ: {delta_gamma}, σ: {sigma}"
            elif model == "DOUBLE_POWER_LAW":
                history_params += f"J₀: {j0}, γ: {gamma}, Δγ: {delta_gamma}, σ₁: {sigma1}, σ₂: {sigma2}"
            elif model == "POWER_LAW_BEECK":
                history_params += f"J₀: {j0}, γ: {gamma}, Δγ: {delta_gamma}, A: {a}, B: {b}"
            elif model == "CUSTOM":
                history_params += "Custom Spectrum"
            
            self.simulation_history.append({
                "date": dt.datetime.now(),
                "params": history_params
            })
            self._update_history_view()
            
            self._update_status("Simulation completed successfully")
        except Exception as e:
            self._update_status("Error: " + str(e))
            messagebox.showerror("Error", str(e))

    def display_result(self, result):
        """Display simulation results in the table"""
        for item in self.output_table.get_children():
            self.output_table.delete(item)
        columns = list(result.columns)
        self.output_table["columns"] = columns
        self.output_table["show"] = "headings"
        for col in columns:
            self.output_table.heading(col, text=col)
            self.output_table.column(col, width=100, anchor="center")
        for _, row in result.iterrows():
            self.output_table.insert("", "end", values=list(row))

    def _update_history_view(self):
        """Update the history view with latest simulations"""
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        for sim in reversed(self.simulation_history):
            self.history_tree.insert("", "end", values=(
                sim["date"].strftime("%Y-%m-%d %H:%M:%S"),
                sim["params"]
            ))

    def _show_location_selector(self):
        selector = LocationSelector(self.root)
        self.root.wait_window(selector)
        if selector.result:
            self.grid_points = selector.result
            n_points = len(self.grid_points)
            messagebox.showinfo("Success", f"Selected {n_points} location points")

    def _create_settings_widgets(self):
        # Group inputs in labeled frames
        input_frame = ttk.Labelframe(self.settings_frame, text="Simulation Parameters", padding=10)
        input_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create a frame for model selection
        model_frame = ttk.Labelframe(input_frame, text="Model Selection", padding=10)
        model_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)

        # Add GLE presets frame after model selection
        gle_frame = ttk.Labelframe(input_frame, text="GLE Presets", padding=10)
        gle_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        
        # Create GLE selection combobox
        ttk.Label(gle_frame, text="Historic GLE Events:").pack(side="left", padx=5)
        self.gle_var = ttk.StringVar()
        self.gle_combo = ttk.Combobox(gle_frame, textvariable=self.gle_var, 
                                     values=list(self.gle_presets.keys()),
                                     width=30, state="readonly")
        self.gle_combo.pack(side="left", padx=5)
        
        # Add Load button
        ttk.Button(gle_frame, text="Load GLE Parameters", 
                  command=self._load_gle_preset).pack(side="left", padx=5)
        
        # Add description label
        self.gle_description = ttk.Label(gle_frame, text="", wraplength=400)
        self.gle_description.pack(side="left", padx=5)
        
        # Bind selection change
        self.gle_combo.bind('<<ComboboxSelected>>', self._update_gle_description)

        self.model_choice = tk.StringVar(value="DLR")
        models = [
            ("DLR Cosmic Ray Model", "DLR"),
            ("Power Law + Gaussian", "POWER_LAW"),
            ("Double Power Law + Gaussian", "DOUBLE_POWER_LAW"),
            ("Power Law + Beeck", "POWER_LAW_BEECK"),
            ("Custom Spectrum", "CUSTOM")
        ]
        
        for text, value in models:
            ttk.Radiobutton(model_frame, text=text, variable=self.model_choice, 
                           value=value, command=self._update_model_params).pack(side="left", padx=5)

        # Create frames for different model parameters
        self.dlr_frame = ttk.Frame(input_frame)
        self.power_law_frame = ttk.Frame(input_frame)
        self.double_power_law_frame = ttk.Frame(input_frame)
        self.beeck_frame = ttk.Frame(input_frame)
        self.custom_frame = ttk.Frame(input_frame)

        # DLR Model parameters
        ttk.Label(self.dlr_frame, text="Parameter Choice:").grid(row=0, column=0, sticky="w", pady=5)
        self.param_choice = tk.StringVar(value="OULU")
        ttk.Radiobutton(self.dlr_frame, text="OULU Count Rate (counts/s)", variable=self.param_choice, 
                        value="OULU", command=self.toggle_entries).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(self.dlr_frame, text="W Parameter (MeV/nucleon)", variable=self.param_choice,
                        value="W", command=self.toggle_entries).grid(row=0, column=2, padx=5)

        ttk.Label(self.dlr_frame, text="OULU Count Rate (counts/s):").grid(row=1, column=0, sticky="w", pady=5)
        self.oulu_count_rate_entry = ttk.Entry(self.dlr_frame)
        self.oulu_count_rate_entry.grid(row=1, column=1, columnspan=2, sticky="we", pady=5)
        self.oulu_count_rate_entry.insert(0, "100")

        ttk.Label(self.dlr_frame, text="W Parameter (MeV/nucleon):").grid(row=2, column=0, sticky="w", pady=5)
        self.w_parameter_entry = ttk.Entry(self.dlr_frame)
        self.w_parameter_entry.grid(row=2, column=1, columnspan=2, sticky="we", pady=5)
        self.w_parameter_entry.insert(0, "100")

        # Power Law parameters
        ttk.Label(self.power_law_frame, text="J₀ (particles/cm²/s/sr/GV):").grid(row=0, column=0, sticky="w", pady=5)
        self.j0_entry = ttk.Entry(self.power_law_frame)
        self.j0_entry.grid(row=0, column=1, sticky="we", pady=5)
        self.j0_entry.insert(0, "100")  # Default J₀
        
        ttk.Label(self.power_law_frame, text="γ (spectral index):").grid(row=1, column=0, sticky="w", pady=5)
        self.gamma_entry = ttk.Entry(self.power_law_frame)
        self.gamma_entry.grid(row=1, column=1, sticky="we", pady=5)
        self.gamma_entry.insert(0, "2.5")  # Default gamma
        
        ttk.Label(self.power_law_frame, text="Δγ (spectral index variation):").grid(row=2, column=0, sticky="w", pady=5)
        self.delta_gamma_entry = ttk.Entry(self.power_law_frame)
        self.delta_gamma_entry.grid(row=2, column=1, sticky="we", pady=5)
        self.delta_gamma_entry.insert(0, "0.1")  # Default delta gamma
        
        ttk.Label(self.power_law_frame, text="σ (pitch angle width, radians):").grid(row=3, column=0, sticky="w", pady=5)
        self.sigma_entry = ttk.Entry(self.power_law_frame)
        self.sigma_entry.grid(row=3, column=1, sticky="we", pady=5)
        self.sigma_entry.insert(0, "0.5")  # Default sigma

        # Double Power Law additional parameters
        ttk.Label(self.double_power_law_frame, text="J₀ (particles/cm²/s/sr/GV):").grid(row=0, column=0, sticky="w", pady=5)
        self.j0_entry = ttk.Entry(self.double_power_law_frame)
        self.j0_entry.grid(row=0, column=1, sticky="we", pady=5)
        self.j0_entry.insert(0, "100")  # Default J₀
        
        ttk.Label(self.double_power_law_frame, text="γ (spectral index):").grid(row=1, column=0, sticky="w", pady=5)
        self.gamma_entry = ttk.Entry(self.double_power_law_frame)
        self.gamma_entry.grid(row=1, column=1, sticky="we", pady=5)
        self.gamma_entry.insert(0, "2.5")  # Default gamma
        
        ttk.Label(self.double_power_law_frame, text="Δγ (spectral index variation):").grid(row=2, column=0, sticky="w", pady=5)
        self.delta_gamma_entry = ttk.Entry(self.double_power_law_frame)
        self.delta_gamma_entry.grid(row=2, column=1, sticky="we", pady=5)
        self.delta_gamma_entry.insert(0, "0.1")  # Default delta gamma
        
        ttk.Label(self.double_power_law_frame, text="σ₁ (primary pitch angle width, radians):").grid(row=3, column=0, sticky="w", pady=5)
        self.sigma_entry = ttk.Entry(self.double_power_law_frame)
        self.sigma_entry.grid(row=3, column=1, sticky="we", pady=5)
        self.sigma_entry.insert(0, "0.5")  # Default sigma
        
        ttk.Label(self.double_power_law_frame, text="σ₂ (secondary pitch angle width, radians):").grid(row=4, column=0, sticky="w", pady=5)
        self.sigma2_entry = ttk.Entry(self.double_power_law_frame)
        self.sigma2_entry.grid(row=4, column=1, sticky="we", pady=5)
        self.sigma2_entry.insert(0, "0.3")  # Default sigma2
        
        ttk.Label(self.double_power_law_frame, text="B (secondary/primary ratio):").grid(row=5, column=0, sticky="w", pady=5)
        self.b_entry = ttk.Entry(self.double_power_law_frame)
        self.b_entry.grid(row=5, column=1, sticky="we", pady=5)
        self.b_entry.insert(0, "0.1")  # Default B
        
        ttk.Label(self.double_power_law_frame, text="α' (secondary peak angle, radians):").grid(row=6, column=0, sticky="w", pady=5)
        self.alpha_prime_entry = ttk.Entry(self.double_power_law_frame)
        self.alpha_prime_entry.grid(row=6, column=1, sticky="we", pady=5)
        self.alpha_prime_entry.insert(0, "0.785")  # Default alpha prime (π/4)

        # Beeck model parameters
        ttk.Label(self.beeck_frame, text="J₀ (particles/cm²/s/sr/GV):").grid(row=0, column=0, sticky="w", pady=5)
        self.j0_entry = ttk.Entry(self.beeck_frame)
        self.j0_entry.grid(row=0, column=1, sticky="we", pady=5)
        self.j0_entry.insert(0, "100")  # Default J₀
        
        ttk.Label(self.beeck_frame, text="γ (spectral index):").grid(row=1, column=0, sticky="w", pady=5)
        self.gamma_entry = ttk.Entry(self.beeck_frame)
        self.gamma_entry.grid(row=1, column=1, sticky="we", pady=5)
        self.gamma_entry.insert(0, "2.5")  # Default gamma
        
        ttk.Label(self.beeck_frame, text="Δγ (spectral index variation):").grid(row=2, column=0, sticky="w", pady=5)
        self.delta_gamma_entry = ttk.Entry(self.beeck_frame)
        self.delta_gamma_entry.grid(row=2, column=1, sticky="we", pady=5)
        self.delta_gamma_entry.insert(0, "0.1")  # Default delta gamma
        
        ttk.Label(self.beeck_frame, text="A (pitch angle parameter):").grid(row=3, column=0, sticky="w", pady=5)
        self.beeck_a_entry = ttk.Entry(self.beeck_frame)
        self.beeck_a_entry.grid(row=3, column=1, sticky="we", pady=5)
        self.beeck_a_entry.insert(0, "1.0")  # Default A parameter
        
        ttk.Label(self.beeck_frame, text="B (pitch angle exponent):").grid(row=4, column=0, sticky="w", pady=5)
        self.beeck_b_entry = ttk.Entry(self.beeck_frame)
        self.beeck_b_entry.grid(row=4, column=1, sticky="we", pady=5)
        self.beeck_b_entry.insert(0, "2.0")  # Default B parameter

        # Custom Spectrum frame
        ttk.Label(self.custom_frame, text="Rigidity Spectrum Formula:").grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
        self.spectrum_formula = ttk.Text(self.custom_frame, height=3, width=40)
        self.spectrum_formula.grid(row=1, column=0, columnspan=2, sticky="we", pady=5)
        self.spectrum_formula.insert("1.0", "J0 * (R/R0)^(-gamma) * exp(-deltaGamma * log(R/R0)^2)")
        
        ttk.Label(self.custom_frame, text="Variables:").grid(row=2, column=0, sticky="w", pady=5)
        self.variables_text = ttk.Text(self.custom_frame, height=5, width=40)
        self.variables_text.grid(row=3, column=0, columnspan=2, sticky="we", pady=5)
        self.variables_text.insert("1.0", "J0 = 100\nR0 = 1\ngamma = 2.5\ndeltaGamma = 0.1")

        ttk.Button(self.custom_frame, text="Preview Spectrum", 
                   command=lambda: self._preview_spectrum("CUSTOM")).grid(row=4, column=0, columnspan=2, pady=10)

        # Reference direction frame for all models except DLR
        self.ref_dir_frame = ttk.Frame(input_frame)
        ttk.Label(self.ref_dir_frame, text="Reference Direction (Asymptotic Direction):").grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Label(self.ref_dir_frame, text="Latitude (degrees, -90 to 90):").grid(row=1, column=0, sticky="w", pady=5)
        self.ref_lat_entry = ttk.Entry(self.ref_dir_frame)
        self.ref_lat_entry.grid(row=1, column=1, sticky="we", pady=5)
        self.ref_lat_entry.insert(0, "0.0")
        ttk.Label(self.ref_dir_frame, text="Longitude (degrees, -180 to 180):").grid(row=2, column=0, sticky="w", pady=5)
        self.ref_lon_entry = ttk.Entry(self.ref_dir_frame)
        self.ref_lon_entry.grid(row=2, column=1, sticky="we", pady=5)
        self.ref_lon_entry.insert(0, "45.0")

        # Create altitude frame
        altitude_frame = ttk.Labelframe(input_frame, text="Altitude Selection", padding=10)
        altitude_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)

        # Altitude presets
        self.altitude_preset = tk.StringVar(value="fl350")
        ttk.Radiobutton(altitude_frame, text="FL350 (35,000 ft / 10.7 km)", 
                       variable=self.altitude_preset, value="fl350",
                       command=self._update_altitude).pack(side="left", padx=5)
        ttk.Radiobutton(altitude_frame, text="FL400 (40,000 ft / 12.2 km)", 
                       variable=self.altitude_preset, value="fl400",
                       command=self._update_altitude).pack(side="left", padx=5)
        ttk.Radiobutton(altitude_frame, text="Custom", 
                       variable=self.altitude_preset, value="custom",
                       command=self._update_altitude).pack(side="left", padx=5)

        # Altitude slider frame
        slider_frame = ttk.Frame(altitude_frame)
        slider_frame.pack(fill="x", pady=5)
        
        self.altitude_value = tk.DoubleVar(value=10.7)  # Default to FL350
        self.altitude_label = ttk.Label(slider_frame, 
                                      text=self._format_altitude_label(self.altitude_value.get()))
        self.altitude_label.pack(side="top")
        
        # Slider now goes from 0 to 18.3 km (0 to 60,000 ft)
        self.altitude_slider = ttk.Scale(slider_frame, from_=0, to=18.3,
                                       variable=self.altitude_value,
                                       command=self._on_altitude_change,
                                       length=200)
        self.altitude_slider.pack(side="top", fill="x", padx=10)

        # Add min/max labels under the slider
        label_frame = ttk.Frame(slider_frame)
        label_frame.pack(fill="x", padx=10)
        ttk.Label(label_frame, text="0 ft / 0 km").pack(side="left")
        ttk.Label(label_frame, text="60,000 ft / 18.3 km").pack(side="right")

        ttk.Label(input_frame, text="Kp Index (0-9):").grid(row=3, column=0, sticky="w", pady=5)
        self.kp_index_entry = ttk.Entry(input_frame)
        self.kp_index_entry.grid(row=3, column=1, columnspan=2, sticky="we", pady=5)
        self.kp_index_entry.insert(0, "0")

        ttk.Label(input_frame, text="Date and Time (UTC, YYYY-MM-DD HH:MM:SS):").grid(row=4, column=0, sticky="w", pady=5)
        self.datetime_entry = ttk.Entry(input_frame)
        self.datetime_entry.grid(row=4, column=1, columnspan=2, sticky="we", pady=5)
        self.datetime_entry.insert(0, "2017-09-10 16:00:00")  # GLE72 peak time

        # Replace lat/lon entries with location selection button
        ttk.Button(input_frame, text="Select Locations...", 
                  command=self._show_location_selector).grid(
                      row=5, column=0, columnspan=3, sticky="we", pady=5)

        # Run simulation button with integrated progress indicator
        self.run_button = ttk.Button(input_frame, text="Run Simulation", command=self._start_simulation, bootstyle=SUCCESS)
        self.run_button.grid(row=6, column=0, columnspan=3, pady=10)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(input_frame, variable=self.progress_var, mode="determinate")
        self.progress_bar.grid(row=7, column=0, columnspan=3, sticky="ew", pady=5)

        for i in range(3):
            input_frame.columnconfigure(i, weight=1)

        # Update the layout
        self._update_model_params()

    def _update_model_params(self):
        """Update visible parameter frames based on selected model"""
        # Hide all frames
        for frame in [self.dlr_frame, self.power_law_frame, self.double_power_law_frame, 
                     self.beeck_frame, self.custom_frame, self.ref_dir_frame]:
            frame.grid_forget()

        # Show relevant frames based on selection
        model = self.model_choice.get()
        if model == "DLR":
            self.dlr_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        elif model == "POWER_LAW":
            self.power_law_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
            self.ref_dir_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        elif model == "DOUBLE_POWER_LAW":
            self.double_power_law_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
            self.ref_dir_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        elif model == "POWER_LAW_BEECK":
            self.beeck_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
            self.ref_dir_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        elif model == "CUSTOM":
            self.custom_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
            self.ref_dir_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)

        # Add preview buttons to each model frame
        ttk.Button(self.power_law_frame, text="Preview Spectrum", 
                   command=lambda: self._preview_spectrum("POWER_LAW")).grid(
                       row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(self.double_power_law_frame, text="Preview Spectrum",
                   command=lambda: self._preview_spectrum("DOUBLE_POWER_LAW")).grid(
                       row=7, column=0, columnspan=2, pady=10)
        
        ttk.Button(self.beeck_frame, text="Preview Spectrum",
                   command=lambda: self._preview_spectrum("POWER_LAW_BEECK")).grid(
                       row=5, column=0, columnspan=2, pady=10)
        
        # Replace existing preview button in custom frame
        ttk.Button(self.custom_frame, text="Preview Spectrum", 
                   command=lambda: self._preview_spectrum("CUSTOM")).grid(
                       row=4, column=0, columnspan=2, pady=10)

        # Add preview button to DLR frame
        ttk.Button(self.dlr_frame, text="Preview Spectrum", 
                   command=lambda: self._preview_spectrum("DLR")).grid(
                       row=3, column=0, columnspan=3, pady=10)

    def _preview_spectrum(self, model_type=None):
        """Preview the spectrum in a new window"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Create a new window
            preview_window = ttk.Toplevel(self.root)
            preview_window.title("Spectrum Preview")
            preview_window.geometry("800x600")
            
            # Create rigidity range
            R = np.logspace(-1, 2, 1000)
            
            # Calculate spectrum based on model type
            if model_type is None:
                model_type = self.model_choice.get()
            
            # Initialize spectrum and title
            spectrum = None
            title = ""
            
            try:
                if model_type == "DLR":
                    # Get DLR parameters
                    if self.param_choice.get() == "OULU":
                        oulu = float(self.oulu_count_rate_entry.get() or 100)
                        # Approximate spectrum based on OULU count rate
                        # This is a simplified model for visualization
                        base = oulu / 100  # Normalize to typical values
                        spectrum = base * 100 * (R/1.0)**(-2.7) * np.exp(-0.1 * np.log(R/1.0)**2)
                    else:
                        w = float(self.w_parameter_entry.get() or 100)
                        # Approximate spectrum based on W parameter
                        base = w / 100  # Normalize to typical values
                        spectrum = base * 100 * (R/1.0)**(-2.7) * np.exp(-0.1 * np.log(R/1.0)**2)
                    title = "DLR Model Spectrum (Approximation)"
                
                elif model_type == "CUSTOM":
                    # Get formula and variables
                    formula = self.spectrum_formula.get("1.0", "end-1c").strip()
                    if not formula:
                        raise ValueError("Formula cannot be empty")
                        
                    var_text = self.variables_text.get("1.0", "end-1c").strip()
                    if not var_text:
                        raise ValueError("Variables cannot be empty")
                    
                    # Parse variables
                    variables = {}
                    for line in var_text.split("\n"):
                        if "=" in line:
                            name, value = line.split("=")
                            variables[name.strip()] = float(value.strip())
                    
                    # Create local namespace with variables and numpy functions
                    namespace = {**variables, "R": R, "np": np}
                    
                    # Evaluate formula safely
                    allowed_names = {
                        'np': np,
                        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                        'exp': np.exp, 'log': np.log, 'log10': np.log10,
                        'sqrt': np.sqrt, 'power': np.power,
                        'pi': np.pi, 'e': np.e
                    }
                    namespace = {**variables, "R": R, **allowed_names}
                    
                    # Evaluate formula
                    spectrum = eval(formula, {"__builtins__": None}, namespace)
                    title = "Custom Spectrum"
                    
                elif model_type == "POWER_LAW":
                    j0 = float(self.j0_entry.get() or 100)
                    gamma = float(self.gamma_entry.get() or 2.5)
                    delta_gamma = float(self.delta_gamma_entry.get() or 0.1)
                    spectrum = j0 * (R/1.0)**(-gamma) * np.exp(-delta_gamma * np.log(R/1.0)**2)
                    title = "Power Law Spectrum"
                    
                elif model_type == "DOUBLE_POWER_LAW":
                    j0 = float(self.j0_entry.get() or 100)
                    gamma = float(self.gamma_entry.get() or 2.5)
                    delta_gamma = float(self.delta_gamma_entry.get() or 0.1)
                    sigma1 = float(self.sigma_entry.get() or 0.5)
                    sigma2 = float(self.sigma2_entry.get() or 0.3)
                    b = float(self.b_entry.get() or 0.1)
                    alpha_prime = float(self.alpha_prime_entry.get() or 0.785)
                    
                    # First component: power law with Gaussian modification
                    spectrum1 = j0 * (R/1.0)**(-gamma) * np.exp(-delta_gamma * np.log(R/1.0)**2)
                    # Second component: additional power law with different spectral index
                    spectrum2 = b * j0 * (R/1.0)**(-gamma-1) * np.exp(-delta_gamma * np.log(R/1.0)**2)
                    # Combined spectrum
                    spectrum = spectrum1 + spectrum2
                    title = "Double Power Law Spectrum"
                
                elif model_type == "POWER_LAW_BEECK":
                    j0 = float(self.j0_entry.get() or 100)
                    gamma = float(self.gamma_entry.get() or 2.5)
                    delta_gamma = float(self.delta_gamma_entry.get() or 0.1)
                    a = float(self.beeck_a_entry.get() or 1.0)
                    b = float(self.beeck_b_entry.get() or 2.0)
                    spectrum = j0 * (R/1.0)**(-gamma) * np.exp(-delta_gamma * np.log(R/1.0)**2)
                    title = "Power Law Beeck Spectrum"
                
                if spectrum is None:
                    raise ValueError("Failed to calculate spectrum")
                
            except (ValueError, SyntaxError, NameError, ZeroDivisionError) as e:
                messagebox.showerror("Calculation Error", f"Error calculating spectrum: {str(e)}")
                preview_window.destroy()
                return
            
            # Create figure with two subplots
            fig = plt.figure(figsize=(10, 5))
            
            # Spectrum plot
            ax1 = fig.add_subplot(121)
            ax1.loglog(R, spectrum)
            ax1.set_xlabel("Rigidity (GV)")
            ax1.set_ylabel("Flux (cm⁻² s⁻¹ sr⁻¹ GV⁻¹)")
            ax1.set_title(title)
            ax1.grid(True)
            
            # Pitch angle distribution plot
            ax2 = fig.add_subplot(122)
            alpha = np.linspace(0, np.pi, 1000)
            
            try:
                if model_type == "DLR":
                    # DLR uses isotropic distribution
                    pad = np.ones_like(alpha)
                elif model_type == "POWER_LAW":
                    sigma = float(self.sigma_entry.get() or 0.5)
                    pad = np.exp(-alpha**2 / (2 * sigma**2))
                elif model_type == "DOUBLE_POWER_LAW":
                    sigma1 = float(self.sigma_entry.get() or 0.5)
                    sigma2 = float(self.sigma2_entry.get() or 0.3)
                    b = float(self.b_entry.get() or 0.1)
                    alpha_prime = float(self.alpha_prime_entry.get() or np.pi/4)
                    # First Gaussian component centered at 0
                    pad1 = np.exp(-alpha**2 / (2 * sigma1**2))
                    # Second Gaussian component centered at alpha_prime
                    pad2 = b * np.exp(-(alpha - alpha_prime)**2 / (2 * sigma2**2))
                    # Combined pitch angle distribution
                    pad = pad1 + pad2
                    # Normalize to maximum of 1
                    pad = pad / np.max(pad)
                elif model_type == "POWER_LAW_BEECK":
                    a = float(self.beeck_a_entry.get() or 1.0)
                    b = float(self.beeck_b_entry.get() or 2.0)
                    pad = np.exp(-a * (1 - np.cos(alpha))**b)
                else:
                    pad = np.ones_like(alpha)  # Isotropic for custom spectrum
            except (ValueError, ZeroDivisionError) as e:
                messagebox.showwarning("PAD Warning", f"Using isotropic pitch angle distribution due to error: {str(e)}")
                pad = np.ones_like(alpha)
            
            ax2.plot(np.degrees(alpha), pad)
            ax2.set_xlabel("Pitch Angle (degrees)")
            ax2.set_ylabel("Relative Intensity")
            ax2.set_title("Pitch Angle Distribution")
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Embed plot in window
            canvas = FigureCanvasTkAgg(fig, master=preview_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Add preview controls
            control_frame = ttk.Frame(preview_window)
            control_frame.pack(fill="x", padx=10, pady=5)
            
            # Add pitch angle distribution type selector for custom spectra
            if model_type == "CUSTOM":
                ttk.Label(control_frame, text="Pitch Angle Distribution:").pack(side="left", padx=5)
                pad_type = tk.StringVar(value="isotropic")
                
                def update_pad(*args):
                    try:
                        ax2.clear()
                        if pad_type.get() == "isotropic":
                            pad = np.ones_like(alpha)
                        elif pad_type.get() == "gaussian":
                            sigma = float(pad_sigma_entry.get() or 0.5)
                            pad = np.exp(-alpha**2 / (2 * sigma**2))
                        elif pad_type.get() == "beeck":
                            a = float(pad_a_entry.get() or 1.0)
                            b = float(pad_b_entry.get() or 2.0)
                            pad = np.exp(-a * (1 - np.cos(alpha))**b)
                        
                        ax2.plot(np.degrees(alpha), pad)
                        ax2.set_xlabel("Pitch Angle (degrees)")
                        ax2.set_ylabel("Relative Intensity")
                        ax2.set_title("Pitch Angle Distribution")
                        ax2.grid(True)
                        canvas.draw()
                    except (ValueError, ZeroDivisionError) as e:
                        messagebox.showerror("Error", f"Invalid parameter value: {str(e)}")
                
                ttk.Radiobutton(control_frame, text="Isotropic", variable=pad_type, 
                               value="isotropic", command=update_pad).pack(side="left", padx=5)
                ttk.Radiobutton(control_frame, text="Gaussian", variable=pad_type, 
                               value="gaussian", command=update_pad).pack(side="left", padx=5)
                ttk.Radiobutton(control_frame, text="Beeck", variable=pad_type, 
                               value="beeck", command=update_pad).pack(side="left", padx=5)
                
                # Parameters for different distributions
                param_frame = ttk.Frame(preview_window)
                param_frame.pack(fill="x", padx=10, pady=5)
                
                # Gaussian parameters
                ttk.Label(param_frame, text="σ:").pack(side="left", padx=5)
                pad_sigma_entry = ttk.Entry(param_frame, width=10)
                pad_sigma_entry.pack(side="left", padx=5)
                pad_sigma_entry.insert(0, "0.5")
                pad_sigma_entry.bind("<Return>", update_pad)
                
                # Beeck parameters
                ttk.Label(param_frame, text="A:").pack(side="left", padx=5)
                pad_a_entry = ttk.Entry(param_frame, width=10)
                pad_a_entry.pack(side="left", padx=5)
                pad_a_entry.insert(0, "1.0")
                pad_a_entry.bind("<Return>", update_pad)
                
                ttk.Label(param_frame, text="B:").pack(side="left", padx=5)
                pad_b_entry = ttk.Entry(param_frame, width=10)
                pad_b_entry.pack(side="left", padx=5)
                pad_b_entry.insert(0, "2.0")
                pad_b_entry.bind("<Return>", update_pad)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preview spectrum: {str(e)}")
            if preview_window:
                preview_window.destroy()

    def toggle_entries(self):
        if self.param_choice.get() == "OULU":
            self.oulu_count_rate_entry.config(state="normal")
            self.w_parameter_entry.config(state="disabled")
        else:
            self.oulu_count_rate_entry.config(state="disabled")
            self.w_parameter_entry.config(state="normal")

    def _start_simulation(self):
        self.progress_var.set(0)
        simulation_thread = threading.Thread(target=self.run_simulation, daemon=True)
        simulation_thread.start()

    def _format_altitude_label(self, km_value):
        """Format altitude label to show both km and flight level."""
        feet = km_value * 3280.84  # Convert km to feet
        fl = int(feet / 100)  # Convert feet to flight level
        return f"Altitude: {km_value:.1f} km (FL{fl:03d} / {int(feet):,} ft)"

    def _update_altitude(self):
        preset = self.altitude_preset.get()
        if preset == "fl350":
            self.altitude_value.set(10.7)  # 35,000 ft
            self.altitude_slider.config(state="disabled")
        elif preset == "fl400":
            self.altitude_value.set(12.2)  # 40,000 ft
            self.altitude_slider.config(state="disabled")
        else:  # custom
            self.altitude_slider.config(state="normal")
        self._on_altitude_change(None)

    def _on_altitude_change(self, _):
        self.altitude_label.config(text=self._format_altitude_label(self.altitude_value.get()))

    def show_map(self):
        try:
            if self.result is None:
                messagebox.showerror("Error", "No simulation results available.")
                return

            selected_altitude = self.altitude_value.get()
            for widget in self.map_canvas_frame.winfo_children():
                widget.destroy()

            # Find the closest available altitude in the data
            available_altitudes = self.result['altitude (km)'].unique()
            selected_altitude = available_altitudes[np.abs(available_altitudes - selected_altitude).argmin()]
            
            df_to_plot = self.result.query(f"`altitude (km)` == {selected_altitude}")
            
            if self.view_mode.get() == "3D":
                # Create 3D visualization
                fig = self.create_3d_earth_plot(df_to_plot)
                fig.show()
                
                # Add a message in the frame
                msg = ttk.Label(self.map_canvas_frame, 
                              text="3D visualization opened in a new window.\nYou can interact with it there.",
                              justify="center")
                msg.pack(expand=True)
            else:
                # 2D visualization
                plot_contours_flag = True
                if df_to_plot["latitude"].nunique() < 2 or df_to_plot["longitude"].nunique() < 2:
                    plot_contours_flag = False

                fig, ax = plt.subplots(figsize=(6, 4))
                #
                plt.sca(ax)
                plot_dose_map(df_to_plot, plot_contours=plot_contours_flag)
                canvas = FigureCanvasTkAgg(fig, master=self.map_canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_3d_earth_plot(self, df_to_plot):
        """Create an interactive 3D Earth visualization with dose data."""
        fig = go.Figure()

        # Create the base Earth sphere
        phi = np.linspace(0, 2*np.pi, 100)
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        phi, theta = np.meshgrid(phi, theta)

        x = 1.02 * np.cos(theta) * np.cos(phi)  # 1.02 to place slightly above surface
        y = 1.02 * np.cos(theta) * np.sin(phi)
        z = 1.02 * np.sin(theta)

        # Add the Earth's surface
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale='Blues',
            showscale=False,
            opacity=0.3
        ))

        # Add country boundaries
        try:
            # Download and load world map data
            world_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
            world = gpd.read_file(world_url)
            
            # Function to convert lat/lon to 3D coordinates
            def lat_lon_to_3d(lat, lon, r=1.021):  # Slightly above Earth surface
                lat, lon = np.radians(lat), np.radians(lon)
                x = r * np.cos(lat) * np.cos(lon)
                y = r * np.cos(lat) * np.sin(lon)
                z = r * np.sin(lat)
                return x, y, z

            # Add each country's boundary
            for idx, country in world.iterrows():
                if country.geometry.geom_type == 'Polygon':
                    coords = np.array(country.geometry.exterior.coords)
                    lats, lons = coords[:, 1], coords[:, 0]
                    x, y, z = lat_lon_to_3d(lats, lons)
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='lines',
                        line=dict(color='black', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                elif country.geometry.geom_type == 'MultiPolygon':
                    for polygon in country.geometry.geoms:
                        coords = np.array(polygon.exterior.coords)
                        lats, lons = coords[:, 1], coords[:, 0]
                        x, y, z = lat_lon_to_3d(lats, lons)
                        fig.add_trace(go.Scatter3d(
                            x=x, y=y, z=z,
                            mode='lines',
                            line=dict(color='black', width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
        except Exception as e:
            print(f"Warning: Could not add country boundaries: {str(e)}")

        # Convert lat/lon to 3D coordinates and add dose data points
        r = 1.03  # Slightly above the Earth's surface
        lat = np.radians(df_to_plot['latitude'])
        lon = np.radians(df_to_plot['longitude'])
        
        x_data = r * np.cos(lat) * np.cos(lon)
        y_data = r * np.cos(lat) * np.sin(lon)
        z_data = r * np.sin(lat)

        # Add the dose rate data points
        fig.add_trace(go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode='markers',
            marker=dict(
                size=5,
                color=df_to_plot['edose'],
                colorscale='Viridis',
                colorbar=dict(title='Dose Rate'),
                showscale=True
            ),
            hovertext=[f'Lat: {lat:.2f}°<br>Lon: {lon:.2f}°<br>Dose: {dose:.2f}'
                      for lat, lon, dose in zip(df_to_plot['latitude'], 
                                              df_to_plot['longitude'], 
                                              df_to_plot['edose'])],
            hoverinfo='text'
        ))

        # Update the layout for better visualization
        fig.update_layout(
            scene=dict(
                aspectmode='cube',
                xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            width=800,
            height=800
        )

        return fig

    def _show_about(self):
        messagebox.showinfo("About AniMAIRE", "AniMAIRE - Cosmic Ray Dose Modeling\nVersion 1.0")

    def _create_results_widgets(self):
        """Create the results table and visualization widgets"""
        # Create notebook for different result views
        self.results_notebook = ttk.Notebook(self.results_frame)
        self.results_notebook.pack(fill="both", expand=True)
        
        # Single simulation results tab
        self.single_results_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.single_results_frame, text="Single Simulation")
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.analysis_frame, text="Analysis")
        
        # Comparison tab
        self.comparison_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.comparison_frame, text="Compare GLEs")
        
        # Results table with embedded scrollable area (in single results tab)
        self.output_table_frame = ttk.Frame(self.single_results_frame)
        self.output_table_frame.pack(fill="both", expand=True, pady=10)
        
        # Create table with scrollbar
        self.output_table = ttk.Treeview(self.output_table_frame)
        self.output_table.pack(side="left", fill="both", expand=True)
        self.output_scroll = ttk.Scrollbar(self.output_table_frame, orient="vertical", 
                                         command=self.output_table.yview)
        self.output_scroll.pack(side="right", fill="y")
        self.output_table.configure(yscrollcommand=self.output_scroll.set)

        # Create analysis tools
        self._create_analysis_widgets()

        # Visualization controls for single simulation
        self.viz_control_frame = ttk.Frame(self.single_results_frame)
        self.viz_control_frame.pack(fill="x", pady=5)
        
        # View mode selection
        self.view_mode = tk.StringVar(value="2D")
        ttk.Radiobutton(self.viz_control_frame, text="2D Map", variable=self.view_mode, 
                       value="2D", command=self.show_map).pack(side="left", padx=5)
        ttk.Radiobutton(self.viz_control_frame, text="3D Globe", variable=self.view_mode, 
                       value="3D", command=self.show_map).pack(side="left", padx=5)

        # Map display section
        self.map_canvas_frame = ttk.Frame(self.single_results_frame)
        self.map_canvas_frame.pack(fill="both", expand=True, pady=10)

        # Create comparison controls
        comparison_control_frame = ttk.Frame(self.comparison_frame)
        comparison_control_frame.pack(fill="x", pady=5)
        
        # GLE selection for comparison
        ttk.Label(comparison_control_frame, text="Compare with:").pack(side="left", padx=5)
        self.compare_gle_var = ttk.StringVar()
        self.compare_gle_combo = ttk.Combobox(comparison_control_frame, 
                                             textvariable=self.compare_gle_var,
                                             values=list(self.gle_presets.keys()),
                                             width=30, state="readonly")
        self.compare_gle_combo.pack(side="left", padx=5)
        
        # Add comparison button
        ttk.Button(comparison_control_frame, text="Compare", 
                  command=self._compare_with_gle).pack(side="left", padx=5)
        
        # Save comparison button
        ttk.Button(comparison_control_frame, text="Save Comparison", 
                  command=self._save_comparison).pack(side="right", padx=5)
        
        # Comparison display area
        self.comparison_display_frame = ttk.Frame(self.comparison_frame)
        self.comparison_display_frame.pack(fill="both", expand=True, pady=5)
        
        # Store comparison results
        self.comparison_result = None

    def _compare_with_gle(self):
        """Compare current simulation with selected GLE"""
        if self.result is None:
            messagebox.showerror("Error", "Run a simulation first before comparing")
            return
        
        selected_gle = self.compare_gle_var.get()
        if not selected_gle:
            messagebox.showerror("Error", "Select a GLE to compare with")
            return
        
        try:
            # Clear previous comparison display
            for widget in self.comparison_display_frame.winfo_children():
                widget.destroy()
            
            # Create figure with multiple subplots for comparison
            fig = plt.figure(figsize=(15, 10))
            gs = plt.GridSpec(2, 2)
            
            # Spectrum comparison (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            R = np.logspace(-1, 2, 1000)
            
            # Get current simulation spectrum
            current_spectrum = self._calculate_spectrum(self.model_choice.get(), R)
            ax1.loglog(R, current_spectrum, 'b-', label='Current')
            
            # Get GLE spectrum
            gle_preset = self.gle_presets[selected_gle]
            gle_spectrum = self._calculate_spectrum(gle_preset["model"], R, gle_preset)
            ax1.loglog(R, gle_spectrum, 'r--', label=selected_gle)
            
            ax1.set_xlabel("Rigidity (GV)")
            ax1.set_ylabel("Flux (cm⁻² s⁻¹ sr⁻¹ GV⁻¹)")
            ax1.set_title("Spectrum Comparison")
            ax1.grid(True)
            ax1.legend()
            
            # Pitch angle distribution comparison (top right)
            ax2 = fig.add_subplot(gs[0, 1])
            alpha = np.linspace(0, np.pi, 1000)
            
            # Get current PAD
            current_pad = self._calculate_pad(self.model_choice.get(), alpha)
            ax2.plot(np.degrees(alpha), current_pad, 'b-', label='Current')
            
            # Get GLE PAD
            gle_pad = self._calculate_pad(gle_preset["model"], alpha, gle_preset)
            ax2.plot(np.degrees(alpha), gle_pad, 'r--', label=selected_gle)
            
            ax2.set_xlabel("Pitch Angle (degrees)")
            ax2.set_ylabel("Relative Intensity")
            ax2.set_title("Pitch Angle Distribution Comparison")
            ax2.grid(True)
            ax2.legend()
            
            # Dose rate comparison (bottom)
            ax3 = fig.add_subplot(gs[1, :])
            
            # Get dose rates at same locations
            current_doses = self.result['edose']
            
            # Run GLE simulation with same grid points
            gle_result = self._run_gle_simulation(selected_gle)
            gle_doses = gle_result['edose']
            
            # Create comparison plot
            locations = range(len(current_doses))
            ax3.plot(locations, current_doses, 'b.-', label='Current')
            ax3.plot(locations, gle_doses, 'r.--', label=selected_gle)
            
            ax3.set_xlabel("Location Index")
            ax3.set_ylabel("Dose Rate")
            ax3.set_title("Dose Rate Comparison")
            ax3.grid(True)
            ax3.legend()
            
            plt.tight_layout()
            
            # Embed plot in comparison frame
            canvas = FigureCanvasTkAgg(fig, master=self.comparison_display_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Store comparison results
            self.comparison_result = {
                'current': self.result,
                'gle': gle_result,
                'gle_name': selected_gle
            }
            
            # Add statistics table
            stats_frame = ttk.Frame(self.comparison_display_frame)
            stats_frame.pack(fill="x", pady=5)
            
            # Calculate statistics
            current_mean = np.mean(current_doses)
            gle_mean = np.mean(gle_doses)
            current_max = np.max(current_doses)
            gle_max = np.max(gle_doses)
            ratio_mean = current_mean / gle_mean
            ratio_max = current_max / gle_max
            
            # Create statistics table
            stats_text = f"""
            Statistics:
            Current Simulation - Mean: {current_mean:.2f}, Max: {current_max:.2f}
            {selected_gle} - Mean: {gle_mean:.2f}, Max: {gle_max:.2f}
            Ratio (Current/GLE) - Mean: {ratio_mean:.2f}, Max: {ratio_max:.2f}
            """
            ttk.Label(stats_frame, text=stats_text).pack()
            
        except Exception as e:
            messagebox.showerror("Comparison Error", str(e))

    def _calculate_spectrum(self, model_type, R, preset=None):
        """Calculate spectrum for given model type and parameters"""
        try:
            if preset is None:
                # Use current parameters
                if model_type == "POWER_LAW":
                    j0 = float(self.j0_entry.get())
                    gamma = float(self.gamma_entry.get())
                    delta_gamma = float(self.delta_gamma_entry.get())
                    return j0 * (R/1.0)**(-gamma) * np.exp(-delta_gamma * np.log(R/1.0)**2)
                
                elif model_type == "DOUBLE_POWER_LAW":
                    j0 = float(self.j0_entry.get())
                    gamma = float(self.gamma_entry.get())
                    delta_gamma = float(self.delta_gamma_entry.get())
                    b = float(self.b_entry.get())
                    spectrum1 = j0 * (R/1.0)**(-gamma) * np.exp(-delta_gamma * np.log(R/1.0)**2)
                    spectrum2 = b * j0 * (R/1.0)**(-gamma-1) * np.exp(-delta_gamma * np.log(R/1.0)**2)
                    return spectrum1 + spectrum2
                
                elif model_type == "POWER_LAW_BEECK":
                    j0 = float(self.j0_entry.get())
                    gamma = float(self.gamma_entry.get())
                    delta_gamma = float(self.delta_gamma_entry.get())
                    return j0 * (R/1.0)**(-gamma) * np.exp(-delta_gamma * np.log(R/1.0)**2)
                
                else:  # DLR or Custom
                    return np.ones_like(R)  # Default to flat spectrum
            else:
                # Use preset parameters
                if model_type == "POWER_LAW":
                    return preset["j0"] * (R/1.0)**(-preset["gamma"]) * \
                           np.exp(-preset["delta_gamma"] * np.log(R/1.0)**2)
                
                elif model_type == "DOUBLE_POWER_LAW":
                    spectrum1 = preset["j0"] * (R/1.0)**(-preset["gamma"]) * \
                              np.exp(-preset["delta_gamma"] * np.log(R/1.0)**2)
                    spectrum2 = preset["b"] * preset["j0"] * (R/1.0)**(-preset["gamma"]-1) * \
                              np.exp(-preset["delta_gamma"] * np.log(R/1.0)**2)
                    return spectrum1 + spectrum2
                
                elif model_type == "POWER_LAW_BEECK":
                    return preset["j0"] * (R/1.0)**(-preset["gamma"]) * \
                           np.exp(-preset["delta_gamma"] * np.log(R/1.0)**2)
                
                else:  # DLR or Custom
                    return np.ones_like(R)  # Default to flat spectrum
                    
        except Exception as e:
            print(f"Error calculating spectrum: {str(e)}")
            return np.ones_like(R)

    def _calculate_pad(self, model_type, alpha, preset=None):
        """Calculate pitch angle distribution for given model type and parameters"""
        try:
            if preset is None:
                # Use current parameters
                if model_type == "POWER_LAW":
                    sigma = float(self.sigma_entry.get())
                    return np.exp(-alpha**2 / (2 * sigma**2))
                
                elif model_type == "DOUBLE_POWER_LAW":
                    sigma1 = float(self.sigma_entry.get())
                    sigma2 = float(self.sigma2_entry.get())
                    b = float(self.b_entry.get())
                    alpha_prime = float(self.alpha_prime_entry.get())
                    pad1 = np.exp(-alpha**2 / (2 * sigma1**2))
                    pad2 = b * np.exp(-(alpha - alpha_prime)**2 / (2 * sigma2**2))
                    return (pad1 + pad2) / np.max(pad1 + pad2)
                
                elif model_type == "POWER_LAW_BEECK":
                    a = float(self.beeck_a_entry.get())
                    b = float(self.beeck_b_entry.get())
                    return np.exp(-a * (1 - np.cos(alpha))**b)
                
                else:  # DLR or Custom
                    return np.ones_like(alpha)  # Default to isotropic
            else:
                # Use preset parameters
                if model_type == "POWER_LAW":
                    return np.exp(-alpha**2 / (2 * preset["sigma"]**2))
                
                elif model_type == "DOUBLE_POWER_LAW":
                    pad1 = np.exp(-alpha**2 / (2 * preset["sigma1"]**2))
                    pad2 = preset["b"] * np.exp(-(alpha - preset["alpha_prime"])**2 / \
                                              (2 * preset["sigma2"]**2))
                    return (pad1 + pad2) / np.max(pad1 + pad2)
                
                elif model_type == "POWER_LAW_BEECK":
                    return np.exp(-preset["beeck_a"] * (1 - np.cos(alpha))**preset["beeck_b"])
                
                else:  # DLR or Custom
                    return np.ones_like(alpha)  # Default to isotropic
                    
        except Exception as e:
            print(f"Error calculating PAD: {str(e)}")
            return np.ones_like(alpha)

    def _run_gle_simulation(self, gle_name):
        """Run simulation with GLE preset parameters"""
        preset = self.gle_presets[gle_name]
        
        # Get common parameters
        kp_index = int(self.kp_index_entry.get()) if self.kp_index_entry.get() else None
        date_and_time = dt.datetime.strptime(preset["datetime"], "%Y-%m-%d %H:%M:%S")
        
        if preset["model"] == "POWER_LAW":
            return run_from_power_law_gaussian_distribution(
                J0=preset["j0"],
                gamma=preset["gamma"],
                deltaGamma=preset["delta_gamma"],
                sigma=preset["sigma"],
                reference_pitch_angle_latitude=preset["ref_lat"],
                reference_pitch_angle_longitude=preset["ref_lon"],
                Kp_index=kp_index,
                date_and_time=date_and_time,
                array_of_lats_and_longs=self.grid_points
            )
        
        elif preset["model"] == "DOUBLE_POWER_LAW":
            return run_from_double_power_law_gaussian_distribution(
                J0=preset["j0"],
                gamma=preset["gamma"],
                deltaGamma=preset["delta_gamma"],
                sigma_1=preset["sigma1"],
                sigma_2=preset["sigma2"],
                B=preset["b"],
                alpha_prime=preset["alpha_prime"],
                reference_pitch_angle_latitude=preset["ref_lat"],
                reference_pitch_angle_longitude=preset["ref_lon"],
                Kp_index=kp_index,
                date_and_time=date_and_time,
                array_of_lats_and_longs=self.grid_points
            )
        
        elif preset["model"] == "POWER_LAW_BEECK":
            return run_from_power_law_Beeck_gaussian_distribution(
                J0=preset["j0"],
                gamma=preset["gamma"],
                deltaGamma=preset["delta_gamma"],
                A=preset["beeck_a"],
                B=preset["beeck_b"],
                reference_pitch_angle_latitude=preset["ref_lat"],
                reference_pitch_angle_longitude=preset["ref_lon"],
                Kp_index=kp_index,
                date_and_time=date_and_time,
                array_of_lats_and_longs=self.grid_points
            )

    def _save_comparison(self):
        """Save comparison results and plots"""
        if self.comparison_result is None:
            messagebox.showerror("Error", "No comparison results to save")
            return
        
        try:
            # Create comparison report
            current_doses = self.comparison_result['current']['edose']
            gle_doses = self.comparison_result['gle']['edose']
            gle_name = self.comparison_result['gle_name']
            gle_preset = self.gle_presets[gle_name]
            
            # Calculate statistics
            stats = {
                'current_mean': np.mean(current_doses),
                'current_max': np.max(current_doses),
                'gle_mean': np.mean(gle_doses),
                'gle_max': np.max(gle_doses),
                'ratio_mean': np.mean(current_doses) / np.mean(gle_doses),
                'ratio_max': np.max(current_doses) / np.max(gle_doses)
            }
            
            # Create DataFrame with comparison data
            comparison_df = pd.DataFrame({
                'Location': range(len(current_doses)),
                'Current_Dose': current_doses,
                f'{gle_name}_Dose': gle_doses,
                'Ratio': current_doses / gle_doses
            })
            
            # Save to CSV
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_df.to_csv(f'comparison_{timestamp}.csv', index=False)
            
            # Create and save detailed report
            with open(f'comparison_report_{timestamp}.txt', 'w') as f:
                f.write(f"Comparison Report: Current Simulation vs {gle_name}\n")
                f.write(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Statistics:\n")
                f.write(f"Current Simulation - Mean: {stats['current_mean']:.2f}, Max: {stats['current_max']:.2f}\n")
                f.write(f"{gle_name} - Mean: {stats['gle_mean']:.2f}, Max: {stats['gle_max']:.2f}\n")
                f.write(f"Ratio (Current/GLE) - Mean: {stats['ratio_mean']:.2f}, Max: {stats['ratio_max']:.2f}\n\n")
                
                f.write("Model Parameters:\n")
                f.write("Current Simulation:\n")
                f.write(f"Model: {self.model_choice.get()}\n")
                f.write(f"Date/Time: {self.datetime_entry.get()}\n")
                # Add more parameters based on model type
                
                f.write(f"\n{gle_name} Parameters:\n")
                for key, value in self.gle_presets[gle_name].items():
                    if key != "references":  # Skip references in parameter list
                        f.write(f"{key}: {value}\n")
                
                # Add references section
                f.write("\nReferences:\n")
                for ref in gle_preset.get("references", []):
                    f.write(f"• {ref}\n")
            
            messagebox.showinfo("Success", 
                              f"Comparison saved to:\ncomparison_{timestamp}.csv\n" + 
                              f"comparison_report_{timestamp}.txt")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save comparison: {str(e)}")

    def _load_gle_preset(self):
        """Load parameters from selected GLE preset"""
        selected_gle = self.gle_var.get()
        if not selected_gle:
            messagebox.showwarning("Warning", "Please select a GLE preset first")
            return
        
        try:
            preset = self.gle_presets[selected_gle]
            
            # Set model type
            self.model_choice.set(preset["model"])
            self._update_model_params()
            
            # Set datetime
            self.datetime_entry.delete(0, tk.END)
            self.datetime_entry.insert(0, preset["datetime"])
            
            # Set reference direction
            self.ref_lat_entry.delete(0, tk.END)
            self.ref_lat_entry.insert(0, str(preset["ref_lat"]))
            self.ref_lon_entry.delete(0, tk.END)
            self.ref_lon_entry.insert(0, str(preset["ref_lon"]))
            
            # Set model-specific parameters
            if preset["model"] == "POWER_LAW":
                self.j0_entry.delete(0, tk.END)
                self.j0_entry.insert(0, str(preset["j0"]))
                self.gamma_entry.delete(0, tk.END)
                self.gamma_entry.insert(0, str(preset["gamma"]))
                self.delta_gamma_entry.delete(0, tk.END)
                self.delta_gamma_entry.insert(0, str(preset["delta_gamma"]))
                self.sigma_entry.delete(0, tk.END)
                self.sigma_entry.insert(0, str(preset["sigma"]))
            
            elif preset["model"] == "DOUBLE_POWER_LAW":
                self.j0_entry.delete(0, tk.END)
                self.j0_entry.insert(0, str(preset["j0"]))
                self.gamma_entry.delete(0, tk.END)
                self.gamma_entry.insert(0, str(preset["gamma"]))
                self.delta_gamma_entry.delete(0, tk.END)
                self.delta_gamma_entry.insert(0, str(preset["delta_gamma"]))
                self.sigma_entry.delete(0, tk.END)
                self.sigma_entry.insert(0, str(preset["sigma1"]))
                self.sigma2_entry.delete(0, tk.END)
                self.sigma2_entry.insert(0, str(preset["sigma2"]))
                self.b_entry.delete(0, tk.END)
                self.b_entry.insert(0, str(preset["b"]))
                self.alpha_prime_entry.delete(0, tk.END)
                self.alpha_prime_entry.insert(0, str(preset["alpha_prime"]))
            
            elif preset["model"] == "POWER_LAW_BEECK":
                self.j0_entry.delete(0, tk.END)
                self.j0_entry.insert(0, str(preset["j0"]))
                self.gamma_entry.delete(0, tk.END)
                self.gamma_entry.insert(0, str(preset["gamma"]))
                self.delta_gamma_entry.delete(0, tk.END)
                self.delta_gamma_entry.insert(0, str(preset["delta_gamma"]))
                self.beeck_a_entry.delete(0, tk.END)
                self.beeck_a_entry.insert(0, str(preset["beeck_a"]))
                self.beeck_b_entry.delete(0, tk.END)
                self.beeck_b_entry.insert(0, str(preset["beeck_b"]))
            
            self._update_status(f"Loaded parameters for {selected_gle}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load GLE preset: {str(e)}")

    def _update_gle_description(self, event=None):
        """Update the description when a GLE is selected"""
        selected_gle = self.gle_var.get()
        if selected_gle:
            preset = self.gle_presets[selected_gle]
            description = preset.get("description", "No description available")
            references = preset.get("references", [])
            
            # Format the text with description and references
            full_text = f"{description}\n\nReferences:\n"
            for ref in references:
                full_text += f"• {ref}\n"
            
            self.gle_description.config(text=full_text)

    def _draw_logo(self, size):
        """Draw an enhanced version of the AniMAIRE logo on the canvas"""
        # Clear canvas
        self.logo_canvas.delete("all")
        
        # Calculate center and dimensions
        center = size // 2
        outer_radius = size * 0.45
        inner_radius = outer_radius * 0.7
        
        # Enhanced color palette
        primary_blue = "#007ACC"      # Vibrant blue for "Ani"
        deep_purple = "#6B46C1"       # Deep purple for gradient
        accent_teal = "#00B5D8"       # Teal accent
        glow_color = "#4299E1"        # Soft blue for glow effects
        
        # Helper function to create color with opacity
        def adjust_color_brightness(color, factor):
            # Convert hex to RGB
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            
            # Adjust brightness
            r = min(255, int(r * factor))
            g = min(255, int(g * factor))
            b = min(255, int(b * factor))
            
            # Convert back to hex
            return f"#{r:02x}{g:02x}{b:02x}"
        
        # Create radial gradient effect for background glow
        steps = int((outer_radius - inner_radius) / 2)
        for i in range(steps):
            r = outer_radius - (i * 2)
            factor = (r - inner_radius) / (outer_radius - inner_radius)
            glow = adjust_color_brightness(glow_color, factor * 0.7)
            self.logo_canvas.create_oval(
                center - r, center - r,
                center + r, center + r,
                fill="", outline=glow, width=2
            )
        
        # Draw main sphere (Earth-like)
        self.logo_canvas.create_oval(
            center - inner_radius, center - inner_radius,
            center + inner_radius, center + inner_radius,
            fill="#1A365D", outline=accent_teal, width=2
        )
        
        # Draw cosmic ray paths with varying lengths and angles
        for base_angle in range(0, 360, 30):
            for offset in [-5, 0, 5]:
                angle = base_angle + offset
                ray_length = outer_radius * (0.9 + 0.2 * (offset == 0))
                start_x = center + inner_radius * math.cos(math.radians(angle))
                start_y = center + inner_radius * math.sin(math.radians(angle))
                end_x = center + ray_length * math.cos(math.radians(angle))
                end_y = center + ray_length * math.sin(math.radians(angle))
                
                # Create gradient effect for rays
                for i in range(10):
                    t = i / 10
                    x1 = start_x + (end_x - start_x) * t
                    y1 = start_y + (end_y - start_y) * t
                    x2 = start_x + (end_x - start_x) * (t + 0.1)
                    y2 = start_y + (end_y - start_y) * (t + 0.1)
                    ray_color = adjust_color_brightness(accent_teal, 1 - t)
                    self.logo_canvas.create_line(x1, y1, x2, y2, fill=ray_color, width=2)
        
        # Draw atmospheric layers
        for i in range(3):
            radius = inner_radius + (outer_radius - inner_radius) * (i / 4)
            factor = 1 - (i / 3)
            layer_color = adjust_color_brightness(deep_purple, factor * 0.5)
            self.logo_canvas.create_oval(
                center - radius, center - radius,
                center + radius, center + radius,
                fill="", outline=layer_color, width=1
            )
        
        # Add text with enhanced styling
        font_size = int(size * 0.15)
        ani_text = self.logo_canvas.create_text(
            center, center - font_size/4,
            text="Ani", font=("Helvetica", font_size, "bold"),
            fill=primary_blue
        )
        maire_text = self.logo_canvas.create_text(
            center + self.logo_canvas.bbox(ani_text)[2] - self.logo_canvas.bbox(ani_text)[0],
            center - font_size/4,
            text="MAIRE", font=("Helvetica", font_size, "bold"),
            fill=deep_purple
        )

    def _draw_dynamic_flux_logo(self, size):
        """Draw the Dynamic Flux variant of the AniMAIRE logo"""
        # Clear canvas
        self.logo_canvas.delete("all")
        
        # Calculate center and dimensions
        center = size // 2
        radius = size * 0.45
        
        # Colors
        primary = "#3498db"    # Bright blue
        secondary = "#e74c3c"  # Red
        tertiary = "#2ecc71"   # Emerald
        
        # Draw spiral particle paths
        for i in range(0, 360, 30):
            points = []
            for t in range(0, 720, 5):
                angle = math.radians(t + i)
                r = radius * (1 - t/1440)  # Decreasing radius
                x = center + r * math.cos(angle)
                y = center + r * math.sin(angle)
                points.extend([x, y])
            
            # Create spiral with gradient effect
            self.logo_canvas.create_line(points, smooth=True,
                                       fill=secondary, width=2)
        
        # Draw central sphere
        inner_radius = radius * 0.3
        self.logo_canvas.create_oval(center - inner_radius, center - inner_radius,
                                   center + inner_radius, center + inner_radius,
                                   fill=primary, outline=tertiary, width=3)
        
        # Add dynamic particles
        for angle in range(0, 360, 40):
            rad_angle = math.radians(angle)
            particle_x = center + radius * 0.6 * math.cos(rad_angle)
            particle_y = center + radius * 0.6 * math.sin(rad_angle)
            
            # Draw particle with glow effect
            self.logo_canvas.create_oval(particle_x - 5, particle_y - 5,
                                       particle_x + 5, particle_y + 5,
                                       fill=secondary, outline=tertiary)
        
        # Add stylized "A" in the center
        font_size = int(size * 0.25)
        self.logo_canvas.create_text(center, center, text="A",
                                   font=("Helvetica", font_size, "bold"),
                                   fill="#ffffff")

    def _create_analysis_widgets(self):
        """Create comprehensive analysis tools for dose rate data"""
        # Create notebook for different analysis types
        analysis_notebook = ttk.Notebook(self.analysis_frame)
        analysis_notebook.pack(fill="both", expand=True)
        
        # Statistical Analysis Tab
        stats_frame = ttk.Frame(analysis_notebook)
        analysis_notebook.add(stats_frame, text="Statistical Analysis")
        
        # Time Series Analysis Tab
        timeseries_frame = ttk.Frame(analysis_notebook)
        analysis_notebook.add(timeseries_frame, text="Time Series Analysis")
        
        # Geomagnetic Analysis Tab
        geomag_frame = ttk.Frame(analysis_notebook)
        analysis_notebook.add(geomag_frame, text="Geomagnetic Analysis")
        
        # Spectral Analysis Tab
        spectral_frame = ttk.Frame(analysis_notebook)
        analysis_notebook.add(spectral_frame, text="Spectral Analysis")
        
        # Create Statistical Analysis Tools
        stats_control_frame = ttk.Labelframe(stats_frame, text="Statistical Tools", padding=10)
        stats_control_frame.pack(fill="x", padx=5, pady=5)
        
        # Basic Statistics
        ttk.Button(stats_control_frame, text="Calculate Basic Statistics",
                  command=self._calculate_basic_stats).pack(side="left", padx=5)
        
        # Dose Rate Distribution
        ttk.Button(stats_control_frame, text="Plot Dose Rate Distribution",
                  command=self._plot_dose_distribution).pack(side="left", padx=5)
        
        # Correlation Analysis
        ttk.Button(stats_control_frame, text="Correlation Analysis",
                  command=self._analyze_correlations).pack(side="left", padx=5)
        
        # Stats Output
        self.stats_output = ttk.Text(stats_frame, height=10)
        self.stats_output.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Time Series Analysis Tools
        ts_control_frame = ttk.Labelframe(timeseries_frame, text="Time Series Tools", padding=10)
        ts_control_frame.pack(fill="x", padx=5, pady=5)
        
        # Moving Average
        ttk.Label(ts_control_frame, text="Moving Average Window (hours):").pack(side="left", padx=5)
        self.ma_window = ttk.Entry(ts_control_frame, width=10)
        self.ma_window.pack(side="left", padx=5)
        self.ma_window.insert(0, "24")
        
        ttk.Button(ts_control_frame, text="Calculate Moving Average",
                  command=self._calculate_moving_average).pack(side="left", padx=5)
        
        # Trend Analysis
        ttk.Button(ts_control_frame, text="Analyze Trends",
                  command=self._analyze_trends).pack(side="left", padx=5)
        
        # Time Series Plot Frame
        self.ts_plot_frame = ttk.Frame(timeseries_frame)
        self.ts_plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Geomagnetic Analysis Tools
        geo_control_frame = ttk.Labelframe(geomag_frame, text="Geomagnetic Tools", padding=10)
        geo_control_frame.pack(fill="x", padx=5, pady=5)
        
        # Cutoff Rigidity Analysis
        ttk.Button(geo_control_frame, text="Calculate Cutoff Rigidities",
                  command=self._analyze_cutoff_rigidity).pack(side="left", padx=5)
        
        # Kp Index Correlation
        ttk.Button(geo_control_frame, text="Kp Index Correlation",
                  command=self._analyze_kp_correlation).pack(side="left", padx=5)
        
        # L-Shell Analysis
        ttk.Button(geo_control_frame, text="L-Shell Analysis",
                  command=self._analyze_l_shell).pack(side="left", padx=5)
        
        # Geomagnetic Plot Frame
        self.geo_plot_frame = ttk.Frame(geomag_frame)
        self.geo_plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Spectral Analysis Tools
        spec_control_frame = ttk.Labelframe(spectral_frame, text="Spectral Analysis Tools", padding=10)
        spec_control_frame.pack(fill="x", padx=5, pady=5)
        
        # FFT Analysis
        ttk.Button(spec_control_frame, text="FFT Analysis",
                  command=self._perform_fft_analysis).pack(side="left", padx=5)
        
        # Power Spectral Density
        ttk.Button(spec_control_frame, text="Power Spectral Density",
                  command=self._calculate_psd).pack(side="left", padx=5)
        
        # Wavelet Analysis
        ttk.Button(spec_control_frame, text="Wavelet Analysis",
                  command=self._perform_wavelet_analysis).pack(side="left", padx=5)
        
        # Spectral Plot Frame
        self.spec_plot_frame = ttk.Frame(spectral_frame)
        self.spec_plot_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def _calculate_basic_stats(self):
        """Calculate and display basic statistics of dose rates"""
        if self.result is None:
            messagebox.showerror("Error", "No simulation results available")
            return
        
        try:
            stats = {
                'Mean Dose Rate': f"{np.mean(self.result['edose']):.2f} μSv/h",
                'Median Dose Rate': f"{np.median(self.result['edose']):.2f} μSv/h",
                'Std Dev': f"{np.std(self.result['edose']):.2f} μSv/h",
                'Max Dose Rate': f"{np.max(self.result['edose']):.2f} μSv/h",
                'Min Dose Rate': f"{np.min(self.result['edose']):.2f} μSv/h",
                'Total Dose': f"{np.sum(self.result['edose']):.2f} μSv",
                'Skewness': f"{pd.Series(self.result['edose']).skew():.2f}",
                'Kurtosis': f"{pd.Series(self.result['edose']).kurtosis():.2f}"
            }
            
            # Clear and update stats output
            self.stats_output.delete('1.0', tk.END)
            self.stats_output.insert('1.0', "Basic Statistics:\n\n")
            for key, value in stats.items():
                self.stats_output.insert(tk.END, f"{key}: {value}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate statistics: {str(e)}")

    def _plot_dose_distribution(self):
        """Plot the distribution of dose rates"""
        if self.result is None:
            messagebox.showerror("Error", "No simulation results available")
            return
        
        try:
            # Create figure with multiple plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            ax1.hist(self.result['edose'], bins=30, density=True, alpha=0.7)
            ax1.set_xlabel('Dose Rate (μSv/h)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Dose Rate Distribution')
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(self.result['edose'], dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot')
            
            plt.tight_layout()
            
            # Show in a new window
            new_window = ttk.Toplevel(self.root)
            new_window.title("Dose Rate Distribution Analysis")
            canvas = FigureCanvasTkAgg(fig, master=new_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot distribution: {str(e)}")

    def _analyze_correlations(self):
        """Analyze correlations between dose rates and other parameters"""
        if self.result is None:
            messagebox.showerror("Error", "No simulation results available")
            return
        
        try:
            # Calculate correlations
            corr_matrix = self.result.corr()
            
            # Create correlation plot
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(corr_matrix, cmap='coolwarm')
            
            # Add labels
            ax.set_xticks(np.arange(len(corr_matrix.columns)))
            ax.set_yticks(np.arange(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_matrix.columns)
            
            # Add colorbar
            plt.colorbar(im)
            
            plt.title('Parameter Correlations')
            plt.tight_layout()
            
            # Show in a new window
            new_window = ttk.Toplevel(self.root)
            new_window.title("Correlation Analysis")
            canvas = FigureCanvasTkAgg(fig, master=new_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze correlations: {str(e)}")

    def _calculate_moving_average(self):
        """Calculate and plot moving average of dose rates"""
        if self.result is None:
            messagebox.showerror("Error", "No simulation results available")
            return
        
        try:
            window = int(self.ma_window.get())
            ma = pd.Series(self.result['edose']).rolling(window=window).mean()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.result['edose'], label='Raw Data', alpha=0.5)
            ax.plot(ma, label=f'{window}-point Moving Average', linewidth=2)
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Dose Rate (μSv/h)')
            ax.set_title('Dose Rate Moving Average Analysis')
            ax.legend()
            
            plt.tight_layout()
            
            # Clear previous plot and show new one
            for widget in self.ts_plot_frame.winfo_children():
                widget.destroy()
            
            canvas = FigureCanvasTkAgg(fig, master=self.ts_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate moving average: {str(e)}")

    def _analyze_trends(self):
        """Analyze and plot trends in dose rate data"""
        if self.result is None:
            messagebox.showerror("Error", "No simulation results available")
            return
        
        try:
            from scipy import signal
            
            # Detrend data
            trend = signal.detrend(self.result['edose'])
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Original data and trend
            ax1.plot(self.result['edose'], label='Original Data')
            ax1.plot(self.result['edose'] - trend, label='Trend')
            ax1.set_xlabel('Sample Index')
            ax1.set_ylabel('Dose Rate (μSv/h)')
            ax1.set_title('Dose Rate Trend Analysis')
            ax1.legend()
            
            # Detrended data
            ax2.plot(trend, label='Detrended Data')
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Detrended Dose Rate (μSv/h)')
            ax2.legend()
            
            plt.tight_layout()
            
            # Clear previous plot and show new one
            for widget in self.ts_plot_frame.winfo_children():
                widget.destroy()
            
            canvas = FigureCanvasTkAgg(fig, master=self.ts_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze trends: {str(e)}")

    def _analyze_cutoff_rigidity(self):
        """Analyze relationship between dose rates and cutoff rigidity"""
        if self.result is None:
            messagebox.showerror("Error", "No simulation results available")
            return
        
        try:
            # Calculate approximate vertical cutoff rigidity using Störmer's equation
            def calc_cutoff_rigidity(lat):
                # Störmer's equation (simplified)
                return 14.9 * (np.cos(np.radians(lat)))**4
            
            cutoff = [calc_cutoff_rigidity(lat) for lat in self.result['latitude']]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(cutoff, self.result['edose'], 
                               c=self.result['latitude'], cmap='viridis')
            
            ax.set_xlabel('Cutoff Rigidity (GV)')
            ax.set_ylabel('Dose Rate (μSv/h)')
            ax.set_title('Dose Rate vs. Cutoff Rigidity')
            
            plt.colorbar(scatter, label='Latitude (degrees)')
            plt.tight_layout()
            
            # Clear previous plot and show new one
            for widget in self.geo_plot_frame.winfo_children():
                widget.destroy()
            
            canvas = FigureCanvasTkAgg(fig, master=self.geo_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze cutoff rigidity: {str(e)}")

    def _analyze_kp_correlation(self):
        """Analyze correlation between dose rates and Kp index"""
        if self.result is None:
            messagebox.showerror("Error", "No simulation results available")
            return
        
        try:
            kp = float(self.kp_index_entry.get())
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(self.result['latitude'], self.result['edose'],
                               c=self.result['longitude'], cmap='viridis')
            
            ax.set_xlabel('Latitude (degrees)')
            ax.set_ylabel('Dose Rate (μSv/h)')
            ax.set_title(f'Dose Rate Distribution (Kp = {kp})')
            
            plt.colorbar(scatter, label='Longitude (degrees)')
            plt.tight_layout()
            
            # Clear previous plot and show new one
            for widget in self.geo_plot_frame.winfo_children():
                widget.destroy()
            
            canvas = FigureCanvasTkAgg(fig, master=self.geo_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze Kp correlation: {str(e)}")

    def _analyze_l_shell(self):
        """Analyze dose rates in relation to L-shell parameter"""
        if self.result is None:
            messagebox.showerror("Error", "No simulation results available")
            return
        
        try:
            # Calculate approximate L-shell parameter
            def calc_l_shell(lat):
                # Simple approximation for L-shell
                return 1 / (np.cos(np.radians(lat))**2)
            
            l_shell = [calc_l_shell(lat) for lat in self.result['latitude']]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(l_shell, self.result['edose'],
                               c=self.result['latitude'], cmap='viridis')
            
            ax.set_xlabel('L-shell Parameter')
            ax.set_ylabel('Dose Rate (μSv/h)')
            ax.set_title('Dose Rate vs. L-shell Parameter')
            
            plt.colorbar(scatter, label='Latitude (degrees)')
            plt.tight_layout()
            
            # Clear previous plot and show new one
            for widget in self.geo_plot_frame.winfo_children():
                widget.destroy()
            
            canvas = FigureCanvasTkAgg(fig, master=self.geo_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze L-shell: {str(e)}")

    def _perform_fft_analysis(self):
        """Perform FFT analysis on dose rate data"""
        if self.result is None:
            messagebox.showerror("Error", "No simulation results available")
            return
        
        try:
            # Perform FFT
            fft_result = np.fft.fft(self.result['edose'])
            freqs = np.fft.fftfreq(len(self.result['edose']))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(freqs[1:len(freqs)//2], np.abs(fft_result[1:len(freqs)//2]))
            
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Amplitude')
            ax.set_title('FFT Analysis of Dose Rates')
            ax.set_yscale('log')
            
            plt.tight_layout()
            
            # Clear previous plot and show new one
            for widget in self.spec_plot_frame.winfo_children():
                widget.destroy()
            
            canvas = FigureCanvasTkAgg(fig, master=self.spec_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to perform FFT analysis: {str(e)}")

    def _calculate_psd(self):
        """Calculate and plot power spectral density of dose rates"""
        if self.result is None:
            messagebox.showerror("Error", "No simulation results available")
            return
        
        try:
            from scipy import signal
            
            # Calculate PSD
            freqs, psd = signal.welch(self.result['edose'])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.semilogy(freqs, psd)
            
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Power Spectral Density')
            ax.set_title('Power Spectral Density of Dose Rates')
            
            plt.tight_layout()
            
            # Clear previous plot and show new one
            for widget in self.spec_plot_frame.winfo_children():
                widget.destroy()
            
            canvas = FigureCanvasTkAgg(fig, master=self.spec_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate PSD: {str(e)}")

    def _perform_wavelet_analysis(self):
        """Perform wavelet analysis on dose rate data"""
        if self.result is None:
            messagebox.showerror("Error", "No simulation results available")
            return
        
        try:
            import pywt
            
            # Perform wavelet transform
            scales = np.arange(1, 128)
            wavelet = 'morl'
            
            [coefficients, frequencies] = pywt.cwt(self.result['edose'], 
                                                 scales, wavelet)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(np.abs(coefficients), aspect='auto', 
                         extent=[0, len(self.result['edose']), 1, 128])
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Scale')
            ax.set_title('Wavelet Transform of Dose Rates')
            
            plt.colorbar(im, ax=ax, label='Magnitude')
            plt.tight_layout()
            
            # Clear previous plot and show new one
            for widget in self.spec_plot_frame.winfo_children():
                widget.destroy()
            
            canvas = FigureCanvasTkAgg(fig, master=self.spec_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to perform wavelet analysis: {str(e)}")

def run_the_ui():
    root = ttk.Window(themename="flatly")
    app = ImprovedAniMAIREUI(root)
    root.mainloop()

if __name__ == "__main__":
    run_the_ui()