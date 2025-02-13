import ttkbootstrap as ttk
import tkinter as tk
from tkinter import messagebox

class Utils:
    def __init__(self):
        pass

    def show_location_selector(self, root):
        # Placeholder for showing location selector logic
        pass

    def toggle_theme(self, app):
        if app.current_theme == "flatly":
            app.current_theme = "darkly"
        else:
            app.current_theme = "flatly"
        app.style = ttk.Style(app.current_theme)
        self.update_status(app, f"Theme changed to {app.current_theme}")

    def show_help(self):
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

    def create_menu(self, app):
        menu_bar = tk.Menu(app.root)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=False)
        file_menu.add_command(label="New Simulation", accelerator="Ctrl+N")
        file_menu.add_command(label="Open...", accelerator="Ctrl+O")
        file_menu.add_command(label="Save Results...", accelerator="Ctrl+S", 
                            command=app.simulation_analysis.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Export Plot...", command=app.visualization.export_plot)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=app.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(menu_bar, tearoff=False)
        edit_menu.add_command(label="Copy Results", accelerator="Ctrl+C")
        edit_menu.add_command(label="Copy Plot", accelerator="Ctrl+Shift+C")
        menu_bar.add_cascade(label="Edit", menu=edit_menu)
        
        # View menu
        view_menu = tk.Menu(menu_bar, tearoff=False)
        view_menu.add_command(label="Toggle Theme", accelerator="Ctrl+D",
                            command=lambda: self.toggle_theme(app))
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
                            command=self.show_help)
        help_menu.add_command(label="Keyboard Shortcuts...")
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        app.root.config(menu=menu_bar)

    def update_status(self, app, message):
        app.status_bar.config(text=message)

    def add_tooltips(self, app):
        # Placeholder for adding tooltips logic
        pass

    def create_settings_widgets(self, app):
        # Placeholder for creating settings widgets logic
        pass

    def create_results_widgets(self, app):
        # Placeholder for creating results widgets logic
        pass

    def create_favorites_widgets(self, app):
        # Placeholder for creating favorites widgets logic
        pass

    def create_history_widgets(self, app):
        # Placeholder for creating history widgets logic
        pass

    def show_about(self):
        messagebox.showinfo("About AniMAIRE", "AniMAIRE - Cosmic Ray Dose Modeling\nVersion 1.0")