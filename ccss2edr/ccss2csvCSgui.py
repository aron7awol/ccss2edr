#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
import colour.algebra.interpolation as ci
from colour.algebra import Extrapolator  # <-- Import the Extrapolator
from dataclasses import dataclass

# --- Matplotlib Imports ---
# We use the 'Agg' backend for Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# --- Local Imports ---
# Assumes cgats.py is in the same directory
try:
    from cgats import CGATS
except ImportError:
    print("Error: Could not find cgats.py.")
    print("Please make sure cgats.py is in the same directory as this script.")
    exit()

# -------------------------------------------------------------------
#  SPECTRAL DATA CLASS (from previous version)
# -------------------------------------------------------------------

@dataclass
class SpectralData:
    """Holds and processes spectral data, ensuring 380-780nm @ 1nm output."""
    start_nm: float
    end_nm: float
    space_nm: float
    norm: float
    num_bands: int
    num_sets: int
    sets: np.ndarray

    @staticmethod
    def from_ccss(ccss: CGATS):
        orig_num_bands = int(ccss["SPECTRAL_BANDS"])
        orig_start_nm = float(ccss["SPECTRAL_START_NM"])
        orig_end_nm = float(ccss["SPECTRAL_END_NM"])
        norm = float(ccss["SPECTRAL_NORM"])
        orig_space_nm = (orig_end_nm - orig_start_nm) / (orig_num_bands - 1)

        skip_fields = 1  # Always skip SAMPLE_ID

        if orig_start_nm > 380.0:
            raise Exception(
                f"spectral data start must be <= 380.0 nm, is {orig_start_nm}"
            )
        elif orig_start_nm < 380.0:
            skip_samples = int(round((380.0 - orig_start_nm) / orig_space_nm))
            skip_fields += skip_samples
            num_bands = orig_num_bands - skip_samples
            start_nm = 380.0
            print(f"Warning: spectral data starts at {orig_start_nm} nm, "
                  f"skipping {skip_samples} leading bands to start at 380 nm.")
        else:
            num_bands = orig_num_bands
            start_nm = orig_start_nm

        num_sets = int(ccss["NUMBER_OF_SETS"])
        end_nm = start_nm + (num_bands - 1) * orig_space_nm

        setslist = []
        for data in ccss.data:
            raw_vals = data[skip_fields : skip_fields + num_bands]
            setslist.append([float(val) for val in raw_vals])
        sets_orig = np.array(setslist)

        x_orig = np.linspace(start_nm, end_nm, num_bands)
        x_target = np.arange(380, 780 + 1, 1)  # 401 points
        sets_final = np.zeros((num_sets, x_target.shape[0]))

        for i, y_orig in enumerate(sets_orig):
            # 1. Create the base interpolator
            interpolator = ci.SpragueInterpolator(x_orig, y_orig)
            
            # 2. Wrap it with an Extrapolator
            #    This will fill any values outside the x_orig range (e.g., 731-780nm)
            #    with a constant value of 0.0.
            extrapolator = Extrapolator(
                interpolator=interpolator,
                method='Constant',  # Use constant extrapolation
                left=0.0,           # Value for x < min(x_orig)
                right=0.0          # Value for x > max(x_orig)
            )

            # 3. Call the *extrapolator* instead of the interpolator
            y_target = extrapolator(x_target)
            
            # 4. Assign the result. 
            #    np.nan_to_num() is no longer needed as the extrapolator
            #    handles all cases.
            sets_final[i] = y_target

        return SpectralData(380.0, 780.0, 1.0, norm,
                            x_target.shape[0], num_sets, sets_final)

# -------------------------------------------------------------------
#  GUI APPLICATION CLASS
# -------------------------------------------------------------------

class SpectralApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectral Data Viewer")
        self.root.geometry("800x600")

        # --- Data Storage ---
        self.spectral_data = None
        self.normalized_data = None
        self.last_filepath = None

        # --- Main Frame ---
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=1, padx=10, pady=10)

        # --- Top Controls Frame ---
        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(fill=tk.X)

        self.load_button = tk.Button(
            controls_frame, text="Load .ccss File", command=self.load_file
        )
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.export_button = tk.Button(
            controls_frame,
            text="Export Normalized CSV",
            command=self.export_csv,
            state=tk.DISABLED
        )
        self.export_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.filename_label = tk.Label(
            controls_frame, text="No file loaded.", relief=tk.SUNKEN, anchor=tk.W
        )
        self.filename_label.pack(
            side=tk.LEFT, fill=tk.X, expand=1, padx=5, pady=5
        )

        # --- Matplotlib Figure ---
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=1, pady=5
        )

        # --- Matplotlib Toolbar ---
        toolbar = NavigationToolbar2Tk(self.canvas, main_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Initialize plot
        self.plot_data()

    def load_file(self):
        """Opens a file dialog to load and process a .ccss file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("CGATS files", "*.ccss"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            self.filename_label.config(text=f"Loading: {os.path.basename(filepath)}...")
            self.root.update_idletasks() # Force UI update
            
            with open(filepath, 'r') as f:
                ccss = CGATS(f)
            
            self.spectral_data = SpectralData.from_ccss(ccss)
            self.last_filepath = filepath
            
            self.process_and_plot_data()

            self.filename_label.config(text=f"Loaded: {os.path.basename(filepath)}")
            self.export_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or process file:\n{e}")
            self.filename_label.config(text="File load failed.")
            self.export_button.config(state=tk.DISABLED)
            # Reset data
            self.spectral_data = None
            self.normalized_data = None
            self.plot_data() # Clear plot

    def process_and_plot_data(self):
        """Normalizes and plots the loaded spectral data."""
        if self.spectral_data is None:
            self.plot_data() # Plot empty
            return

        # Get the 380-780nm data
        data = self.spectral_data.sets
        
        # --- Normalize all measurements to peak at 1 ---
        # Get max of each row (axis=1)
        sdm = np.amax(data, axis=1, keepdims=True)
        
        # Divide by max, avoiding division by zero
        self.normalized_data = np.where(sdm > 0, data / sdm, data)

        self.plot_data(self.normalized_data)

    def plot_data(self, data_to_plot=None):
        """Clears and redraws the plot with new data."""
        self.ax.clear()
        
        if data_to_plot is not None:
            # Create the 380-780nm x-axis
            x_axis = np.arange(
                self.spectral_data.start_nm, self.spectral_data.end_nm + 1, 1
            )
            
            # Plot each spectrum
            for spectrum in data_to_plot:
                self.ax.plot(x_axis, spectrum, lw=0.8) # Thinner lines
            
            self.ax.set_title("Normalized Spectral Data (380-780 nm)")
            self.ax.set_ylim(0, 1.1) # Set Y-axis limit
        else:
            self.ax.set_title("Load a .ccss file to view data")
            self.ax.set_ylim(0, 1.1)

        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Normalized Intensity")
        self.ax.set_xlim(380, 780)
        self.ax.grid(True, linestyle=':', alpha=0.7)
        self.fig.tight_layout()
        self.canvas.draw()

    def export_csv(self):
        """Exports the *normalized* data to a CSV file."""
        if self.normalized_data is None:
            messagebox.showwarning("No Data", "No normalized data to export.")
            return

        # Suggest a default filename based on the input
        if self.last_filepath:
            default_name = os.path.basename(self.last_filepath)
            default_name = os.path.splitext(default_name)[0] + "_normalized.csv"
        else:
            default_name = "normalized_spectral_data.csv"

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_name
        )
        if not filepath:
            return

        try:
            # Save the normalized data, no header, one row per spectrum
            np.savetxt(filepath, self.normalized_data, delimiter=',')
            messagebox.showinfo(
                "Success", f"Successfully exported normalized data to:\n{filepath}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

# -------------------------------------------------------------------
#  MAIN EXECUTION
# -------------------------------------------------------------------

def main():
    try:
        root = tk.Tk()
        app = SpectralApp(root)
        root.mainloop()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # In case Tkinter fails to initialize
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()


