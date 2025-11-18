#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Frame, Label, Entry, Button
import os
import numpy as np
import colour.algebra.interpolation as ci
from colour.algebra import Extrapolator
from dataclasses import dataclass
import time
import struct
import locale

# --- Matplotlib Imports ---
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# --- Local Imports ---
try:
    from cgats import CGATS
except ImportError:
    print("Error: Could not find cgats.py.")
    print("Please make sure cgats.py is in the same directory.")
    exit()

try:
    from edr import EDRHeader, EDRDisplayDataHeader, EDRSpectralDataHeader, TECH_STRINGS_TO_INDEX, TECH
except ImportError:
    print("Error: Could not find edr.py.")
    print("Please make sure edr.py is in the same directory.")
    exit()


# -------------------------------------------------------------------
#  HELPER FUNCTIONS
# -------------------------------------------------------------------
def unasctime(timestr):
    """Parses a C-style asctime string into a struct_time."""
    try:
        loc = locale.getlocale()
        locale.setlocale(locale.LC_TIME, "C")
        st = time.strptime(timestr)
        locale.setlocale(locale.LC_TIME, loc)
        return st
    except ValueError:
        return time.localtime() # Fallback to now if parsing fails


# -------------------------------------------------------------------
#  SPECTRAL DATA CLASS
# -------------------------------------------------------------------

@dataclass
class SpectralData:
    """Holds and processes spectral data, ensuring 380-780nm @ 1nm output."""
    # Core spectral info
    start_nm: float
    end_nm: float
    space_nm: float
    norm: float
    num_bands: int
    num_sets: int
    sets: np.ndarray
    
    # Metadata fields
    descriptor: str = "Unknown"
    originator: str = "Unknown"
    manufacturer: str = "Unknown"
    manufacturer_id: str = "Unknown" # Added for EDR
    display: str = "Unknown"
    technology: str = "Unknown"
    created_raw: str = "" # Store original date string

    @staticmethod
    def process_raw_data(start_nm, end_nm, num_bands, sets_raw):
        """Helper to interpolate raw data to standard 380-780nm @ 1nm."""
        space_nm = (end_nm - start_nm) / (num_bands - 1) if num_bands > 1 else 1.0
        
        # If data is already perfect, just return it
        if start_nm == 380.0 and end_nm == 780.0 and space_nm == 1.0 and num_bands == 401:
             return np.array(sets_raw)

        # Handle start_nm != 380.0 logic (same as before)
        skip_fields = 0
        if start_nm > 380.0:
             # This case will be handled by extrapolation below, 
             # but we raise error if it's logically impossible for user to have meant this
             pass 
        elif start_nm < 380.0:
            skip_samples = int(round((380.0 - start_nm) / space_nm))
            skip_fields += skip_samples
            num_bands -= skip_samples
            start_nm = 380.0
            print(f"Warning: spectral data starts at {start_nm} nm, skipping leading bands.")
            
            # Slice the raw sets
            sets_raw = [s[skip_fields:] for s in sets_raw]
            
            # Re-calc end if needed (though end_nm usually stays same unless we cut past it)
            end_nm = start_nm + (num_bands - 1) * space_nm

        # Convert to numpy
        sets_orig = np.array(sets_raw)
        num_sets = len(sets_raw)

        # Interpolation
        x_orig = np.linspace(start_nm, end_nm, num_bands)
        x_target = np.arange(380, 780 + 1, 1)  # 401 points
        sets_final = np.zeros((num_sets, x_target.shape[0]))

        for i, y_orig in enumerate(sets_orig):
            interpolator = ci.SpragueInterpolator(x_orig, y_orig)
            extrapolator = Extrapolator(
                interpolator=interpolator,
                method='Constant',
                left=0.0,
                right=0.0
            )
            y_target = extrapolator(x_target)
            sets_final[i] = y_target
            
        return sets_final

    @staticmethod
    def from_ccss(ccss: CGATS):
        orig_num_bands = int(ccss["SPECTRAL_BANDS"])
        orig_start_nm = float(ccss["SPECTRAL_START_NM"])
        orig_end_nm = float(ccss["SPECTRAL_END_NM"])
        norm = float(ccss["SPECTRAL_NORM"])
        
        # Extract raw values
        # Always skip the first field, SAMPLE_ID
        setslist = []
        for data in ccss.data:
            setslist.append([float(val) for val in data[1:]]) # Skip index 0
            
        final_sets = SpectralData.process_raw_data(
            orig_start_nm, orig_end_nm, orig_num_bands, setslist
        )

        return SpectralData(
            start_nm=380.0, end_nm=780.0, space_nm=1.0, norm=norm,
            num_bands=final_sets.shape[1], num_sets=final_sets.shape[0],
            sets=final_sets,
            descriptor=ccss.params.get("DESCRIPTOR", "From CCSS"),
            originator=ccss.params.get("ORIGINATOR", "Unknown"),
            manufacturer=ccss.params.get("MANUFACTURER", "Unknown"),
            manufacturer_id=ccss.params.get("MANUFACTURER_ID", "Unknown"),
            display=ccss.params.get("DISPLAY", "Unknown"),
            technology=ccss.params.get("TECHNOLOGY", "Unknown"),
            created_raw=ccss.params.get("CREATED", "")
        )

    @staticmethod
    def from_edr(file_obj):
        """Reads a binary EDR file."""
        # Read Header
        edr_header = EDRHeader.unpack_from(file_obj.read(EDRHeader.struct.size))
        
        setslist = []
        
        # Read Sets
        for _ in range(edr_header.num_sets):
            # Read Display Data Header (skip for now)
            file_obj.read(EDRDisplayDataHeader.struct.size)
            
            # Read Spectral Data Header
            spec_header = EDRSpectralDataHeader.unpack_from(
                file_obj.read(EDRSpectralDataHeader.struct.size)
            )
            
            # Read raw doubles
            # array of doubles (8 bytes each)
            raw_bytes = file_obj.read(8 * spec_header.num_samples)
            # Use struct.unpack for each double
            fmt = f"<{spec_header.num_samples}d"
            raw_vals = struct.unpack(fmt, raw_bytes)
            
            # Convert W -> mW (multiply by 1000) to match CCSS standard
            # This ensures internal consistency so the app can process everything as mW
            setslist.append([v * 1000.0 for v in raw_vals])

        # Although EDR stores start/end, we must infer number of bands from data
        # Usually EDR is 380-780 @ 1nm or similar.
        # We assume uniform spacing based on header start/end and sample count.
        
        final_sets = SpectralData.process_raw_data(
            edr_header.spectral_start_nm,
            edr_header.spectral_end_nm,
            len(setslist[0]) if setslist else 0,
            setslist
        )
        
        # Decode metadata bytes to strings
        def decode(b): return b.decode('ascii', 'ignore').strip('\x00')

        tech_str = TECH.get(edr_header.tech_type, "Unknown")

        return SpectralData(
            start_nm=380.0, end_nm=780.0, space_nm=1.0, 
            norm=edr_header.spectral_norm,
            num_bands=final_sets.shape[1], num_sets=final_sets.shape[0],
            sets=final_sets,
            descriptor=decode(edr_header.display_description),
            originator=decode(edr_header.creation_tool),
            manufacturer=decode(edr_header.display_manufacturer),
            manufacturer_id=decode(edr_header.display_manufacturer_id),
            display=decode(edr_header.display_model),
            technology=tech_str,
            created_raw=time.ctime(edr_header.creation_time)
        )


# -------------------------------------------------------------------
#  METADATA DIALOG CLASS
# -------------------------------------------------------------------

class MetadataDialog(Toplevel):
    """A modal dialog to edit CCSS metadata before exporting."""
    def __init__(self, parent_widget, app_instance, current_data: SpectralData, export_callback, file_ext=".ccss"):
        super().__init__(parent_widget)
        self.title(f"Edit Metadata ({file_ext})")
        self.transient(parent_widget)
        self.grab_set()
        self.parent_widget = parent_widget
        self.app_instance = app_instance
        self.current_data = current_data
        self.export_callback = export_callback
        self.file_ext = file_ext
        
        self.entries = {}
        
        # Fields to show
        self.metadata_fields = [
            'descriptor', 'originator', 'manufacturer', 
            'manufacturer_id', 'display', 'technology'
        ]

        frame = Frame(self, padx=15, pady=15)
        frame.pack(fill=tk.BOTH, expand=1)

        for i, field_name in enumerate(self.metadata_fields):
            Label(frame, text=f"{field_name.replace('_', ' ').capitalize()}:").grid(
                row=i, column=0, sticky=tk.W, pady=2
            )
            
            entry = Entry(frame, width=50)
            entry.grid(row=i, column=1, padx=5, pady=2)
            
            current_value = getattr(current_data, field_name, "")
            entry.insert(0, current_value)
            
            self.entries[field_name] = entry

        btn_frame = Frame(frame)
        btn_frame.grid(row=len(self.metadata_fields), column=0, columnspan=2, pady=10)

        save_btn = Button(
            btn_frame, text="Save & Export", command=self.on_save_export
        )
        save_btn.pack(side=tk.LEFT, padx=5)

        cancel_btn = Button(btn_frame, text="Cancel", command=self.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        self.update_idletasks()
        x = parent_widget.winfo_x() + (parent_widget.winfo_width() - self.winfo_width()) // 2
        y = parent_widget.winfo_y() + (parent_widget.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
        
        self.wait_window()

    def on_save_export(self):
        """Read values from entries, update data, and trigger export."""
        for field_name, entry in self.entries.items():
            setattr(self.current_data, field_name, entry.get())
            
        if self.app_instance.last_filepath:
            default_name = os.path.basename(self.app_instance.last_filepath)
            default_name = os.path.splitext(default_name)[0] + "_normalized" + self.file_ext
        else:
            default_name = "normalized_spectral_data" + self.file_ext

        if self.file_ext == ".edr":
            ftypes = [("EDR files", "*.edr"), ("All files", "*.*")]
        else:
            ftypes = [("CGATS files", "*.ccss"), ("All files", "*.*")]

        filepath = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=self.file_ext,
            filetypes=ftypes,
            initialfile=default_name,
            title=f"Save Normalized {self.file_ext.upper()}"
        )
        
        if not filepath:
            return

        try:
            self.export_callback(self.current_data, filepath)
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}", parent=self)


# -------------------------------------------------------------------
#  GUI APPLICATION CLASS
# -------------------------------------------------------------------

class SpectralApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectral Data Viewer & Converter")
        self.root.geometry("800x600")

        self.spectral_data: SpectralData = None
        self.normalized_data = None
        self.last_filepath = None
        
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=1, padx=10, pady=10)

        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(fill=tk.X)

        self.load_button = tk.Button(
            controls_frame, text="Load File (.ccss, .edr, .csv)", command=self.load_file
        )
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.export_csv_button = tk.Button(
            controls_frame, text="Export .csv", command=self.export_csv, state=tk.DISABLED
        )
        self.export_csv_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.export_ccss_button = tk.Button(
            controls_frame, text="Export .ccss", command=self.export_ccss, state=tk.DISABLED
        )
        self.export_ccss_button.pack(side=tk.LEFT, padx=5, pady=5)

        # --- NEW EDR EXPORT BUTTON ---
        self.export_edr_button = tk.Button(
            controls_frame, text="Export .edr", command=self.export_edr, state=tk.DISABLED
        )
        self.export_edr_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.filename_label = tk.Label(
            controls_frame, text="No file loaded.", relief=tk.SUNKEN, anchor=tk.W
        )
        self.filename_label.pack(side=tk.LEFT, fill=tk.X, expand=1, padx=5, pady=5)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1, pady=5)
        toolbar = NavigationToolbar2Tk(self.canvas, main_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.plot_data()


    def load_file(self):
        filepath = filedialog.askopenfilename(
            title="Select a spectral file",
            filetypes=[("Spectral Files", "*.ccss *.edr *.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            self.filename_label.config(text=f"Loading: {os.path.basename(filepath)}...")
            self.root.update_idletasks()
            
            if filepath.lower().endswith('.ccss'):
                with open(filepath, 'r') as f:
                    ccss = CGATS(f)
                self.spectral_data = SpectralData.from_ccss(ccss)
            
            elif filepath.lower().endswith('.edr'):
                with open(filepath, 'rb') as f:
                    self.spectral_data = SpectralData.from_edr(f)
            
            elif filepath.lower().endswith('.csv'):
                loaded_sets = np.loadtxt(filepath, delimiter=',')
                if loaded_sets.ndim == 1:
                    loaded_sets = loaded_sets.reshape(1, -1)
                num_sets, num_bands = loaded_sets.shape
                
                if num_bands != 401:
                    raise ValueError(f"CSV must have 401 columns (380-780nm). Has {num_bands}.")
                
                self.spectral_data = SpectralData(
                    start_nm=380.0, end_nm=780.0, space_nm=1.0, norm=1.0, 
                    num_bands=num_bands, num_sets=num_sets, sets=loaded_sets,
                    descriptor="Loaded from CSV", originator="Spectral-GUI",
                    manufacturer="Unknown", manufacturer_id="Unknown",
                    display="Unknown", technology="Unknown", created_raw=""
                )
            else:
                raise ValueError("Unsupported file type.")

            self.last_filepath = filepath
            self.process_and_plot_data()
            self.filename_label.config(text=f"Loaded: {os.path.basename(filepath)}")
            
            self.export_csv_button.config(state=tk.NORMAL)
            self.export_ccss_button.config(state=tk.NORMAL)
            self.export_edr_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            self.filename_label.config(text="File load failed.")
            self.export_csv_button.config(state=tk.DISABLED)
            self.export_ccss_button.config(state=tk.DISABLED)
            self.export_edr_button.config(state=tk.DISABLED)
            self.spectral_data = None
            self.normalized_data = None
            self.plot_data()

    def process_and_plot_data(self):
        if self.spectral_data is None:
            self.plot_data()
            return

        data = self.spectral_data.sets
        sdm = np.amax(data, axis=1, keepdims=True)
        self.normalized_data = np.where(sdm > 0, data / sdm, data)
        self.plot_data(self.normalized_data)

    def plot_data(self, data_to_plot=None):
        self.ax.clear()
        if data_to_plot is not None:
            x_axis = np.arange(self.spectral_data.start_nm, self.spectral_data.end_nm + 1, 1)
            for spectrum in data_to_plot:
                self.ax.plot(x_axis, spectrum, lw=0.8)
            self.ax.set_title("Normalized Spectral Data (380-780 nm)")
            self.ax.set_ylim(0, 1.1)
        else:
            self.ax.set_title("Load a .ccss or .csv file to view data")
            self.ax.set_ylim(0, 1.1)
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Normalized Intensity")
        self.ax.set_xlim(380, 780)
        self.ax.grid(True, linestyle=':', alpha=0.7)
        self.fig.tight_layout()
        self.canvas.draw()

    def export_csv(self):
        if self.normalized_data is None: return
        if self.last_filepath:
            default_name = os.path.basename(self.last_filepath)
            default_name = os.path.splitext(default_name)[0] + "_normalized.csv"
        else:
            default_name = "normalized_spectral_data.csv"
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            initialfile=default_name, title="Save Normalized CSV"
        )
        if filepath:
            try:
                np.savetxt(filepath, self.normalized_data, delimiter=',')
                messagebox.showinfo("Success", f"Exported to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def create_ccss_string(self, spec_data: SpectralData, data_array: np.ndarray) -> str:
        num_sets, num_bands = data_array.shape
        lines = ["CCSS", "", f"DESCRIPTOR \"{spec_data.descriptor}\"", 
                 f"ORIGINATOR \"{spec_data.originator}\"", f"CREATED \"{time.ctime()}\"",
                 f"MANUFACTURER \"{spec_data.manufacturer}\"", f"DISPLAY \"{spec_data.display}\"",
                 f"TECHNOLOGY \"{spec_data.technology}\"", f"SPECTRAL_BANDS {num_bands}",
                 f"SPECTRAL_START_NM {spec_data.start_nm:.6f}", f"SPECTRAL_END_NM {spec_data.end_nm:.6f}",
                 f"SPECTRAL_NORM {spec_data.norm:.6f}", ""]
        
        if spec_data.manufacturer_id and spec_data.manufacturer_id != "Unknown":
             lines.insert(6, f"MANUFACTURER_ID \"{spec_data.manufacturer_id}\"")

        lines.append(f"NUMBER_OF_FIELDS {num_bands + 1}")
        lines.append("BEGIN_DATA_FORMAT")
        fields = ["SAMPLE_ID"] + [f"SPEC_{i}" for i in range(380, 781)]
        lines.append(" ".join(fields))
        lines.append("END_DATA_FORMAT")
        lines.append("")
        lines.append(f"NUMBER_OF_SETS {num_sets}")
        lines.append("BEGIN_DATA")
        for i, spectrum in enumerate(data_array):
            spec_str = [f"{val:.8f}" for val in spectrum]
            lines.append(" ".join([f"{i+1}"] + spec_str))
        lines.append("END_DATA")
        return "\n".join(lines)

    def export_ccss(self):
        if self.normalized_data is None: return
        MetadataDialog(self.root, self, self.spectral_data, self.do_ccss_export, ".ccss")

    def do_ccss_export(self, final_spec_data: SpectralData, filepath: str):
        content = self.create_ccss_string(final_spec_data, self.normalized_data)
        with open(filepath, 'w') as f: f.write(content)
        messagebox.showinfo("Success", f"Exported to:\n{filepath}")

    # --- EDR EXPORT LOGIC ---

    def export_edr(self):
        if self.spectral_data is None: return
        # NOTE: EDR usually wants non-normalized data (W/nm/m^2).
        # We use the raw spectral_data.sets, NOT self.normalized_data.
        MetadataDialog(self.root, self, self.spectral_data, self.do_edr_export, ".edr")

    def do_edr_export(self, data: SpectralData, filepath: str):
        """Writes the binary EDR file."""
        
        # 1. Prepare EDR Header
        edr_header = EDRHeader()
        edr_header.display_description = data.descriptor.encode('ascii', 'ignore')
        edr_header.creation_tool = (data.originator + " (Spectral-GUI)").encode('ascii', 'ignore')
        edr_header.display_manufacturer = data.manufacturer.encode('ascii', 'ignore')
        edr_header.display_manufacturer_id = data.manufacturer_id.encode('ascii', 'ignore')
        edr_header.display_model = data.display.encode('ascii', 'ignore')
        
        # Parse time or default to now
        if data.created_raw:
            st = unasctime(data.created_raw)
            edr_header.creation_time = int(time.mktime(st))
        else:
            edr_header.creation_time = int(time.time())

        # Handle Technology Mapping
        tech = data.technology
        # Remove common suffixes to match dictionary keys if needed
        if tech not in TECH_STRINGS_TO_INDEX and tech.endswith(" IPS"): tech = tech[:-4]
        elif tech not in TECH_STRINGS_TO_INDEX and tech.endswith(" VPA"): tech = tech[:-4]
        elif tech not in TECH_STRINGS_TO_INDEX and tech.endswith(" TFT"): tech = tech[:-4]
        
        if tech in TECH_STRINGS_TO_INDEX:
            edr_header.tech_type = TECH_STRINGS_TO_INDEX[tech]
        else:
            print(f"Warning: Unknown technology '{tech}', defaulting to 1 (Custom)")
            edr_header.tech_type = 1

        edr_header.spectral_start_nm = data.start_nm
        edr_header.spectral_end_nm = data.end_nm
        edr_header.spectral_norm = data.norm
        edr_header.num_sets = data.num_sets
        edr_header.has_spectral_data = 1

        # 2. Write Binary File
        with open(filepath, 'wb') as f:
            # Write Main Header
            f.write(edr_header.pack())

            display_data_header = EDRDisplayDataHeader()
            
            # Prepare Spectral Header
            spectral_header = EDRSpectralDataHeader()
            spectral_header.num_samples = data.num_bands

            # Write Sets
            # NOTE: We use the *raw* data sets from self.spectral_data.sets
            # because EDR expects absolute values. We also divide by 1000.0
            # to convert from mW to W, matching standard ccss2edr logic.
            for spectrum in data.sets:
                f.write(display_data_header.pack())
                f.write(spectral_header.pack())
                
                for val in spectrum:
                    # Convert mW -> W as per ccss2edr standard
                    val_w = val / 1000.0
                    f.write(struct.pack("<d", val_w))

        messagebox.showinfo("Success", f"Exported EDR to:\n{filepath}")

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
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()