#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:
import os
import re
import math
import shutil
import subprocess

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.dates import DateFormatter
from dateutil.tz import tzutc, tzlocal
from scipy import stats
from scipy.optimize import brentq, curve_fit, fsolve


import sys
import signal
import tempfile
import time
from collections import Counter
from IPython.display import clear_output

def create_tides_timeseries_string(tides_df, tsf_file_path):
    """
    Creates a string formatted as a tide file.
    
    Parameters:
        tides_df (pd.DataFrame): DataFrame with 'accumulated_minutes' and 'Predicted_m' columns.
    
    Returns:
        str: The formatted tide data as a single string.
    """

    xys_dict = {}
    
    with open(tsf_file_path, "r") as file:
        for line in file:
            match = re.match(r'^XYS\s+(\d+)\s+\d+\s+"(.+?)"', line)
            if match:
                number = int(match.group(1))
                key = match.group(2)
                xys_dict[key] = number
    
    tide_number = xys_dict['TIDES']

    lines = [f'XYS {tide_number} {len(tides_df)} "TIDES"']
    
    for _, row in tides_df.iterrows():
        minutes = float(row['accumulated_minutes'])
        height = float(row['Predicted_m'])
        lines.append(f"{minutes:.1f} {height:.9f}")

    return "\n".join(lines)


def create_streamflow_timeseries_string(streamflow_df, tsf_file_path):
    """
    Creates a streamflow .txt formatted string.

    Parameters:
        streamflow_df (pd.DataFrame): DataFrame with 'accumulated_minutes' and 'm3_s' columns.

    Returns:
        str: The streamflow content as a string.
    """
    # Read the file and extract XYS lines into a dictionary
    xys_dict = {}
    
    with open(tsf_file_path, "r") as file:
        for line in file:
            match = re.match(r'^XYS\s+(\d+)\s+\d+\s+"(.+?)"', line)
            if match:
                number = int(match.group(1))
                key = match.group(2)
                xys_dict[key] = number

    
    flow_number = xys_dict['FLOW']

    lines = [f'XYS {flow_number} {len(streamflow_df)} "FLOW"']
    for _, row in streamflow_df.iterrows():
        minutes = float(row['accumulated_minutes'])
        flow = float(row['m3_s'])
        lines.append(f"{minutes:.1f} {flow:.9f}")

    return "\n".join(lines)


# def tsf_GSSHA_file(tides_df, streamflow_df, tsf_file_name, txt_file_path, OG_GSSHA_file_path):
#     # Define the output file path
    
#     tide_txt_str = create_tides_timeseries_string(tides_df, tsf_file_path = OG_GSSHA_file_path)
#     streamflow_txt_str = create_streamflow_timeseries_string(streamflow_df, tsf_file_path = OG_GSSHA_file_path)
#     final_combined_txt = streamflow_txt_str + "\n" + tide_txt_str


#     full_file_path = txt_file_path / tsf_file_name
#     # Write the string to file
#     with open(full_file_path, 'w') as file:
#         file.write(final_combined_txt)

#     print(f"✅ Forcing file saved to: {full_file_path}")
#     return str(full_file_path)

def tsf_GSSHA_file_tides(tides_df,  tsf_file_name, txt_file_path, OG_GSSHA_file_path):
    # Define the output file path
    
    tide_txt_str = create_tides_timeseries_string(tides_df, tsf_file_path = OG_GSSHA_file_path)



    full_file_path = txt_file_path / tsf_file_name
    # Write the string to file
    with open(full_file_path, 'w') as file:
        file.write(tide_txt_str)

    print(f"✅ Forcing file saved to: {full_file_path}")
    return str(full_file_path)


def gag_GSSHA_file(gag_file_name, txt_file_path, gage_name, x_UTMcoord, y_UTMcoord, raw_dataframe):
    """
    Creates a .gag file and saves it into a local subdirectory named after the first date in the DataFrame.
    The subdirectory is created relative to the script's current directory (i.e., ./events/YYYY-MM-DD).

    Parameters:
        gag_file_name (str): Name of the output file (e.g., "gage.gag").
        gage_name (str): Name of the rain gage for inside the gag file
        x_UTMcoord (float or str): UTM X coordinate.
        y_UTMcoord (float or str): UTM Y coordinate.
        raw_dataframe (pd.DataFrame): DataFrame with 'DateTime' and 'Incremental_RF_mm'.
    """
    # Extract first date from the DataFrame
    first_date = pd.to_datetime(raw_dataframe['DateTime'].iloc[0]).date()
    folder_path = Path.cwd() / "events" / str(first_date)

    # Create folder if it doesn't exist
    folder_path.mkdir(parents=True, exist_ok=True)

    # Define full file path
    full_file_path = txt_file_path / gag_file_name

    # Prepare lines for the .gag file
    lines = [
        'EVENT "Rain Gage"',
        "NRGAG 1",
        f"NRPDS {len(raw_dataframe)}",
        f'COORD {x_UTMcoord} {y_UTMcoord} "{gage_name}"'
    ]

    for _, row in raw_dataframe.iterrows():
        dt = pd.to_datetime(row['DateTime'])
        rf = float(row['Incremental_RF_mm'])
        lines.append(f"GAGES {dt.year} {dt.month:02} {dt.day:02} {dt.hour:02} 00 {rf:.3f}")

    # Write to file
    with open(full_file_path, 'w') as file:
        file.write("\n".join(lines))

    return str(full_file_path)


def multi_gag_GSSHA_file(gag_file_name, export_path, gage_info_list, df_list):
    """
    Creates a .gag file with multiple rain gages.
    
    Parameters:
        gag_file_name (str): Output file name (e.g., "gage.gag").
        export_path (Path): Output directory.
        gage_info_list (list of tuples): Each tuple contains (gage_name, x_UTMcoord, y_UTMcoord).
        df_list (list of pd.DataFrame): List of DataFrames for each gage. Must have 'DateTime' and 'RF_mm'.
    """

    first_date = pd.to_datetime(df_list[0]['DateTime'].iloc[0]).date()

    
    # Extract the first date from the first DataFrame
    first_date = pd.to_datetime(df_list[0]['DateTime'].iloc[0]).date()

    full_file_path = export_path / gag_file_name

    # Header
    lines = ['EVENT "Rain Gage"', f"NRGAG {len(df_list)}", f"NRPDS {len(df_list[0])}"]
    
    # COORD lines for each gage
    for (name, x, y) in gage_info_list:
        lines.append(f'COORD {x} {y} "{name}"')

    # Merge DataFrames on DateTime
    merged_df = df_list[0][['DateTime']].copy()
    for i, df in enumerate(df_list):
        merged_df = merged_df.merge(df[['DateTime', 'RF_mm']], on='DateTime', suffixes=('', f'_g{i+1}'))

    # GAGES lines
    for _, row in merged_df.iterrows():
        dt = pd.to_datetime(row['DateTime'])
        rf_values = [f"{float(row[col]):.3f}" for col in merged_df.columns if col.startswith('RF_mm')]
        line = f"GAGES {dt.year} {dt.month:02} {dt.day:02} {dt.hour:02} 00 " + " ".join(rf_values)
        lines.append(line)

    # Write to file
    with open(full_file_path, 'w') as f:
        f.write("\n".join(lines))
    print(full_file_path)
    return full_file_path

# Function to generate GSSHA-compatible streamflow string
def create_gssha_streamflow_format(df_flow):
    """
    Creates a GSSHA streamflow string in the format:
    GSSHA_TS
    "FLOW"
    ABSOLUTE
    YYYY MM DD HH mm flow
    ...
    END_TS
    """
    lines = ["GSSHA_TS", '"FLOW"', "ABSOLUTE"]
    
    # Ensure DateTime is datetime format
    df_flow["DateTime"] = pd.to_datetime(df_flow["DateTime"])

    for _, row in df_flow.iterrows():
        dt = row["DateTime"]
        flow = float(row["m3_s"])
        line = f"{dt.year} {dt.month:02} {dt.day:02} {dt.hour} {dt.minute:02} {flow:.6f}"
        lines.append(line)
    
    lines.append("END_TS")
    return "\n".join(lines)


# Function to generate GSSHA-compatible tide string
def create_gssha_tide_string(df_tides):
    """
    Creates a GSSHA-compatible tide time series string in the format:
    GSSHA_TS
    "TIDES"
    ABSOLUTE
    YYYY MM DD HH mm value
    ...
    END_TS
    """
    lines = ["GSSHA_TS", '"TIDES"', "ABSOLUTE"]

    # Ensure DateTime is in datetime format
    df_tides["DateTime"] = pd.to_datetime(df_tides["DateTime"])

    for _, row in df_tides.iterrows():
        dt = row["DateTime"]
        height = float(row["Predicted_m"])
        lines.append(f"{dt.year} {dt.month:02} {dt.day:02} {dt.hour} {dt.minute:02} {height:.6f}")

    lines.append("END_TS")
    return "\n".join(lines)


def xys_GSSHA_file(df_tides,  xys_file_name, txt_file_path): #df_streamflow,
    #flow_str = create_gssha_streamflow_format(df_streamflow)
    tide_str = create_gssha_tide_string(df_tides)
    
    combined_str = f"{tide_str}"
    output_dir = Path(txt_file_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / xys_file_name
    with open(output_file, 'w') as file:
        file.write(combined_str)
    return str(output_file)

def convert_df_to_prj(df_prj, output_folder, prj_file_name):
    """
    Converts df_prj (with columns: Parameter, Value, [commented]) into a .prj config file for GSSHA.
    Adds GSSHA header at the top.

    Parameters:
    - df_prj (pd.DataFrame): DataFrame with at least 'Parameter' and 'Value' columns.
    - output_folder (str or Path): Folder where the .prj file will be saved.
    - prj_file_name (str): Name of the .prj file (without extension).
    """
    # Ensure output folder ex

    # Define output file path
    output_path = output_folder / f"{prj_file_name}.prj"

    # Open and write line-by-line
    with open(output_path, 'w', encoding='utf-8') as f:
        # Add GSSHA header
        f.write("GSSHAPROJECT\n")
        f.write("WMS WMS 11.1.10 (64-bit)\n")

        for _, row in df_prj.iterrows():
            key = str(row['Parameter']).strip()
            value = str(row['Value']).strip()
            
            # Optional 'commented' column
            is_commented = False
            if 'commented' in df_prj.columns:
                is_commented = str(row['commented']).strip().lower() in ['true', '1', 'yes', '#']

            # Add quotes if value looks like a file path
            if "\\" in value or "/" in value:
                value = f'{value}'

            line = f"{key:<25} {value}"
            if is_commented:
                line = f"#{line}"
            
            f.write(line + '\n')

    print(f"✅ Successfully wrote: {output_path}")
    return output_path

def read_prj_file(GSSHA_prj_name, prj_folder_path):
    #read the prj file
    prj_path = prj_folder_path / GSSHA_prj_name 
    
    # Read lines, skipping the first two GSSHA header lines
    with open(prj_path, "r") as file:
        lines = file.readlines()[2:]
    
    # Parse lines into [Parameter, Value] lists
    rows = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue  # Skip empty lines
    
        parts = stripped.split(None, 1)  # split on first space(s)
        if len(parts) == 2:
            rows.append([parts[0], parts[1]])
        else:
            rows.append([parts[0], ""])
    
    # Create the DataFrame
    df_prj = pd.DataFrame(rows, columns=["Parameter", "Value"])
    df_prj = df_prj.set_index("Parameter")

    return df_prj


def copy_gssha_apps_to_model(GSSHA_SOURCE_DIR, MODEL_DIR):
    for file in os.listdir(GSSHA_SOURCE_DIR):
        src = os.path.join(GSSHA_SOURCE_DIR, file)
        dst = os.path.join(MODEL_DIR, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    print("✔️ GSSHA applications copied to model folder.")


def run_gssha(MODEL_DIR, PROJECT_FILE):
    exe_path = os.path.join(MODEL_DIR, 'gssha.exe')
    prj_path = os.path.join(MODEL_DIR, PROJECT_FILE)

    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"GSSHA executable not found: {exe_path}")
    if not os.path.exists(prj_path):
        raise FileNotFoundError(f"Project file not found: {prj_path}")

    print(f"📦 Running GSSHA:\n  Executable: {exe_path}\n  Project File: {prj_path}")
    
    result = subprocess.run(
        [exe_path, prj_path],
        cwd=MODEL_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Save logs to file for debugging
    with open(os.path.join(MODEL_DIR, "gssha_log.txt"), "w") as f:
        f.write("---- STDOUT ----\n")
        f.write(result.stdout + "\n")
        f.write("---- STDERR ----\n")
        f.write(result.stderr + "\n")

    if result.returncode != 0:
        print("❌ GSSHA failed. Check gssha_log.txt for details.")
    else:
        print("✅ GSSHA simulation ran successfully. Check gssha_log.txt for output.")

def start_q_window(stop_file, close_file):
    """
    Open a small window where the user can type q and press Enter.

    Creates stop_file when q is entered.
    Closes automatically when close_file appears.
    """

    window_code = r"""
import os
import sys
import tkinter as tk

stop_file = sys.argv[1]
close_file = sys.argv[2]

root = tk.Tk()
root.title("GSSHA Control")
root.geometry("360x150")
root.resizable(False, False)

label = tk.Label(
    root,
    text="Type q and press Enter to stop GSSHA:",
    font=("Arial", 11)
)
label.pack(pady=(20, 8))

entry = tk.Entry(
    root,
    width=15,
    justify="center",
    font=("Arial", 14)
)
entry.pack()
entry.focus_force()

status = tk.Label(
    root,
    text="GSSHA is running",
    font=("Arial", 10)
)
status.pack(pady=8)


def submit_command(event=None):
    command = entry.get().strip().lower()

    if command == "q":
        with open(stop_file, "w") as file:
            file.write("stop")

        status.config(text="Shutdown requested...")
        entry.config(state="disabled")

    else:
        entry.delete(0, tk.END)
        status.config(text="Type q, then press Enter.")


def check_close_signal():
    if os.path.exists(close_file):
        root.destroy()
        return

    root.after(250, check_close_signal)


entry.bind("<Return>", submit_command)

# Allow manual window closing without stopping GSSHA.
root.protocol("WM_DELETE_WINDOW", root.destroy)

check_close_signal()
root.mainloop()
"""

    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            window_code,
            str(stop_file),
            str(close_file)
        ],
        creationflags=subprocess.CREATE_NO_WINDOW
    )

def read_GSSHA_dep(folder_path, filename):
    """
    Read a GSSHA DEP file, including a potentially incomplete final
    timestep.

    Returns
    -------
    dict
        {
            "timesteps": list[float],
            "depths": list[np.ndarray]
        }
    """

    dep_file = os.path.join(folder_path, filename)

    timesteps = []
    depths = []

    current_timestep = None
    current_depths = []
    reading = False

    with open(
        dep_file,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("TS"):
                # Save the previous timestep before beginning the new one.
                if reading and current_timestep is not None:
                    timesteps.append(current_timestep)
                    depths.append(
                        np.asarray(current_depths, dtype=np.float64)
                    )

                parts = line.split()

                if len(parts) < 3:
                    continue

                current_timestep = float(parts[2])
                current_depths = []
                reading = True
                continue

            if line == "ENDDS":
                if reading and current_timestep is not None:
                    timesteps.append(current_timestep)
                    depths.append(
                        np.asarray(current_depths, dtype=np.float64)
                    )

                reading = False
                current_timestep = None
                current_depths = []
                break

            if reading:
                try:
                    current_depths.append(float(line))
                except ValueError:
                    # This may occur if GSSHA is writing the exact line
                    # while Python is reading the file.
                    continue

    # During a live run, the file may end before ENDDS.
    # Save the current block so calculate_dep_changes() can determine
    # whether it is complete.
    if reading and current_timestep is not None:
        timesteps.append(current_timestep)
        depths.append(
            np.asarray(current_depths, dtype=np.float64)
        )

    return {
        "timesteps": timesteps,
        "depths": depths
    }

def calculate_dep_changes(
    dep_data,
    cell_size,
    depth_threshold=0.01
):
    """
    Calculate depth and inundated-area changes between consecutive
    complete GSSHA DEP timesteps.
    """

    timesteps = dep_data["timesteps"]
    depths = dep_data["depths"]

    output_columns = [
        "timestep",
        "cumulative_change",
        "max_positive_change_per_cell",
        "inundated_area_change"
    ]

    if len(timesteps) != len(depths):
        raise ValueError(
            "dep_data['timesteps'] and dep_data['depths'] must have "
            f"the same length. Found {len(timesteps)} timesteps and "
            f"{len(depths)} depth arrays."
        )

    if len(depths) < 2:
        return pd.DataFrame(columns=output_columns)

    flattened_depths = [
        np.asarray(depth, dtype=np.float64).ravel()
        for depth in depths
    ]

    array_sizes = [
        depth.size
        for depth in flattened_depths
    ]

    expected_size = Counter(array_sizes).most_common(1)[0][0]

    complete_timesteps = []
    complete_depths = []

    for timestep, depth, array_size in zip(
        timesteps,
        flattened_depths,
        array_sizes
    ):
        if array_size != expected_size:
            print(
                f"Skipping incomplete DEP timestep {timestep}: "
                f"found {array_size:,} values; "
                f"expected {expected_size:,}."
            )
            continue

        complete_timesteps.append(timestep)
        complete_depths.append(depth)

    if len(complete_depths) < 2:
        return pd.DataFrame(columns=output_columns)

    cell_area = float(cell_size) ** 2
    results = []

    for i in range(1, len(complete_depths)):
        previous_depth = complete_depths[i - 1]
        current_depth = complete_depths[i]

        depth_change = current_depth - previous_depth

        cumulative_change = np.nansum(
            depth_change,
            dtype=np.float64
        )

        positive_change = np.maximum(
            depth_change,
            0.0
        )

        max_positive_change_per_cell = np.nanmax(
            positive_change
        )

        previous_wet_cells = np.count_nonzero(
            previous_depth > depth_threshold
        )

        current_wet_cells = np.count_nonzero(
            current_depth > depth_threshold
        )

        inundated_area_change = (
            current_wet_cells - previous_wet_cells
        ) * cell_area

        results.append({
            "timestep": complete_timesteps[i],
            "cumulative_change": cumulative_change,
            "max_positive_change_per_cell": (
                max_positive_change_per_cell
            ),
            "inundated_area_change": inundated_area_change
        })

    return pd.DataFrame(
        results,
        columns=output_columns
    )

def run_gssha_convergence_view(
    MODEL_DIR,
    PROJECT_FILE,
    DEP_FILE,
    cell_size,
    depth_threshold=0.01,
    enable_keyboard_stop=True,
    dep_check_seconds=300,
    process_check_seconds=0.5,
    display_last_n=None
):
    """
    Run GSSHA while periodically reading and analyzing a live DEP file.

    Parameters
    ----------
    MODEL_DIR : str or Path
        GSSHA model directory.

    PROJECT_FILE : str
        Name of the GSSHA project file.

    DEP_FILE : str
        Name of the DEP file being written by GSSHA.

    cell_size : float
        GSSHA grid-cell size. For a 10 m grid, use 10.

    depth_threshold : float, optional
        Minimum depth used to classify a cell as inundated.

    enable_keyboard_stop : bool, optional
        When True, opens the separate q control window.

    dep_check_seconds : float, optional
        Seconds between DEP checks. Default is 300 seconds.

    process_check_seconds : float, optional
        Seconds between checks of the GSSHA process.

    display_last_n : int or None, optional
        Number of most recent DataFrame rows to display.
        None displays the entire DataFrame.

    Returns
    -------
    int
        GSSHA return code.
    """

    model_dir = os.path.abspath(os.fspath(MODEL_DIR))

    exe_path = os.path.join(model_dir, "gssha.exe")
    prj_path = os.path.join(model_dir, PROJECT_FILE)
    dep_path = os.path.join(model_dir, DEP_FILE)
    log_path = os.path.join(model_dir, "gssha_log.txt")

    if not os.path.isfile(exe_path):
        raise FileNotFoundError(
            f"GSSHA executable not found: {exe_path}"
        )

    if not os.path.isfile(prj_path):
        raise FileNotFoundError(
            f"GSSHA project file not found: {prj_path}"
        )

    print(
        "Running GSSHA:\n"
        f"  Executable: {exe_path}\n"
        f"  Project file: {prj_path}\n"
        f"  DEP file: {dep_path}\n"
        f"  DEP check interval: {dep_check_seconds} seconds"
    )

    safe_stop_sent = False
    return_code = None
    process = None
    q_window_process = None

    latest_dep_dataframe = None
    dep_check_number = 0

    control_dir = None
    stop_file = None
    close_file = None

    if enable_keyboard_stop:
        control_dir = Path(
            tempfile.mkdtemp(prefix="gssha_control_")
        )

        stop_file = control_dir / "stop_requested.txt"
        close_file = control_dir / "close_window.txt"

    next_dep_check = (
        time.monotonic() + dep_check_seconds
    )

    try:
        with open(
            log_path,
            "w",
            encoding="utf-8",
            errors="replace",
            buffering=1
        ) as log_file:

            log_file.write("---- GSSHA OUTPUT ----\n")
            log_file.flush()

            process = subprocess.Popen(
                [exe_path, PROJECT_FILE],
                cwd=model_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                )
            )

            print(f"GSSHA started. Process ID: {process.pid}")

            if enable_keyboard_stop:
                q_window_process = start_q_window(
                    stop_file=stop_file,
                    close_file=close_file
                )

                print(
                    "The GSSHA control window is open. "
                    "Type q and press Enter there to stop."
                )

            while process.poll() is None:

                # -------------------------------------
                # Check for q-window shutdown request
                # -------------------------------------
                if (
                    enable_keyboard_stop
                    and not safe_stop_sent
                    and stop_file.exists()
                ):
                    print(
                        "\nSending safe shutdown request to GSSHA..."
                    )

                    try:
                        process.send_signal(
                            signal.CTRL_BREAK_EVENT
                        )

                        safe_stop_sent = True

                    except ProcessLookupError:
                        pass

                    except OSError as error:
                        print(
                            "Could not send shutdown signal: "
                            f"{error}"
                        )

                # -------------------------------------
                # Periodic live DEP-file calculation
                # -------------------------------------
                current_time = time.monotonic()

                if current_time >= next_dep_check:
                    dep_check_number += 1

                    # The next check is scheduled from the current
                    # time, preventing rapid catch-up checks.
                    next_dep_check = (
                        current_time + dep_check_seconds
                    )

                    clear_output(wait=True)

                    print(
                        f"GSSHA is running — DEP check "
                        f"{dep_check_number}"
                    )

                    print(f"DEP file: {DEP_FILE}")

                    if enable_keyboard_stop:
                        print(
                            "Use the GSSHA control window to stop "
                            "the simulation."
                        )

                    try:
                        if not os.path.isfile(dep_path):
                            print(
                                "\nWaiting for the DEP file to be "
                                "created."
                            )

                        else:
                            dep_data = read_GSSHA_dep(
                                folder_path=model_dir,
                                filename=DEP_FILE
                            )

                            dep_dataframe = calculate_dep_changes(
                                dep_data=dep_data,
                                cell_size=cell_size,
                                depth_threshold=depth_threshold
                            )

                            latest_dep_dataframe = dep_dataframe

                            print(
                                "\nDEP timestep blocks read: "
                                f"{len(dep_data['timesteps'])}"
                            )

                            if dep_dataframe.empty:
                                print(
                                    "\nFewer than two complete DEP "
                                    "timesteps are available."
                                )

                            elif display_last_n is None:
                                display(dep_dataframe)

                            else:
                                display(
                                    dep_dataframe.tail(
                                        display_last_n
                                    )
                                )

                    except (
                        PermissionError,
                        OSError,
                        ValueError,
                        IndexError
                    ) as error:
                        print(
                            "\nDEP file could not be processed "
                            f"during this check:\n{error}"
                        )

                time.sleep(process_check_seconds)

            # Fully reap GSSHA before proceeding.
            return_code = process.wait()

    finally:
        # Tell the q control window to close.
        if close_file is not None:
            try:
                close_file.touch(exist_ok=True)
            except OSError:
                pass

        # Wait for the q-window subprocess.
        if q_window_process is not None:
            try:
                q_window_process.wait(timeout=5)

            except subprocess.TimeoutExpired:
                q_window_process.terminate()

                try:
                    q_window_process.wait(timeout=2)

                except subprocess.TimeoutExpired:
                    q_window_process.kill()
                    q_window_process.wait()

        # Remove temporary control files.
        if control_dir is not None:
            for path in (stop_file, close_file):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

            try:
                control_dir.rmdir()
            except OSError:
                pass

    # -------------------------------------
    # Final DEP calculation after completion
    # -------------------------------------
    final_dep_error = None

    if os.path.isfile(dep_path):
        try:
            final_dep_data = read_GSSHA_dep(
                folder_path=model_dir,
                filename=DEP_FILE
            )

            latest_dep_dataframe = calculate_dep_changes(
                dep_data=final_dep_data,
                cell_size=cell_size,
                depth_threshold=depth_threshold
            )

        except (
            PermissionError,
            OSError,
            ValueError,
            IndexError
        ) as error:
            final_dep_error = error

    clear_output(wait=True)

    # -------------------------------------
    # Final run status
    # -------------------------------------
    if safe_stop_sent:
        print(
            "GSSHA exited after the shutdown request. "
            f"Return code: {return_code}"
        )

    elif return_code == 0:
        print(
            "GSSHA simulation completed normally. "
            "The q window closed automatically."
        )

    else:
        print(
            f"GSSHA exited with return code {return_code}. "
            "Check gssha_log.txt."
        )

    if final_dep_error is not None:
        print(
            "\nThe final DEP file could not be analyzed:\n"
            f"{final_dep_error}"
        )

    elif latest_dep_dataframe is None:
        print("\nNo DEP DataFrame was generated.")

    elif latest_dep_dataframe.empty:
        print(
            "\nThe DEP file does not contain at least two "
            "complete timesteps."
        )

    else:
        print("\nFinal DEP change results:")

        if display_last_n is None:
            display(latest_dep_dataframe)

        else:
            display(
                latest_dep_dataframe.tail(display_last_n)
            )

    return return_code

def move_and_rename_gssha_output(MODEL_DIR, RESULTS_DIR, output_description = "TEST_RUN", extension = "otl"):
    #otl, ows, dep, etc. extentsions
    """
    Find the first file in `model_folder` with the given extension,
    rename it, and save it in `results_folder`.
    
    Parameters:
        extension (str): File extension to look for (e.g., 'otl', 'ows').
        model_folder (str): Path to folder where GSSHA outputs are saved.
        results_folder (str): Destination to copy renamed file.
    """

    # Ensure folders exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Search for matching files
    matching_files = [f for f in os.listdir(MODEL_DIR) if f.lower().endswith(f'.{extension}')]

    if not matching_files:
        print(f"❌ No .{extension} files found in {MODEL_DIR}")
        return

    # Use first match
    original_file = matching_files[0]
    base_name = os.path.splitext(original_file)[0]

    # Create new name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_name = f"{base_name}_{output_description}.{extension}"

    # Full paths
    src_path = os.path.join(MODEL_DIR, original_file)
    dst_path = os.path.join(RESULTS_DIR, new_name)

    # Copy and rename
    shutil.copy2(src_path, dst_path)
    print(f"✅ File '{original_file}' saved as '{new_name}' in Results folder.")




def copy_text_file(
    folder_path,
    original_filename,
    output_folder,
    new_filename
):
    """
    Copies a text file and saves it with a new filename
    in the specified output folder.

    Parameters
    ----------
    folder_path : str or Path
        Folder containing the original file.

    original_filename : str
        Name of the existing text file.

    output_folder : str or Path
        Folder where the copied file will be saved.
        Created automatically if it does not exist.

    new_filename : str
        Name of the copied file.
    """

    source = Path(folder_path) / original_filename

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    destination = output_folder / new_filename

    shutil.copy2(source, destination)

    return destination


def cleanup_model_dir(MODEL_DIR):
    # Remove copied EXEs and DLLs
    for file in os.listdir(MODEL_DIR):
        if file.endswith(('.exe', '.dll', '.cfg', '.manifest')):
            os.remove(os.path.join (MODEL_DIR, file))
    print("✔️ Cleaned up model folder.")


#GSSHA auto shut down
def GSSHA_auto_shutdown(model_dir):
    """
    Launch GSSHA using only the model directory.

    Returns None when the model directory does not contain gssha.exe
    or a project file.

    Parameters
    ----------
    model_dir : str or Path
        Folder containing gssha.exe and exactly one .prj file.

    Returns
    -------
    subprocess.Popen or None
        Running GSSHA process, or None when GSSHA cannot be launched.
    """

    model_dir = Path(model_dir)

    if not model_dir.exists():
        print(f"Model directory does not exist: {model_dir}")
        return None

    gssha_exe = model_dir / "gssha.exe"

    if not gssha_exe.exists():
        print(
            "No gssha.exe found. "
            "The model folder may already be clean."
        )
        return None

    prj_files = list(model_dir.glob("*.prj"))

    if len(prj_files) == 0:
        print(
            "No .prj file found. "
            "There is no GSSHA model to launch."
        )
        return None

    if len(prj_files) > 1:
        raise RuntimeError(
            "Multiple project files found:\n"
            + "\n".join(file.name for file in prj_files)
        )

    project_file = prj_files[0]

    try:
        process = subprocess.Popen(
            [str(gssha_exe), project_file.name],
            cwd=str(model_dir)
        )

    except OSError as error:
        print(f"GSSHA could not be launched: {error}")
        return None

    print(
        f"GSSHA started with process ID {process.pid}\n"
        f"Project file: {project_file.name}"
    )

    return process
def force_shutdown_gssha(process=None, model_dir=None, timeout=15):
    """
    Force-stop GSSHA and its child processes, then wait until gssha.exe
    is no longer locked.

    Parameters
    ----------
    process : subprocess.Popen, optional
        The process returned when GSSHA was launched.

    model_dir : str or Path, optional
        Directory containing gssha.exe.

    timeout : float
        Maximum seconds to wait for process and file-lock release.

    Returns
    -------
    bool
        True when GSSHA is stopped and gssha.exe is unlocked.
    """

    if process is not None and process.poll() is None:
        print(
            f"Force-stopping GSSHA process tree "
            f"for PID {process.pid}..."
        )

        result = subprocess.run(
            [
                "taskkill",
                "/PID", str(process.pid),
                "/T",
                "/F"
            ],
            capture_output=True,
            text=True,
            shell=False
        )

        if result.stdout.strip():
            print(result.stdout.strip())

        if result.stderr.strip():
            print(result.stderr.strip())

    # Also terminate any remaining gssha.exe processes.
    # This catches orphaned runs from a restarted Jupyter kernel.
    subprocess.run(
        [
            "taskkill",
            "/IM", "gssha.exe",
            "/T",
            "/F"
        ],
        capture_output=True,
        text=True,
        shell=False
    )

    deadline = time.time() + timeout

    # Wait until Windows no longer reports a running GSSHA process.
    while time.time() < deadline:
        check = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq gssha.exe"],
            capture_output=True,
            text=True,
            shell=False
        )

        if "gssha.exe" not in check.stdout.lower():
            break

        print("Waiting for the GSSHA process to disappear...")
        time.sleep(0.5)

    else:
        print(
            "GSSHA still appears in Task Manager after "
            f"{timeout} seconds."
        )
        return False

    # Wait for the executable file lock to be released.
    if model_dir is not None:
        exe_path = Path(model_dir) / "gssha.exe"

        while time.time() < deadline:
            try:
                # Opening for append without writing tests whether another
                # process still has an incompatible lock on the file.
                with open(exe_path, "ab"):
                    pass

                print("GSSHA stopped and gssha.exe is unlocked.")
                return True

            except PermissionError:
                print(
                    "GSSHA has stopped, but Windows has not released "
                    "gssha.exe yet..."
                )
                time.sleep(0.5)

        print(
            "The GSSHA process stopped, but gssha.exe remains locked."
        )
        return False

    print("GSSHA stopped.")
    return True


#the function that chris made to deal with the otl file from wms
def process_ows_file(StartDate, OWSFile):
    
    #example start date: "2018-08-23 00:00"

    # Create the start time object to enumerate the number of minutes in the outlet file 
    StartDateTime = pd.to_datetime(StartDate)
    
    # read in the outlet file 
    OutHydro = pd.read_csv(OWSFile, header = None, delim_whitespace=True)
    OutHydro.columns = ["Minutes", "WMS_m"]
    
    # da magic: Turn stupid minutes into useful datetime objects 
    OutHydro["DateTime"] = OutHydro["Minutes"].apply(lambda x: StartDateTime + timedelta(minutes=x))
    
    
    # Set the index to the date
    OutHydro.set_index("DateTime", inplace=True)
    
    return OutHydro

#combine the observed and modeled streamflow into a clean dataframe for further analysis
def Combo_Obs_Mod_df(Start_Date_Time, OWSFile, Observed):
   # 15 min resampled data: hanalei_stream15min_resampling.csv
    #1 min data 'Hanalei_stream_16103000_HAWAII-TIME.csv'

    #read otl GSSHA file
    df_mod = process_ows_file(StartDate = Start_Date_Time, OWSFile = OWSFile)
    
    #read streamflow file
    df_obs = pd.read_csv(Observed)
    df_obs = df_obs[['DateTime', 'm' ]].copy()
    df_obs =  df_obs.set_index('DateTime')
    df_obs.index = pd.to_datetime(df_obs.index)
    
    #merge the observed streamflow to the modeled streamflow on the date index
    merged = pd.merge(df_mod, df_obs, how = 'inner', left_index = True, right_index = True)
    
    return [merged]

def RMSE(merged_dataframe):
    Oi = np.array(list(merged_dataframe.m))
    Pi = np.array(list(merged_dataframe.WMS_m))
    n = len(Oi)
    
    RMSE = math.sqrt( np.sum( (Oi - Pi)**2 ) / n  )
    return round(RMSE, 2)

def PBIAS(merged_dataframe):
    Oi = np.array(list(merged_dataframe.m))
    Pi = np.array(list(merged_dataframe.WMS_m))
    
    PBIAS = (( np.sum(Oi - Pi) ) / np.sum(Oi))*100
    return round(PBIAS, 2)

def NSE(merged_dataframe):
    Oi = np.array(list(merged_dataframe.m))
    Pi = np.array(list(merged_dataframe.WMS_m))
    O_bar = np.mean(Oi)
    
    NSE = 1 - ( np.sum( (Oi - Pi)**2 )  /  np.sum( (Oi - O_bar)**2 ) )
    return round(NSE, 2)

def max_streamflow_percent_error(merged_dataframe):
    Oi = np.array(list(merged_dataframe.m))
    Pi = np.array(list(merged_dataframe.WMS_m))
    Oi_max = np.max(Oi)
    Pi_max = np.max(Pi)
    
    PE = (( Pi_max - Oi_max) / Oi_max)*100
    return round(PE, 2)

def max_streamflow_timing_difference(merged_dataframe):
    Oi = list(merged_dataframe.m)
    Pi = list(merged_dataframe.WMS_m)
    Oi_max = max(Oi)
    Pi_max = max(Pi)
    Oi_max_index = Oi.index(Oi_max)
    Pi_max_index = Pi.index(Pi_max)
    
    date_time = merged_dataframe.index
    Oi_max_time = date_time[Oi_max_index]
    Pi_max_time = date_time[Pi_max_index]
    
    time_dif = (Pi_max_time - Oi_max_time) / pd.Timedelta(hours = 1)
    return round(time_dif, 2)

def KSE(merged_dataframe):
    Oi = np.array(list(merged_dataframe.m))
    Pi = np.array(list(merged_dataframe.WMS_m))
    O_std = np.std(Oi)
    P_std = np.std(Pi)
    O_mean = np.mean(Oi)
    P_mean = np.mean(Pi)
    r = stats.pearsonr(Oi, Pi)[0]
    KSE = 1 - math.sqrt((r-1)**2 + ((P_std/O_std) - 1)**2 + (P_mean / O_mean)**2)
    return round(KSE, 2)
    
    
def GET_STATS(merged_dataframe_start_datetime_list):
    compiled_df = []
    for i in merged_dataframe_start_datetime_list: 
        merged_dataframe = i[0]
        Start_date_time = i[0].index[0].strftime("%Y-%m-%d")

        RMSE_stat = RMSE(merged_dataframe)
        NSE_stat = NSE(merged_dataframe)
        PBIAS_stat = PBIAS(merged_dataframe)
        PE_maxstreamflow = max_streamflow_percent_error(merged_dataframe)
        E_maxstreamflow = max_streamflow_timing_difference(merged_dataframe)
        KSE_stat = KSE(merged_dataframe)


        STATS_dict = {'RMSE' : RMSE_stat , 'NSE' : NSE_stat, 'KSE': KSE_stat, 'PBIAS' : PBIAS_stat, 'Percent Error of Max height' : PE_maxstreamflow, 'Max Stream height time difference' : E_maxstreamflow }
        df = pd.DataFrame(STATS_dict, index= [Start_date_time])
        compiled_df.append(df)
    result = pd.concat(compiled_df)
    return result


def replace_value_in_gssha_file(read_dir, gssha_sample_file, save_filename, old_value, new_value):
    """
    Reads a GSSHA sample file, replaces all occurrences of '100.0' with the new_value,
    and saves the updated file in the same directory with a new filename.

    Args:
        read_dir (str): Directory where the file is located and will be saved.
        gssha_sample_file (str): The name of the GSSHA sample file to read.
        save_filename (str): Name for the saved file (with extension).
        new_value (float): The value to replace '100.0' with.
    """
    import os

    # Full path to the input file
    read_path = os.path.join(read_dir, gssha_sample_file)

    # Read the file
    with open(read_path, 'r') as file:
        content = file.read()

    # Replace all occurrences of '100.0' with the new value
    updated_content = content.replace(old_value, str(new_value))

    # Full path for the output file (same directory)
    save_path = os.path.join(read_dir, save_filename)

    # Write updated content to the new file
    with open(save_path, 'w') as file:
        file.write(updated_content)

    return save_path


def max_OWS(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    max_value = float('-inf')
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                value = float(parts[1])
                if value > max_value:
                    max_value = value
            except ValueError:
                continue  # Skip non-numeric values
    return max_value

def invert_hybrid_safe(target_wse, a, b, c, q_lo=0.0, q_hi=None, max_expand=40):
    """
    Robustly invert hybrid_fit(Q) = target_wse for Q >= 0.
    Expands the upper bracket until sign change or until max_expand attempts.
    Returns np.nan if no bracket is found (target out of modeled range).
    """
    # Lower bound (flows can't be negative)
    lo = max(0.0, float(q_lo))
    f_lo = hybrid_fit(lo, a, b, c) - target_wse

    # Start upper bound
    if q_hi is None:
        q_hi = max(1.0, float(np.nanmax(flow_valid)) if len(flow_valid) else 1.0)
    hi = float(q_hi)
    f_hi = hybrid_fit(hi, a, b, c) - target_wse

    # If already bracketed, solve
    if f_lo == 0:
        return lo
    if f_lo * f_hi < 0:
        return brentq(lambda Q: hybrid_fit(Q, a, b, c) - target_wse, lo, hi)

    # Expand upward until bracket or give up
    attempts = 0
    while attempts < max_expand and np.isfinite(f_hi) and (f_lo * f_hi > 0):
        hi *= 2.0
        f_hi = hybrid_fit(hi, a, b, c) - target_wse
        attempts += 1

    if np.isfinite(f_hi) and f_lo * f_hi < 0:
        return brentq(lambda Q: hybrid_fit(Q, a, b, c) - target_wse, lo, hi)

    # No bracket found → target outside modeled range
    return np.nan

def hybrid_fit(Q, a, b, c):
    # use log(Q+1) to avoid log(0) and allow Q=0
    return a * np.log(Q + 1.0) + b * Q + c


# In[ ]:




