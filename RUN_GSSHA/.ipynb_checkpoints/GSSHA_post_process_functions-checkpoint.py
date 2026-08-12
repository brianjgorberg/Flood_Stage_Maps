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
from scipy.optimize import brentq, curve_fit, fsolve, least_squares



import sys
import signal
import tempfile
import time
from collections import Counter
from IPython.display import clear_output





def cumulative_to_instantaneous(
    df,
    cumulative_column="cumulative_discharge",
    time_column="timestep_min",
    output_column="instantaneous_discharge",
):
    """
    Convert cumulative volume (m³) to instantaneous discharge (m³/s).

    Parameters
    ----------
    df : pandas.DataFrame
    cumulative_column : str
        Column containing cumulative volume (m³).
    time_column : str
        Column containing timestep in minutes.
    output_column : str
        Name of the output discharge column.

    Returns
    -------
    pandas.DataFrame
    """
    df = df.copy()

    dV = df[cumulative_column].diff()
    dt = df[time_column].diff() * 60.0  # minutes -> seconds

    df[output_column] = dV / dt

    # Set the first value to NaN since there is no previous timestep
    df.loc[df.index[0], output_column] = float("nan")

    return df

def read_GSSHA_oqc(filepath):
    """
    Read a GSSHA .oqc file.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - timestep_min
        - discharge
        - cumulative_discharge
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=["timestep_min", "cumulative_discharge"],
    )

    return df
def get_flow_file_dict(folder_path, extension, cfs_to_cms=0.0283168466):
    """
    Returns a dictionary mapping rounded flow (cms) to file paths.

    Parameters
    ----------
    folder_path : str or Path
        Folder containing the files.
    extension : str
        File extension (e.g., ".dep", ".xys", ".otl").
    cfs_to_cms : float, optional
        Conversion factor from cfs to cms.

    Returns
    -------
    dict
        Keys are rounded flow values (cms, int).
        Values are pathlib.Path objects.
    """
    folder_path = Path(folder_path)

    files = {
        round(int(file_path.stem.split("-")[-1]) * cfs_to_cms): file_path
        for file_path in folder_path.glob(f"*{extension}")
    }

    return dict(sorted(files.items()))


def hybrid_fit(q_inflow, a, b, c):
    """
    Predict gauge discharge from upstream inflow.

    Equation
    --------
    Q_gauge = a * Q_inflow**b + c * Q_inflow
    """
    q_inflow = np.asarray(q_inflow, dtype=float)

    return (
        a * q_inflow**b
        + c * q_inflow
    )


def fit_gauge_discharge_model(
    stable_discharge_dict,
    initial_guess=(2.0, 0.7, 0.4),
):
    """
    Fit the hybrid upstream-inflow to gauge-discharge model.

    Parameters
    ----------
    stable_discharge_dict : dict
        Keys are upstream inflows in cms.
        Values are stable gauge discharges in cms.
    initial_guess : tuple
        Starting guesses for a, b, and c.

    Returns
    -------
    dict
        Dictionary containing fitted parameters, input data,
        predictions, and model diagnostics.
    """
    q_inflow = np.array(
        sorted(stable_discharge_dict.keys()),
        dtype=float,
    )

    q_gauge = np.array(
        [stable_discharge_dict[q] for q in q_inflow],
        dtype=float,
    )

    parameters, covariance = curve_fit(
        hybrid_fit,
        q_inflow,
        q_gauge,
        p0=initial_guess,
        bounds=(
            [0.0, 0.01, 0.0],
            [np.inf, 0.99, np.inf],
        ),
        maxfev=50000,
    )

    a, b, c = parameters

    q_gauge_predicted = hybrid_fit(
        q_inflow,
        a,
        b,
        c,
    )

    residuals = q_gauge - q_gauge_predicted

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum(
        (q_gauge - q_gauge.mean())**2
    )

    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(residuals**2))

    return {
        "model_type": "hybrid_power_linear",
        "equation": "Q_gauge = a * Q_inflow**b + c * Q_inflow",
        "parameters": {
            "a": float(a),
            "b": float(b),
            "c": float(c),
        },
        "covariance": covariance,
        "r_squared": float(r_squared),
        "rmse": float(rmse),
        "q_inflow": q_inflow,
        "q_gauge_observed": q_gauge,
        "q_gauge_predicted": q_gauge_predicted,
        "residuals": residuals,
        "source_data": stable_discharge_dict.copy(),
    }


def predict_gauge_discharge(
    q_inflow,
    gauge_discharge_model,
):
    """
    Predict gauge discharge using a stored fitted model.
    """
    parameters = gauge_discharge_model["parameters"]

    return hybrid_fit(
        q_inflow,
        parameters["a"],
        parameters["b"],
        parameters["c"],
    )


def plot_gauge_discharge_model(
    gauge_discharge_model,
    q_plot_max=None,
):
    """
    Plot:
      - Model results
      - Hybrid predictive curve
      - 1:1 reference line

    Parameters
    ----------
    gauge_discharge_model : dict
        Output from fit_gauge_discharge_model().

    q_plot_max : float, optional
        Maximum inflow to display.
        Defaults to 1.25 × largest modeled inflow.
    """

    q_inflow = gauge_discharge_model["q_inflow"]
    q_gauge = gauge_discharge_model["q_gauge_observed"]

    a = gauge_discharge_model["parameters"]["a"]
    b = gauge_discharge_model["parameters"]["b"]
    c = gauge_discharge_model["parameters"]["c"]

    if q_plot_max is None:
        q_plot_max = q_inflow.max() * 1.25

    q_smooth = np.linspace(
        0,
        q_plot_max,
        1000,
    )

    q_predict = hybrid_fit(
        q_smooth,
        a,
        b,
        c,
    )

    plt.figure(figsize=(8,6))

    plt.scatter(
        q_inflow,
        q_gauge,
        s=60,
        color="C0",
        label="Model results",
        zorder=3,
    )

    plt.plot(
        q_smooth,
        q_predict,
        lw=2.5,
        color="C1",
        label="Hybrid predictive curve",
    )

    plt.plot(
        [0, q_plot_max],
        [0, q_plot_max],
        "k--",
        lw=1.5,
        label="1:1 line",
    )

    plt.xlabel("Upstream Inflow (cms)")
    plt.ylabel("Gauge Discharge (cms)")
    plt.title("Upstream Inflow–Gauge Discharge Relationship")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(
        f"Q_gauge = "
        f"{a:.6f} × Q_inflow^{b:.6f} "
        f"+ {c:.6f} × Q_inflow"
    )

    print(f"R² = {gauge_discharge_model['r_squared']:.6f}")

def find_equilibrium_timestep(
    df,
    column="inundated_area_change",
    threshold=1000,
    consecutive_steps=6,
):
    """
    Find the first timestep where 'consecutive_steps' consecutive values
    in the specified column are below the threshold.

    Returns
    -------
    float or None
        The timestep corresponding to the last timestep in the qualifying
        sequence, or None if not found.
    """
    count = 0

    for _, row in df.iterrows():
        if row[column] < threshold:
            count += 1
            if count == consecutive_steps:
                return row["timestep"]
        else:
            count = 0

    return None





def get_flow_file_dict(folder_path, extension):
    """
    Returns a dictionary mapping flow values (parsed from filenames)
    to the corresponding file paths.

    Parameters
    ----------
    folder_path : str or Path
        Folder containing the files.

    extension : str
        File extension to search for (e.g. ".ows", ".oqc", ".xys").

    Returns
    -------
    dict
        Keys are flow values (float if needed, otherwise int).
        Values are Path objects.
    """

    folder_path = Path(folder_path)

    file_dict = {}

    for file in folder_path.glob(f"*{extension}"):
    
        print(file.name)
    
        match = re.search(r"-flow-cfs-([-+]?\d*\.?\d+)", file.stem)
    
        print(match)
    
        if match:
            flow = float(match.group(1))
    
            if flow.is_integer():
                flow = int(flow)
    
            file_dict[flow] = file
    return dict(sorted(file_dict.items()))

def rating_curve_equation(Q, a, b):
    """
    Rating curve forced through (0, 0):

        stage = a * discharge**b
    """
    Q = np.asarray(Q, dtype=float)
    return a * Q**b


def fit_rating_curve(rating_curve_dict):
    """
    Fit all stage-discharge points using ordinary nonlinear
    least squares. No outlier down-weighting or removal.
    """
    sorted_keys = sorted(rating_curve_dict)

    discharge = np.array(
        [
            rating_curve_dict[k]["discharge_cms"]
            for k in sorted_keys
        ],
        dtype=float,
    )

    stage = np.array(
        [
            rating_curve_dict[k]["stage_m"]
            for k in sorted_keys
        ],
        dtype=float,
    )

    parameters, covariance = curve_fit(
        rating_curve_equation,
        discharge,
        stage,
        p0=[0.1, 0.5],
        bounds=(
            [0.0, 0.0],
            [np.inf, 2.0],
        ),
        maxfev=100000,
    )

    a, b = parameters

    stage_fit = rating_curve_equation(
        discharge,
        a,
        b,
    )

    residuals = stage - stage_fit

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((stage - stage.mean())**2)

    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(residuals**2))

    return {
        "parameters": {
            "a": float(a),
            "b": float(b),
        },
        "equation": f"h = {a:.6f} * Q^{b:.6f}",
        "r_squared": float(r_squared),
        "rmse": float(rmse),
        "covariance": covariance,
        "discharge": discharge,
        "stage": stage,
        "stage_fit": stage_fit,
        "residuals": residuals,
    }



def shifted_rating_curve(Q, a, b, delta_Q, delta_h):
    """
    Shift the original rating curve horizontally and vertically.

    Q       : discharge in cms
    delta_Q : horizontal shift in cms
    delta_h : vertical shift in meters
    """
    Q = np.asarray(Q, dtype=float)

    shifted_Q = Q + delta_Q

    # Prevent invalid negative discharge inside the power function
    if np.any(shifted_Q < 0):
        return np.full_like(Q, np.nan, dtype=float)

    return a * shifted_Q**b + delta_h


def get_flow_file_dict(folder_path, extension):
    """
    Returns a dictionary mapping flow values from the corresponding
    TSF files to the requested file paths.

    The TSF file has the same filename as the requested file,
    except its extension is ".tsf".

    The flow value is read from the second column of the TSF file.
    """

    folder_path = Path(folder_path)

    file_dict = {}

    for file in folder_path.glob(f"*{extension}"):

        print(file.name)

        # Same filename, but with .tsf extension
        tsf_file = file.with_suffix(".tsf")

        if not tsf_file.exists():
            print(f"TSF file not found: {tsf_file.name}")
            continue

        # Read first data line after TSF header
        with open(tsf_file, "r") as f:
            f.readline()  # skip header

            first_data_line = f.readline().split()

        # Flow is the right column
        flow = float(first_data_line[1])

        if flow.is_integer():
            flow = int(flow)

        print("Flow from TSF:", flow)

        file_dict[flow] = file

    return dict(sorted(file_dict.items()))

def rating_curve_equation(Q, a, b):
    """
    Rating curve forced through (0, 0):

        stage = a * discharge**b
    """
    Q = np.asarray(Q, dtype=float)
    return a * Q**b


def fit_rating_curve(rating_curve_dict):
    """
    Fit all stage-discharge points using ordinary nonlinear
    least squares.

    The curve is anchored at the stage corresponding to the
    minimum observed discharge.
    """
    sorted_keys = sorted(rating_curve_dict)

    discharge = np.array(
        [
            rating_curve_dict[k]["discharge_cms"]
            for k in sorted_keys
        ],
        dtype=float,
    )

    stage = np.array(
        [
            rating_curve_dict[k]["stage_m"]
            for k in sorted_keys
        ],
        dtype=float,
    )

    # Anchor curve at the minimum-discharge observation
    minimum_index = np.argmin(discharge)

    Q0 = discharge[minimum_index]
    h0 = stage[minimum_index]

    def equation_for_fitting(Q, a, b):
        return rating_curve_equation(
            Q,
            a,
            b,
            Q0,
            h0,
        )

    parameters, covariance = curve_fit(
        rating_curve_equation,
        discharge,
        stage,
        p0=[0.1, 0.5],
        bounds=(
            [0.0, 0.0],
            [np.inf, 2.0],
        ),
        maxfev=100000,
    )
    
    a, b = parameters
        


    stage_fit = rating_curve_equation(
        discharge,
        a,
        b,
    )
    residuals = stage - stage_fit

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum(
        (stage - stage.mean())**2
    )

    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt(
        np.mean(residuals**2)
    )

    return {
        "parameters": {
            "a": float(a),
            "b": float(b),
            "Q0": float(Q0),
            "h0": float(h0),
        },
        "equation": (
            f"h = {h0:.6f} + "
            f"{a:.6f} * max(Q - {Q0:.6f}, 0)^{b:.6f}"
        ),
        "r_squared": float(r_squared),
        "rmse": float(rmse),
        "covariance": covariance,
        "discharge": discharge,
        "stage": stage,
        "stage_fit": stage_fit,
        "residuals": residuals,
    }


def shifted_rating_curve(Q, a, b, delta_Q, delta_h):
    """
    Shift the original rating curve horizontally and vertically.

    Q       : discharge in cms
    delta_Q : horizontal shift in cms
    delta_h : vertical shift in meters
    """
    Q = np.asarray(Q, dtype=float)

    shifted_Q = Q + delta_Q

    # Prevent invalid negative discharge inside the power function
    if np.any(shifted_Q < 0):
        return np.full_like(Q, np.nan, dtype=float)

    return a * shifted_Q**b + delta_h

def read_GSSHA_ows(folder_path, filename):
    """
    Read a GSSHA .ows file.

    Parameters
    ----------
    folder_path : str or Path
        Folder containing the .ows file.
    filename : str
        Name of the .ows file.

    Returns
    -------
    pandas.DataFrame
        Columns:
            timestep_min
            WSE_m
    """
    filepath = folder_path / filename

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=[
            "timestep_min",
            "WSE_m",
        ],
    )

    return df

import numpy as np
from scipy.optimize import least_squares


def hybrid_fit(x, a, b, c):
    x = np.asarray(x, dtype=float)
    return a * x**b + c * x


def fit_robust_hybrid_curve(
    data_dict,
    loss="soft_l1",
    f_scale=0.25,
):
    """
    Robustly fit:

        y = a*x**b + c*x

    Large residuals receive less influence than they would under
    ordinary least squares.

    Parameters
    ----------
    data_dict : dict
        Keys are x-values and values are y-values.

    loss : str
        Robust loss used by scipy.optimize.least_squares.
        Good options: "soft_l1", "huber", or "cauchy".

    f_scale : float
        Residual size at which down-weighting begins, in y-units.
        For WSE measured in meters, 0.25 means residuals larger than
        roughly 0.25 m begin receiving reduced influence.

    Returns
    -------
    dict
        Stored robust model, parameters, predictions, residuals,
        effective weights, R², and RMSE.
    """
    x = np.array(sorted(data_dict.keys()), dtype=float)
    y = np.array([data_dict[key] for key in sorted(data_dict)], dtype=float)

    def residuals(parameters):
        a, b, c = parameters
        return hybrid_fit(x, a, b, c) - y

    result = least_squares(
        residuals,
        x0=[1.0, 0.3, 0.001],
        bounds=(
            [0.0, 0.01, 0.0],
            [np.inf, 0.99, np.inf],
        ),
        loss=loss,
        f_scale=f_scale,
        max_nfev=100000,
    )

    a, b, c = result.x
    y_predicted = hybrid_fit(x, a, b, c)
    residual = y - y_predicted

    ss_res = np.sum(residual**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(residual**2))

    # Approximate effective weights for soft-L1.
    # Smaller values identify points receiving less influence.
    if loss == "soft_l1":
        effective_weights = 1 / np.sqrt(
            1 + (residual / f_scale)**2
        )
    else:
        effective_weights = np.full_like(residual, np.nan)

    return {
        "model_type": "robust_hybrid_power_linear",
        "parameters": {
            "a": float(a),
            "b": float(b),
            "c": float(c),
        },
        "equation": (
            f"y = {a:.8f} * x^{b:.8f} + "
            f"{c:.8f} * x"
        ),
        "loss": loss,
        "f_scale": float(f_scale),
        "x": x,
        "y_observed": y,
        "y_predicted": y_predicted,
        "residuals": residual,
        "effective_weights": effective_weights,
        "r_squared": float(r_squared),
        "rmse": float(rmse),
        "optimizer_result": result,
    }


def invert_hybrid_safe(y_target, a, b, c):
    """
    Invert:
        y = a*x^b + c*x

    Returns x for a given y.
    """

    def residual(x):
        return hybrid_fit(x, a, b, c) - y_target

    lo = 0.0
    hi = 1.0

    # Expand upper bound until the target is bracketed
    while residual(hi) < 0:
        hi *= 2

        if hi > 1e7:
            return np.nan

    return brentq(residual, lo, hi)

def get_flow_file_dict(folder_path, extension):
    """
    Returns a dictionary where:
        key   = flow in cms (rounded to nearest whole number)
        value = Path to the file
    """

    folder_path = Path(folder_path)

    file_dict = {}

    for file in folder_path.glob(f"*{extension}"):

        # Extract flow from filename (stored in cfs)
        flow_cfs = float(file.stem.split("-")[-1])

        # Convert to cms and round to nearest whole number
        flow_cms = round(flow_cfs * 0.028316846592)

        file_dict[flow_cms] = file

    return dict(sorted(file_dict.items()))

import numpy as np
from scipy.optimize import least_squares


def power_log_fit(x, a, b, c):
    """
    Power + logarithmic curve.

    Equation
    --------
    y = a * x**b + c * ln(1 + x)
    """
    x = np.asarray(x, dtype=float)

    return (
        a * x**b
        + c * np.log1p(x)
    )

# Define the drop-in replacement and run it on the file we just inspected.
def read_ascii_header_and_rows_cols(ascii_reference_path: str, *, nodata_value: float = -9999, tol: float = 1e-6):
    """
    Reads a GSSHA-style header (north/south/east/west/rows/cols) and returns:
      - asc_header: ESRI ASCII header lines
      - rows: int
      - cols: int
      - cellsize: float
    """
    vals = {}
    with open(ascii_reference_path, "r", errors="ignore") as f:
        for _ in range(6):
            line = f.readline()
            if not line:
                break
            if ":" in line:
                k, v = line.split(":", 1)
                key = k.strip().lower()
                val = v.strip()
                if key in ("rows", "cols"):
                    vals[key] = int(float(val))
                else:
                    vals[key] = float(val)

    required = {"north", "south", "east", "west", "rows", "cols"}
    if not required.issubset(vals):
        missing = required - set(vals)
        raise ValueError(f"GSSHA header missing keys: {', '.join(sorted(missing))}")

    north, south = vals["north"], vals["south"]
    east,  west  = vals["east"],  vals["west"]
    rows,  cols  = vals["rows"],  vals["cols"]

    # Compute cellsize and ensure square cells
    cellsize_x = (east  - west ) / cols
    cellsize_y = (north - south) / rows
    if abs(cellsize_x - cellsize_y) > tol:
        raise ValueError(f"Non-square cells detected: cellsize_x={cellsize_x} vs cellsize_y={cellsize_y}")
    cellsize = cellsize_x

    # Build ESRI ASCII header
    asc_header = [
        f"NCOLS {cols}",
        f"NROWS {rows}",
        f"XLLCORNER {west}",
        f"YLLCORNER {south}",
        f"CELLSIZE {cellsize}",
        f"NODATA_VALUE {nodata_value}"
    ]

    return asc_header, rows, cols




def _parse_gfl_values_strict(gfl_path: str) -> np.ndarray:
    """
    Parses numeric values from a GSSHA gfl file strictly:
    - Finds the line beginning with 'TS '
    - Reads subsequent lines until an END marker (ENDSCL/ENDDS/END) or EOF
    - Returns all numeric tokens (floats) in order
    """
    with open(gfl_path, "r", errors="ignore") as f:
        lines = f.read().splitlines()
    # find TS line index
    ts_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith("TS "):
            ts_idx = i
            break
    if ts_idx is None:
        raise ValueError("TS line not found in GFL file.")

    nums = []
    num_re = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$')
    for ln in lines[ts_idx + 1:]:
        u = ln.strip().upper()
        if u in ("ENDSCL", "ENDDS", "END"):
            break
        # collect numeric tokens from the line
        for tok in ln.split():
            if num_re.fullmatch(tok):
                nums.append(float(tok))
    return np.array(nums, dtype=float)


def convert_gfl_ASCII(gfl_path: str, ascii_header_path: str, out_ascii_path: str) -> str:
    """
    Convert a GSSHA .gfl to an ASCII grid file that exactly matches the
    header format (including ordering and line endings) of a provided ASCII file.

    Args:
        gfl_path (str): Path to source .gfl file.
        ascii_header_path (str): Path to an existing ASCII grid file whose
                                 first 6 header lines (north/south/east/west/rows/cols)
                                 will be copied verbatim, and from which rows/cols are read.
        out_ascii_path (str): Path where the converted ASCII grid will be written.

    Returns:
        str: The path to the written ASCII grid file.
    """
    # 1) Read header & dimensions from the reference ASCII
    header_lines, rows, cols = read_ascii_header_and_rows_cols(ascii_header_path)
    n = rows * cols

    # 2) Parse numeric values from the GFL
    values = _parse_gfl_values_strict(gfl_path)

    # Many GFL files contain multiple blocks of n values after TS.
    # Based on inspection, the second block matched the target ASCII file.
    if values.size < 2 * n:
        raise ValueError(
            f"GFL has {values.size} numeric values; need at least {2*n} values "
            f"to extract the second block of size rows*cols."
        )

    grid_vals = values[n:2 * n].reshape((rows, cols))

    # 3) Write output:
    #    - copy the first 6 header lines verbatim from ascii_header_path
    #    - data lines with '%.6f' values separated by a single space
    #    - trailing space at end of each data line
    #    - CRLF ('\\r\\n') line endings to match the reference
    with open(out_ascii_path, "wb") as f:
        for i in range(6):
            f.write((header_lines[i] + "\r\n").encode("utf-8"))
        for r in range(rows):
            line = " ".join(f"{v:.6f}" for v in grid_vals[r]) + " "
            f.write((line + "\r\n").encode("utf-8"))

    return out_ascii_path
    
def get_input_flows_for_gauge_stages(
    modeling_results_directory,
    requested_WSE,
    plot=True,
):
    
    #read model gauge output from GSSHA
    ows_file_paths = get_flow_file_dict(
        modeling_results_directory,
        ".ows"
    )
    ows_file_read = {}
    for flow_cms, filepath in ows_file_paths.items():
        ows = read_GSSHA_ows(filepath.parent, filepath.name)
    
        ows_file_read[flow_cms] = ows
    
    peak_wse = {}
    for flow_cms, df in ows_file_read.items():
        peak_wse[flow_cms] = df["WSE_m"].tail(120).mean()
    
    
    
    def power_log_fit(q, a, b, c):
        q = np.asarray(q, dtype=float)
    
        return (
            a * q**b
            + c * np.log1p(q)
        )

    
    # Convert dictionary to arrays
    inflow = np.array(
        sorted(peak_wse.keys()),
        dtype=float,
    )
    
    wse = np.array(
        [peak_wse[q] for q in inflow],
        dtype=float,
    )
    
    # Fit
    params, covariance = curve_fit(
        power_log_fit,
        inflow,
        wse,
        p0=(2.0, 0.5, 0.2),
        bounds=(
            [0.0, 0.01, 0.0],
            [np.inf, 0.99, np.inf],
        ),
        maxfev=50000,
    )
    
    a, b, c = params
    
    # Predictions
    wse_predicted = power_log_fit(
        inflow,
        a,
        b,
        c,
    )
    
    # Statistics
    residuals = wse - wse_predicted
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((wse - wse.mean())**2)
    
    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(residuals**2))
    
    def invert_power_log_safe(
        target_wse,
        a,
        b,
        c,
        q_min=0.0,
        q_max=100000.0,
    ):
        """
        Solve for Q in:
    
            target_wse = a * Q**b + c * ln(1 + Q)
        """
    
        def equation(q):
            return (
                a * q**b
                + c * np.log1p(q)
                - target_wse
            )
    
        if equation(q_min) > 0:
            return q_min
    
        while equation(q_max) < 0:
            q_max *= 2
    
            if q_max > 1e9:
                raise ValueError(
                    f"Could not bracket a solution for WSE={target_wse}"
                )
    
        return brentq(
            equation,
            q_min,
            q_max,
        )
    
    result = [
        invert_power_log_safe(
            target_wse,
            a,
            b,
            c,
        )
        for target_wse in requested_WSE
    ]

    
    if plot:
        q_plot = np.linspace(0, inflow.max() * 1.25, 1000)
    
        plt.figure(figsize=(10, 6))
    
        plt.scatter(
            inflow,
            wse,
            s=60,
            label="Observed",
        )
    
        plt.plot(
            q_plot,
            power_log_fit(q_plot, a, b, c),
            linewidth=2.5,
            label="Power + logarithmic fit",
        )
    
        plt.xlabel("Upstream inflow (cms)")
        plt.ylabel("WSE (ft)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result

def view_timestep_convergence(model_results_directory, area_change_threshold = 1000, number_timesteps_below_threshold = 6):
    dep_file_paths = get_flow_file_dict(
        model_results_directory,
        ".dep"
    )
    dep_file_read ={}
    for flow_cms, filepath in dep_file_paths.items():
        dep = read_GSSHA_dep(filepath.parent, filepath.name)
        
        dep_file_read[flow_cms] = dep
    
    dep_changes = {}
    for flow_cms, dep_data in dep_file_read.items():
        dep_changes[flow_cms] = calculate_dep_changes(
            dep_data,
            cell_size=10,
            depth_threshold=0.01
        )
    
    equilibrium_timesteps = {
        flow_cms: find_equilibrium_timestep(
        df,
        column="inundated_area_change",
        threshold=area_change_threshold,
        consecutive_steps=number_timesteps_below_threshold,)
        for flow_cms, df in dep_changes.items()
    }
    
    
    # Convert equilibrium_timesteps dictionary to arrays
    flow = np.array(
        sorted(equilibrium_timesteps.keys()),
        dtype=float,
    )
    
    time = np.array(
        [equilibrium_timesteps[f] for f in flow],
        dtype=float,
    )
    
    
    # Remove missing values
    valid = np.isfinite(flow) & np.isfinite(time)
    
    flow_fit = flow[valid]
    time_fit = time[valid]
    
    
    
    plt.figure(figsize=(9, 4))
    
    plt.scatter(
        flow_fit,
        time_fit,
        s=60,
        zorder=3,
        label="Model runs",
    )
    

    plt.xlabel("Flow (cms)")
    plt.ylabel("Equilibrium timestep (min)")
    plt.title("Predicted Equilibrium Timestep by Inflow")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def predict_timestep_convergence(
    input_flows,
    medium_flow_threshold,
    high_flow_threshold,
    low_flow_timing,
    medium_flow_timing,
    high_flow_timing,
):
    equilibrium_times = []

    for flow in input_flows:
        flow = float(flow)

        if flow < medium_flow_threshold:
            equilibrium_times.append(low_flow_timing)

        elif flow < high_flow_threshold:
            equilibrium_times.append(medium_flow_timing)

        else:
            equilibrium_times.append(high_flow_timing)

    return equilibrium_times
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
# In[ ]: