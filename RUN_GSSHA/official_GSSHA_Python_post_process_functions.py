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


# In[ ]: