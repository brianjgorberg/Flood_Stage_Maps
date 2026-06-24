#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


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


def cleanup_model_dir(MODEL_DIR):
    # Remove copied EXEs and DLLs
    for file in os.listdir(MODEL_DIR):
        if file.endswith(('.exe', '.dll', '.cfg', '.manifest')):
            os.remove(os.path.join (MODEL_DIR, file))
    print("✔️ Cleaned up model folder.")


#GSSHA auto shut down
def GSSHA_auto_shutdown():
    gssha_exe = Path.cwd() / "Model" / "gssha.exe"
    process = subprocess.Popen([gssha_exe, "Test_Waiahole_model.prj"], cwd=Path.cwd() / "Model")


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


def replace_value_in_gssha_file(read_dir, gssha_sample_file, save_filename, new_value):
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
    updated_content = content.replace("XX", str(new_value))

    # Full path for the output file (same directory)
    save_path = os.path.join(read_dir, save_filename)

    # Write updated content to the new file
    with open(save_path, 'w') as file:
        file.write(updated_content)

    return save_path

# In[ ]:




