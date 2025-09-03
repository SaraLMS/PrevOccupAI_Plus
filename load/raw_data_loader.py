# -------------------------------------------------------------------------------------------------------------------- #
# imports
# -------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
from tqdm import tqdm

# internal imports
from constants import PHONE, WATCH, VALID_MBAN_DATA, NSEQ, IMU_SENSORS, TIME_COLUMN_NAME, ROT, NOISE, HEART, MBAN
from .path_handler import get_sensor_paths_per_device
from .parser import extract_sensor_from_filename
from .interpolate import cubic_spline_interpolation, slerp_interpolation, zero_order_hold_interpolation, \
    interpolate_heart_rate_sensor
# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #

# padding types
PADDING_SAME = 'same'
PADDING_ZERO = 'zero'
VALID_PADDING_TYPES = ['same', 'zero']

# sensor data dictionary keys
LOADED_SENSORS = 'loaded sensors'
STARTING_TIMES = 'starting times'
STOPPING_TIMES = 'stopping times'

# -------------------------------------------------------------------------------------------------------------------- #
# public functions
# -------------------------------------------------------------------------------------------------------------------- #
def load_daily_acquisitions(folder_path: str, load_devices: Dict[str, List[str]], fs_android: int = 100,
                            padding_type: str = PADDING_SAME) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load sensor data of an entire day.

    This function loads sensor data, defined in load_sensors, that is inside folder_path. This function assumes that
    folder_path pertains to the date of the acquisition and that inside there are subfolders regarding the scheduled
    acquisition times, with the correspondent sensor files. This function loads all the data from the devices and sensors
    defined in load_sensors, into a nested dictionary.
    :param folder_path:
    :param load_devices:
    :param fs_android:
    :param padding_type:
    :return:
    """
    # innit dictionary to hold the dataframes
    dataframes_dict: Dict[str, Dict[str, pd.DataFrame]] = {}

    # get paths for all loaded devices/sensors sorted by device and acquisition time
    paths_dict = get_sensor_paths_per_device(folder_path, load_devices)

    # if all nested dictionaries are empty
    if paths_dict and not all(not v for v in paths_dict.values()):

        # cycle over the devices in the dictionary
        for device, acquisitions_dic in paths_dict.items():

            # inform user
            print(f"\nLoading data from device: {device}.")

            # add entry to the results dictionary
            if device not in dataframes_dict:
                dataframes_dict[device] = {}

            # loop over the acquisition times keys
            for acquisition_time, paths_list in acquisitions_dic.items():

                # if the device is a muscleban the loading is handled differently
                if device != PHONE and device != WATCH:

                    # get sensors to be loaded for the mban - to get only the chosen sensors
                    sensor_list_mban = load_devices[MBAN]

                    # muscleBAN only has one file per acquisition
                    # load muscleBAN data - only the sensors defined in load_devices
                    muscleban_sensor_data = _load_muscleban_data(paths_list[0], sensor_list_mban)

                    # add to dictionary
                    dataframes_dict[device][acquisition_time] = muscleban_sensor_data

                # if its android device
                else:

                    # load the data
                    sensor_data, report = _load_raw_data(paths_list)

                    # align the data
                    # (1) pad the data (all sensors start and stop at the same timestep)
                    padded_data = _pad_data(sensor_data, report, padding_type)

                    # (2) resample the data to 100 Hz
                    interpolated_data = _re_sample_data(padded_data, report, fs=fs_android)

                    # (3) create a DataFrame containing all the data
                    aligned_sensor_df = pd.concat(interpolated_data, axis=1)
                    aligned_sensor_df = aligned_sensor_df.sort_index()

                    # add to dictionary
                    dataframes_dict[device][acquisition_time] = aligned_sensor_df
    else:
        print(f"\nWarning: No data was found in {folder_path}. This function will return an empty dictionary.")

    # inform user
    _create_loading_report(load_devices, dataframes_dict)

    return dataframes_dict

# -------------------------------------------------------------------------------------------------------------------- #
# private functions
# -------------------------------------------------------------------------------------------------------------------- #

def _load_raw_data(sensor_paths_list: List[Path]) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    """
    Loads sensor data contained in 'folder_path' into a list of pandas DataFrames. Each element in the list corresponds
    to a sensor's data.A dictionary is also returned containing the loaded sensors and the timestamps when each sensor
    started and stopped recording.

    General data cleaning includes:
    (1) Removal of NaN values
    (2) Removal of duplicates
    (3) Resetting of DataFrame index

    :param sensor_paths_list: List with the signal paths (pathlib.Path) to be loaded
    :return: A tuple where the first element is a list of pandas DataFrames for each sensor's data, and the second
             element is a dictionary containing sensor start/stop timestamps and order information.
    """

    # list for holding the loaded DataFrames
    sensor_data = []

    # list for holding the sensor names
    loaded_sensors = []

    # list for holding starting and stopping timestamps
    start_times = []
    stop_times = []

    # cycle over the sensor names
    for sensor_path in tqdm(sensor_paths_list, desc="--> Loading data"):

        # get sensor name from file
        sensor_name = extract_sensor_from_filename(sensor_path.name)

        if sensor_name:

            # load the data
            sensor_df = _load_sensor_file(sensor_path, sensor_name)

            # append the data to sensor_data
            sensor_data.append(sensor_df)

            # append the sensor to loaded_sensors
            loaded_sensors.append(sensor_name)

            # append the start and stop times
            start_times.append(sensor_df[TIME_COLUMN_NAME].iloc[0])
            stop_times.append(sensor_df[TIME_COLUMN_NAME].iloc[-1])

        else:
            print(f"Warning: No file found for sensor {sensor_name}. Skipping this sensor.")

    # create dictionary
    report = {
        LOADED_SENSORS: loaded_sensors,
        STARTING_TIMES: start_times,
        STOPPING_TIMES: stop_times,
    }

    return sensor_data, report


def _load_sensor_file(file_path: Path, sensor_name: str) -> pd.DataFrame:
    """
    Load a sensor file into a pandas DataFrame and cleans it.

    This function reads a sensor data file located in the specified folder, performs initial cleanup
    by removing unnecessary columns, and assigns appropriate column names. For rotation vector data,
    additional steps are taken to ensure that only valid unit quaternions are kept.

    :param file_path: Path of the signal to be loaded
    :param sensor_name: The name of the sensor, used to define appropriate column names and handle
                        sensor-specific preprocessing.
    :return: A cleaned pandas DataFrame containing the sensor data with appropriate column names.
    """

    # read the file
    sensor_df = pd.read_csv(file_path, delimiter='\t', header=None, skiprows=3)

    # remove nan column (the loading of the opensignals sensor file through read_csv(...) generates a nan column
    sensor_df.dropna(axis=1, how='all', inplace=True)

    # column names if it is the noise recorder or heart rate sensor
    if sensor_name == NOISE or sensor_name == HEART:

        col_names = [TIME_COLUMN_NAME, f'{sensor_name}']

    # perform extra steps for rotation vector
    elif sensor_name == ROT:

        # rotation vector from the smartwatch has an extra column (estimated heading) to be removed
        if len(sensor_df.columns) > 5:

            sensor_df = sensor_df.drop(sensor_df.columns[-1], axis=1)

        # add fourth column name
        col_names = [TIME_COLUMN_NAME, f'x_{sensor_name}', f'y_{sensor_name}', f'z_{sensor_name}', f'w_{sensor_name}']

        # remove samples that are not unit vectors
        sensor_df = _remove_non_unit_quaternion(sensor_df)

    # is imu sensor
    else:

        # define column names depending on sensor name
        col_names = [TIME_COLUMN_NAME, f'x_{sensor_name}', f'y_{sensor_name}', f'z_{sensor_name}']

    # add column names
    sensor_df.columns = col_names

    # remove nan values and duplicates + reset index
    sensor_df = _clean_df(sensor_df)

    return sensor_df


def _remove_non_unit_quaternion(rotvec_df: pd.DataFrame, tol: float = 0.5) -> pd.DataFrame:
    """
    Remove corrupted samples from a DataFrame containing Android rotation vector data.
    Android rotation vector data are expected to be unit quaternions (i.e., their norm should be close to 1).
    This function removes samples where the quaternion norm deviates from 1 beyond a given tolerance.

    :param rotvec_df: A DataFrame where the first column represents timestamps, and the remaining columns
                      contain quaternion components (x, y, z, w).
    :param tol: optional (default=0.1). The tolerance for deviation from a unit quaternion. Samples
                with a norm less than `1 - tol` are considered corrupted and removed.
    :return: The cleaned DataFrame containing only valid unit quaternions.
    """

    # get number of samples before removal
    num_samples_pre = len(rotvec_df)

    # calculate the norm of the vector
    vector_norm = np.linalg.norm(rotvec_df.iloc[:, 1:], axis=1)

    # remove samples that do not adhere to the norm (keep samples that adhere to the vector norm)
    rotvec_df = rotvec_df[vector_norm >= 1 - tol]

    # calculate the number of removed samples
    num_samples_removed = num_samples_pre - len(rotvec_df)

    if num_samples_removed > 0:
        print(f"Removed {num_samples_removed} samples that were not normal from Rotation Vector")

    return rotvec_df


def _clean_df(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs general cleaning of the data frame.
    (1) remove nan values
    (2) remove duplicates
    (3) reset index
    Parameters

    :param sensor_df: The data frame that was loaded from the sensor file.
    :return: pandas.DataFrame containing the cleaned data.
    """

    # remove any nan values and duplicates
    sensor_df = sensor_df.dropna()
    sensor_df = sensor_df.drop_duplicates(subset=[TIME_COLUMN_NAME])

    # reset the index to start at zero
    sensor_df = sensor_df.reset_index(drop=True)

    return sensor_df


def _pad_data(sensor_data: List[pd.DataFrame], report: Dict[str, Any], padding_type: str = PADDING_SAME)\
        -> List[pd.DataFrame]:
    """
    Pads the sensor data so that all sensors start and end at the same timestep. Padding is done based on the
    sensor that starts and stops the latest and earliest, respectively. Only data where all sensors are collected
    simultaneously are considered. By default, 'same' padding type is used.

    :param sensor_data: A list of pandas DataFrames containing the sensor data.
    :param report: A dictionary containing metadata such as 'STARTING_TIMES', 'STOPPING_TIMES', and 'LOADED_SENSORS'.
    :param padding_type: The padding type to use. 'same' uses the first and last valid sensor data values for padding,
                         while 'zero' uses zero padding. Default: 'same'.
    :return: A list of pandas.DataFrames containing the padded sensor data.
    """

    # list for holding the padded sensor data
    padded_data = []

    # get the index of the latest start and the earliest stopping times
    start_index = report[STARTING_TIMES].index(max(report[STARTING_TIMES]))
    stop_index = report[STOPPING_TIMES].index(min(report[STOPPING_TIMES]))

    # get the start and stop timestamps
    start_timestamp = report[STARTING_TIMES][start_index]
    end_timestamp = report[STOPPING_TIMES][stop_index]

    # get the time axis of the start and stop sensor
    time_axis_start = sensor_data[start_index][TIME_COLUMN_NAME]
    time_axis_end = sensor_data[stop_index][TIME_COLUMN_NAME]

    # loop over the sensors
    for num, sensor_name in tqdm(enumerate(report[LOADED_SENSORS]), total=len(report[LOADED_SENSORS]),
                                 desc="Padding data to ensure all data begins and ends on the same timestamp."):

        # get the data of the sensor
        sensor_df = sensor_data[num]

        # (1) padding at the beginning
        if start_timestamp > sensor_df[TIME_COLUMN_NAME].iloc[
            0]:  # start_timestamp after current signal start --> crop signal

            # crop the DataFrame
            sensor_df = sensor_df[sensor_df[TIME_COLUMN_NAME] >= start_timestamp]

        # get the timestamp values that need to be padded at the beginning of the DataFrame
        timestamps_start_pad = time_axis_start[time_axis_start < sensor_df[TIME_COLUMN_NAME].iloc[0]]

        # (2) padding at the end
        if end_timestamp < sensor_df[TIME_COLUMN_NAME].iloc[
            -1]:  # end_timestamp before current signal end --> crop signal

            # crop the time axis
            sensor_df = sensor_df[sensor_df[TIME_COLUMN_NAME] <= end_timestamp]

        # get the timestamp values that need to be padded at the end of the DataFrame
        timestamps_end_pad = time_axis_end[time_axis_end > sensor_df[TIME_COLUMN_NAME].iloc[-1]]

        if padding_type == 'same':
            # create padding for beginning and end
            padding_start = _create_padding(timestamps_start_pad, sensor_df.iloc[0, 1:].values)
            padding_end = _create_padding(timestamps_end_pad, sensor_df.iloc[-1, 1:].values)
        else:
            # create zero padding
            padding_start = _create_padding(timestamps_start_pad, np.zeros(len(sensor_df.columns) - 1))
            padding_end = _create_padding(timestamps_end_pad, np.zeros(len(sensor_df.columns) - 1))

        # get the columns of the DataFrame
        column_names = sensor_df.columns

        # create padded array
        padded_df = np.concatenate((padding_start, sensor_df.values, padding_end))

        # append the padded data
        padded_data.append(pd.DataFrame(padded_df, columns=column_names))

    return padded_data


def _create_padding(timestamps: List[Union[int, float]], values: np.ndarray):
    """
    Create padding for the given timestamps using specified values.
    This function replicates the provided `values` for each timestamp in `timestamps`,
    creating a padded array where each row consists of a timestamp followed by the repeated values.

    :param timestamps: A list of timestamp values.
    :param values: A 1D array containing the values to be repeated for each timestamp.
    :return: A 2D array where each row contains a timestamp followed by the replicated values.
    """

    # get the number of timestamps
    n_timestamps = len(timestamps)

    # tile the padding
    padding = np.tile(values, (n_timestamps, 1))

    return np.column_stack((timestamps, padding))


def _re_sample_data(sensor_data: List[pd.DataFrame], report:  Dict[str, Any], fs=100) -> List[pd.DataFrame]:
    """
    Resamples the sensor data from the smartwatch and smartphone to the specified sampling frequency.
    This function takes a list of sensor data DataFrames and resamples each sensor's data to the desired
    sampling frequency (`fs`). For IMU-based sensors (ACC, GYR, MAG), cubic spline interpolation is used,
    and for Rotation Vector data, SLERP interpolation is performed. For the noise recorder and heart rate sensor,
    zero order hold interpolation (repeat the previous value). This function also corrects possible rounding errors
    in the time column.

    :param sensor_data: A list of DataFrames, each containing sensor data. It is assumed that the first contains
                        the time axis, while the other columns contain sensor data.
    :param report: A dictionary containing metadata, including the sensor names under the key 'LOADED_SENSORS'.
    :param fs: The target sampling frequency for the resampled data. Default: 100 (Hz)
    :return: A list of DataFrames containing the resampled sensor data.
    """

    # list to hold the re-sampled data
    re_sampled_data = []

    # cycle over the sensors
    for sensor_df, sensor_name in tqdm(zip(sensor_data, report[LOADED_SENSORS]), total=len(sensor_data),
                                       desc=f"Ensuring equidistant sampling by resampling data to {fs} Hz"):

        # DataFrame for holding the interpolated data
        interpolated_sensor_df = pd.DataFrame()

        # interpolation for IMU (ACC, GYR, MAG)
        if sensor_name in IMU_SENSORS:

            # perform cubic spline interpolation
            interpolated_sensor_df = cubic_spline_interpolation(sensor_df, fs=fs)

        # interpolation for rotation vector (ROT)
        elif sensor_name == ROT:

            # perform SLERP interpolation
            interpolated_sensor_df = slerp_interpolation(sensor_df, fs=fs)

        # interpolate noise recorder (NOISE)
        elif sensor_name == NOISE:

            # perform zero order hold interpolation
            interpolated_sensor_df = zero_order_hold_interpolation(sensor_df, fs=fs)

        # interpolate heart rate sensor
        elif sensor_name == HEART:

            # zero order hold interpolation
            interpolated_sensor_df = interpolate_heart_rate_sensor(sensor_df, fs=fs)

        else:

            # This does not happen - just for code completion
            print(f"There is no interpolation implemented for the sensor you have chosen. Chosen sensor: {sensor_name}.")

        # fix rounding errors in the time column
        interpolated_sensor_df = _fix_rounding_error(interpolated_sensor_df)

        # append interpolated data to list
        re_sampled_data.append(interpolated_sensor_df)

    return re_sampled_data


def _load_muscleban_data(file_path: Path, sensor_list: List[str]) -> pd.DataFrame:
    """
    Loads MuscleBan data into a DataFrame.

    Loads only EMG or/and accelerometer (x, y, z) signals, depending on sensor_list. Removes MAG sensor as it is unreliable.

    :param file_path: pathlib.Path to the folder containing the file.
    :param sensor_list: List of str pertaining to the sensors to be loaded for the mban
    :return:  A DataFrame containing the EMG and ACC data from the muscleban
    """
    # inform user
    print(f"\nLoading muscleBAN data from file: {file_path.name}.")

    # load data into a csv file
    sensor_df = pd.read_csv(file_path, delimiter = '\t', header=None, skiprows=3)

    # remove Nan column that is generated when using pd.read_csv
    sensor_df = sensor_df.dropna(axis=1, how="all")

    # if there are 9 column then the second column is only zeros (happens in some firmware versions)
    if len(sensor_df.columns) > 8:

        # remove zero column
        sensor_df = sensor_df.drop(sensor_df.columns[1], axis=1)

    # remove MAG which are the last three channels
    sensor_df = sensor_df.drop(sensor_df.columns[-3:], axis=1)

    # add column names - nseq, emg and acc columns
    sensor_df.columns = VALID_MBAN_DATA

    # keep only the sensors in sensor list (plus nSeq)
    cols_to_keep = [col for col in sensor_df.columns
                    if any(sensor in col for sensor in sensor_list) or col == NSEQ]
    sensor_df = sensor_df[cols_to_keep]

    return sensor_df


def _create_loading_report(load_devices: Dict[str, List[str]],
                           dataframes_dict: Dict[str, Dict[str, pd.DataFrame]])-> None:
    """
    Prints a report of loaded acquisitions, showing which devices and sensors
    were successfully loaded and which are missing.

    :param load_devices: Dictionary mapping device names to lists of requested sensors.
    :param dataframes_dict: Nested dictionary with loaded data per device and acquisition time.
    :return: None
    """
    print("\n=== Acquisition Report ===")

    for device, requested_sensors in load_devices.items():
        # find all matching loaded devices (e.g. mban → mBAN_left, mBAN_right)
        matching_devices = [
            dev for dev in dataframes_dict.keys()
            if dev.lower().startswith(device.lower())
        ]

        if not matching_devices:
            print(f"\nDevice: {device}")
            print("  No data available.")
            continue

        for actual_device in matching_devices:
            print(f"\nDevice: {actual_device}")
            acquisitions = dataframes_dict[actual_device]

            if not acquisitions:
                print("  No data available.")
                continue

            for acq_time, df in acquisitions.items():
                loaded_columns = list(df.columns)

                # Match requested sensor groups against column names (x_ACC, y_GYR, etc.)
                loaded_sensors = {
                    sensor for sensor in requested_sensors
                    if any(sensor in col for col in loaded_columns)
                }
                missing_sensors = set(requested_sensors) - loaded_sensors

                print(f"  Acquisition {acq_time}:")
                print(f"    Loaded sensors: {list(loaded_sensors)}")
                if missing_sensors:
                    print(f"    ⚠ Missing sensors: {list(missing_sensors)}")


def _fix_rounding_error(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes rounding errors in the time column of the sensor DataFrame. This is done in several steps:

    (1) multiply the time column values by the rounding_factor (sampling rate * 10)

    (2) separates the time series into two: one with the values that are divisible by 10, and the other were they're not

    (3) Sum +1 to all values of the series where te values are not divisible by one

    (4) Concat this two time series back into one and add this to the dataframe as the new time column and device by 1000

    (5) set the time column as axis of the dataframe

    :param sensor_df: pd:DataFrame containing a time column expected to be numeric.
    :return: pd.DataFrame with the corrected time column replacing the original,
        sorted and adjusted for rounding errors. The time column is set as the index.
    """

    # (1) multiply time column by 1000
    time_column = sensor_df[TIME_COLUMN_NAME].multiply(ROUNDING_FACTOR).apply(math.trunc)

    # (2) separate time series into two series
    # first series has the values that are divisible by10
    div_by_10 = time_column[time_column % 10 == 0].sort_values()

    # (3) seconds series has the values that are not divisible by 10
    not_div_by_10 = time_column[time_column % 10 != 0].sort_values().add(1)

    # (4) concat the two series into one
    final_time_series = pd.concat([div_by_10, not_div_by_10]).sort_values()

    # Remove 'time' column
    sensor_df = sensor_df.drop(columns=[TIME_COLUMN_NAME])

    # Assign final_series to 'time' column and divide by the rounding factor
    sensor_df[TIME_COLUMN_NAME] = final_time_series.div(ROUNDING_FACTOR)

    # (5) set time column as axis
    sensor_df = sensor_df.set_index(TIME_COLUMN_NAME)

    return sensor_df