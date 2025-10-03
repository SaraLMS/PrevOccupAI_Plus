# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from load.raw_data_loader import load_daily_acquisitions
from signal_processing.pre_process import apply_pre_processing_pipeline
from HAR.synchonise_predictions import classify_synchronise_predictions

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
LOAD_DAILY_ACQUISITIONS = True
SELECTED_SENSORS = {'phone': ['ACC', 'GYR', 'MAG', 'ROT', 'NOISE'],
                    'watch': ['ACC', 'GYR', 'MAG']}
DAILY_FOLDER_PATH = "D:\\Backup PrevOccupAI data\\jan2023\\data\\group2\\sensors\\LIBPhys #003\\2022-06-21"
W_SIZE = 5.0
FS = 100

# ------------------------------------------------------------------------------------------------------------------- #
# program starts here
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    if LOAD_DAILY_ACQUISITIONS:

        # load all acquisitions from the same day into a nested dictionary
        df_dict = load_daily_acquisitions(DAILY_FOLDER_PATH, SELECTED_SENSORS)

        # pre-process data
        processed_df_dict = apply_pre_processing_pipeline(df_dict, fs_android=FS, downsample_muscleban=True)

        # classify and synchronise predictions
        sync_df = classify_synchronise_predictions(processed_df_dict, w_size=W_SIZE, fs=FS)


