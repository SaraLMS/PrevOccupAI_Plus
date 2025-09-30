# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from load.raw_data_loader import load_daily_acquisitions
from signal_processing.pre_process import apply_pre_processing_pipeline
from HAR.classifier import classify_human_activities

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
LOAD_DAILY_ACQUISITIONS = True
SELECTED_SENSORS = {'phone': ['ACC', 'GYR', 'MAG', 'ROT', 'NOISE'],
                    'mban': ['ACC', 'EMG']}
DAILY_FOLDER_PATH = "D:\\Backup PrevOccupAI data\\jan2023\\data\\group1\\sensors\\LIBPhys #005\\2022-05-02"
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
        processed_df_dict = apply_pre_processing_pipeline(df_dict, fs_android=100)

        # classify phone data
        activities_dict = classify_human_activities(df_dict['phone'], w_size=W_SIZE, fs=FS)