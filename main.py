# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from load.raw_data_loader import load_daily_acquisitions
from signal_processing.pre_process import apply_pre_processing_pipeline
# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
LOAD_DAILY_ACQUISITIONS = True
SELECTED_SENSORS = {'phone': ['ACC', 'GYR', 'MAG', 'ROT', 'NOISE'],
                    'watch': ['ACC', 'GYR', 'MAG','ROT', 'HEART'],
                    'mban': ['ACC', 'EMG']}
DAILY_FOLDER_PATH = "D:\\Backup PrevOccupAI data\\jan2023\\data\\group1\\sensors\\LIBPhys #005\\2022-05-02"

# ------------------------------------------------------------------------------------------------------------------- #
# program starts here
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    if LOAD_DAILY_ACQUISITIONS:

        # load all acquisitions from the same day into a nested dictionary
        df_dict = load_daily_acquisitions(DAILY_FOLDER_PATH, SELECTED_SENSORS)

        processed_df_dict = apply_pre_processing_pipeline(df_dict, fs_android=100)