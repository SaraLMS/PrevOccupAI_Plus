# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from load.path_handler import get_sensor_paths_per_device
from load.raw_data_loader import load_daily_acquisitions
import os
# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #

SELECTED_SENSORS = {'phone': ['ACC', 'GYR', 'MAG', 'ROT', 'NOISE'],
                    'watch': ['ACC', 'GYR', 'MAG','ROT', 'HEART'],
                    'mban': ['ACC', 'EMG']
}

SUBJECT_FOLDER_PATH = "D:\\Backup PrevOccupAI data\\jan2023\\data\\group1\\sensors\\LIBPhys #005\\2022-05-02"
FOLDER_NAME='2022-07-04'
FOLDER_PATH = "D:\\Backup PrevOccupAI data\\jan2023\\data\\group1\\sensors"


# ------------------------------------------------------------------------------------------------------------------- #
# program starts here
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    # for subject_folder in os.listdir(FOLDER_PATH):
    #
    #     print(f"loading data from subject: {subject_folder}")
    #
    #     for folder in os.listdir(os.path.join(FOLDER_PATH, subject_folder)):
    #
    #         path = os.path.join(FOLDER_PATH, subject_folder, folder)
    #
    #         print(f"loading data from: {path}")
    #         paths_dict = load_daily_acquisitions(path, SELECTED_SENSORS)

    paths_dict = load_daily_acquisitions(SUBJECT_FOLDER_PATH, SELECTED_SENSORS)