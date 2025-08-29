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
                    'watch': ['ACC', 'GYR', 'MAG','ROT', 'HR'],
                    'mban': ['ACC', 'EMG']
}

# SUBJECT_FOLDER_PATH = "D:\\Backup PrevOccupAI data\\jan2023\\data\\group3\\sensors\\LIBPhys #003\\2022-07-06"
FOLDER_NAME='2022-07-06'
FOLDER_PATH = "D:\\Backup PrevOccupAI data\\jan2023\\data\\group3\\sensors\\LIBPhys #003"


# ------------------------------------------------------------------------------------------------------------------- #
# program starts here
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    for folder in os.listdir(FOLDER_PATH):

        path = os.path.join(FOLDER_PATH, folder)

        print(f"loading data from: {path}")
        paths_dict = load_daily_acquisitions(path, SELECTED_SENSORS)