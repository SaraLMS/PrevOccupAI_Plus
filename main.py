from load.path_handler import get_sensor_paths_per_device

selected_sensors = {'phone': ['ACC', 'GYR', 'MAG'],
                    'watch': ['ACC', 'HR'],
                    'mban': ['ACC']
}

folder_path = "D:\\Backup PrevOccupAI data\\jan2023\\data\\group3\\sensors\\LIBPhys #003\\2022-07-06"

paths_dict = get_sensor_paths_per_device(folder_path, selected_sensors)