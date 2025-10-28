# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import load_signals
import signal_processing
import HAR

import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
LOAD_DAILY_ACQUISITIONS = True
SELECTED_SENSORS = {'phone': ['ACC', 'GYR', 'MAG'],
                    'watch': ['ACC']}
DAILY_FOLDER_PATH = "D:\\Backup PrevOccupAI data\\jan2023\\data\\group2\\sensors\\LIBPhys #003\\2022-06-21"
W_SIZE = 5.0
FS = 100

# ------------------------------------------------------------------------------------------------------------------- #
# program starts here
# ------------------------------------------------------------------------------------------------------------------- #

def main(classify_and_sync=True):

    if classify_and_sync:

        # load_signals all acquisitions from the same day into a nested dictionary
        df_dict = load_signals.load_daily_acquisitions(DAILY_FOLDER_PATH, SELECTED_SENSORS)

        # pre-process data
        processed_df_dict = signal_processing.apply_pre_processing_pipeline(df_dict, fs_android=FS, downsample_muscleban=True)

        # classify and synchronise predictions
        sync_df = HAR.classify_and_synchronise_predictions(processed_df_dict, w_size=W_SIZE, fs=FS)

        # print(sync_df.columns)
        # # Line plot
        # plt.figure(figsize=(8, 5))
        # plt.plot(sync_df['x_ACC'].iloc[::10], label='x_ACC')
        # plt.plot(sync_df['x_ACC_WEAR'].iloc[::10], label='x_ACC_WEAR')
        # plt.xlabel('Index')
        # plt.ylabel('Values')
        # plt.title('x_ACC vs x_ACC_WEAR')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

if __name__ == '__main__':

    main()




