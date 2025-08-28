# ------------------------------------------------------------------------------------------------------------------- #
# input constants
# ------------------------------------------------------------------------------------------------------------------- #
PHONE = 'phone'
WATCH = 'watch'
MBAN = 'mban'

# ------------------------------------------------------------------------------------------------------------------- #
# device constants
# ------------------------------------------------------------------------------------------------------------------- #
ACQUISITION_PATTERN = r"\d{2}-\d{2}-\d{2}" # hh-mm-ss
MAC_ADDRESS_PATTERN = r'[A-F0-9]{12}'
MBAN_LEFT = 'mBAN_left'
MBAN_RIGHT = 'mBAN_right'

# ------------------------------------------------------------------------------------------------------------------- #
# sensor constants
# ------------------------------------------------------------------------------------------------------------------- #
# definition of valid sensors
ACC = 'ACC'
GYR = 'GYR'
MAG = 'MAG'
ROT = 'ROT'
NOISE = 'NOISE'
HEART = 'HR'
EMG = 'EMG'

# define valid sensors for the three devices
PHONE_SENSORS = [ACC, GYR, MAG, ROT, NOISE]
WATCH_SENSORS = [ACC, GYR, MAG, ROT, HEART]
MBAN_SENSORS = [ACC, EMG]

# mapping of valid sensors to sensor filename - for both watch and phone
SENSOR_MAP = {ACC: 'ACCELEROMETER',
              GYR: 'GYROSCOPE',
              MAG: 'MAGNET', # To find both 'MAGNETIC FIELD' and 'MAGNETOMETER'
              ROT: 'ROTATION_VECTOR',
              NOISE: 'NOISERECORDER',
              HEART: 'HEART_RATE'}
