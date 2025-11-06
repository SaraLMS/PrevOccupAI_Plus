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

# ------------------------------------------------------------------------------------------------------------------- #
# sensor constants
# ------------------------------------------------------------------------------------------------------------------- #
# definition of valid sensors
ACC = 'ACC'
GYR = 'GYR'
MAG = 'MAG'
ROT = 'ROT'
NOISE = 'NOISE'
HEART = 'HEART'
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

IMU_SENSORS = [ACC, GYR, MAG]

ACC_PREFIX = "ACCELEROMETER"
GYR_PREFIX = "GYROSCOPE"
MAG_PREFIX = "MAGNET"
HEART_PREFIX = "HEART_RATE"
ROT_PREFIX = "ROTATION_VECTOR"
NOISE_PREFIX = "NOISERECORDER"

# the order of the sensors in these two lists must be the same
AVAILABLE_ANDROID_PREFIXES = [ACC_PREFIX, GYR_PREFIX, MAG_PREFIX, HEART_PREFIX, ROT_PREFIX, NOISE_PREFIX]
AVAILABLE_ANDROID_SENSORS = [ACC, GYR, MAG, HEART, ROT, NOISE]


# definition of time column
TIME_COLUMN_NAME = 't'
# ------------------------------------------------------------------------------------------------------------------- #
# MuscleBan constants
# ------------------------------------------------------------------------------------------------------------------- #

FS_MBAN = 1000
EMG = 'EMG'
XACC = 'xACC'
YACC = 'yACC'
ZACC = 'zACC'
NSEQ = 'nSeq'

VALID_MBAN_DATA = [NSEQ, EMG, XACC, YACC, ZACC]

MBAN_LEFT = 'mBAN_left'
MBAN_RIGHT = 'mBAN_right'

# ------------------------------------------------------------------------------------------------------------------- #
# Questionnaire constants
# ------------------------------------------------------------------------------------------------------------------- #
CONFIG_FOLDER_NAME = 'config_files'
RESULTS_FOLDER_NAME = 'results'

# ------------------------------------------------------------------------------------------------------------------- #
# Estilo de Vida constants
# ------------------------------------------------------------------------------------------------------------------- #

EV_COLUMN_NAMES_MAP = {
        'q1': 'fuma',
        'q1a': 'cigarros',
        'q1b': 'tempo',
        'q1c': 'cigarros_passado',
        'q1d': 'tempo_passado',
        'q2': 'alcool',
        'q2a': 'bebidas',
    }

EV_ANSWERS_MAP = {
        'fuma': {
            'A1': 'Sim, diariamente',
            'A2': 'Ocasionalmente',
            'A3': 'Não, mas fumou no passado',
            'A4': 'Não, nunca fumou',
            'A5': 'Não sabe/Não responde',
        },
        'alcool': {
            'A1': 'Diariamente',
            'A2': 'Ocasionalmente',
            'A3': 'Nunca',
            'A4': 'Não sabe/Não responde',
        },
        'bebidas': {
            'A1': '< 3 semana',
            'A2': '> 3 semana',
            'A3': '> 3 dia',
        },
    }


# ------------------------------------------------------------------------------------------------------------------- #
# Atividade Fisica constants
# ------------------------------------------------------------------------------------------------------------------- #


AF_OLD_COLUMNS = [
    'q1a', 'q1b[SQ001]', 'q1b[SQ002]',
    'q2a', 'q2b[SQ001]', 'q2b[SQ002]',
    'q3a', 'q3b[SQ001]', 'q3b[SQ002]', 'q3c',
    'q4a[SQ001]', 'q4a[SQ002]', 'q4b[SQ001]', 'q4b[SQ002]'
]

AF_NEW_COLUMNS = [
    'vigorosa_dias', 'vigorosa_horas', 'vigorosa_minutos',
    'moderada_dias', 'moderada_horas', 'moderada_minutos',
    'caminhada_dias', 'caminhada_horas', 'caminhada_minutos', 'caminhada_ritmo',
    'sentada_semana_horas', 'sentada_semana_minutos',
    'sentada_fds_horas', 'sentada_fds_minutos'
]