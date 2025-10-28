# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any


# internal imports
from utils import load_json_file
from questionnaire_loader import load_questionnaire_answers
from json_parser import filter_config_dict_by_id

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
CONFIG_FOLDER_NAME = 'config_files'
JSON_PSICOSSOCIAL_FILENAME = 'cfg_psicossocial.json'
LIKERT_SCALE = 'likert'

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def filter_results_dataframe(folder_path: str, domain: str):

    # load results for all psicossocial questionnaires
    results_dict = load_questionnaire_answers(folder_path, domain)

    # load json file with the info
    config_dict = load_json_file(os.path.join(Path(__file__).parent, CONFIG_FOLDER_NAME, JSON_PSICOSSOCIAL_FILENAME))

    # iterate through the multiple questionnaire results
    for questionnaire_id, results_df in results_dict.items():

        # filter config dict to have only the information for this questionnaire
        questionnaire_info_dict = filter_config_dict_by_id(config_dict, questionnaire_id)

        # get questionnaire topics
        topics_dict = questionnaire_info_dict.get("topics", {})

        # cycle over the dictionary with the questionnaire info
        for subtopic_name, subtopic_items in topics_dict.items():

            # cycle over the inner dict containing the matrix ('S1', 'S2'....) information
            for matrix_id, question_info_dict in subtopic_items.items():

                # get the scale of the answers for the questions with the correspondent matrix id
                scale = question_info_dict['type']

                # Check if the DataFrame has a column containing the matrix_id
                matching_columns = [col for col in results_df.columns if matrix_id in col]

                # Apply cleaning depending on the scale type
                for col in matching_columns:

                    if scale == LIKERT_SCALE:

                        # clean likert scale results
                        pass

                    else:
                        # implement other scales
                        pass


    pass


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

