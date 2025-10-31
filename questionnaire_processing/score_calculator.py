# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from pathlib import Path
import os
from typing import Dict
import pandas as pd

# internal imports
from utils import load_json_file
from .questionnaire_loader import load_questionnaire_answers
from .questionnaire_processor import filter_results_dataframe
from .json_parser import get_questionnaire_name_from_json

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
CONFIG_FOLDER_NAME = 'config_files'
JSON_SCORES_FILENAME = 'scores.json'
JSON_PSICOSSOCIAL_FILENAME = 'cfg_psicossocial.json'

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def calculate_psicossocial_scores(folder_path: str, domain: str):

    # init dictionary to hold the scores
    scores_df = pd.DataFrame()

    # load results for all domain questionnaires into a dictionary
    # (keys: questionnaire id, values: dataframe with the results)
    results_dict = load_questionnaire_answers(folder_path, domain)

    # load json file with the info for the given domain
    config_dict = load_json_file(os.path.join(Path(__file__).parent, CONFIG_FOLDER_NAME, JSON_PSICOSSOCIAL_FILENAME))

    # load json file with the scores info
    scores_dict = load_json_file(os.path.join(Path(__file__).parent, CONFIG_FOLDER_NAME, JSON_SCORES_FILENAME))

    # filter results_dict to hold only relevant information
    results_dict = filter_results_dataframe(results_dict=results_dict, config_dict=config_dict)

    # iterate through the results of the psicossocial questionnaires
    for questionnaire_id, results_df in results_dict.items():

        # (1) get questionnaire name from id
        questionnaire_name = get_questionnaire_name_from_json(config_dict, questionnaire_id)

        # find questionnaire name in the scores json file
        calculation_method = scores_dict[questionnaire_name]["calculation"]

        # calculate scores per subject and add to dataframe
        scores_series = _calculate_scores(results_df, calculation_method)
    #
    #     if 'id' not in scores_df.columns:
    #
    #         # add id to the final dataframe if not already there
    #         scores_df['id'] = results_df['id']
    #
    #     # add scores
    #     scores_df['questionnaire_id'] = scores_series
    #
    # # sort dataframe
    # scores_df = scores_df.sort_values(by='id')
    #
    # return scores_df

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _calculate_scores(results_df: pd.DataFrame, calculation_method: str) -> pd.Series:

    # drop id column
    results_df = results_df.drop(results_df.columns[0], axis=1)

    if calculation_method == 'sum':

        # sum all values per row
        scores_series = results_df.sum(axis=1)

    elif calculation_method == 'mean':

        # calculate the mean of all values per row
        scores_series = results_df.mean(axis=1)

    else:
        raise ValueError(f"The calculation method {calculation_method} does not exist.")

    return scores_series