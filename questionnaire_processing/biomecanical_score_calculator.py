# -------------------------------------------------------------------------------------------------------------------- #
# imports
# -------------------------------------------------------------------------------------------------------------------- #
import os
from pathlib import Path
import pandas as pd

# internal imports
from .questionnaire_loader import load_questionnaire_answers
from utils import load_json_file, create_dir, get_group_from_path
from constants import (CONFIG_FOLDER_NAME, RESULTS_FOLDER_NAME, CSV)


# -------------------------------------------------------------------------------------------------------------------- #
# constants
# -------------------------------------------------------------------------------------------------------------------- #
DESIGN_ESCRITORIO = "Design do Escritório"
EQUIPAMENTOS = "Equipamentos"
INCAPACIDADE_DOR = "Incapacidade e Sofrimento associados a Dor"

# -------------------------------------------------------------------------------------------------------------------- #
# public functions
# -------------------------------------------------------------------------------------------------------------------- #
def calculate_personal_scores(folder_path):

    # load results for all domain questionnaires into a dictionary
    # (keys: questionnaire id, values: dataframe with the results)
    results_dict = load_questionnaire_answers(folder_path, domain="pessoais")

    # load config json file
    config_dict = load_json_file(os.path.join(Path(__file__).parent, CONFIG_FOLDER_NAME, "cfg_pessoais.json"))

    for questionnaire_id, answers_df in results_dict.items():

        # Check if the questionnaire_id exists in config_dict
        if questionnaire_id not in config_dict:
            print(f"Warning: questionnaire_id {questionnaire_id} not found in config. Skipping...")
            continue  # skip to the next one

        # get questionnaire name from config
        questionnaire_name = config_dict[questionnaire_id]

        if questionnaire_name == DESIGN_ESCRITORIO:

            results_df = _get_design_escritorio_results(answers_df)

        elif questionnaire_name == EQUIPAMENTOS:

            results_df = _get_equipamentos_results(answers_df)

        # it's incapacidade....
        else:
            results_df = _get_incapacidade_dor_results(answers_df)


        # set id column to int, set as index of the dataframe, and order
        results_df['id.1'] = pd.to_numeric(results_df['id.1'], errors='coerce')
        results_df = results_df.set_index('id.1').sort_index()

        # save dataframe into a csv file
        folder_path = create_dir(Path(__file__).parent, os.path.join(RESULTS_FOLDER_NAME, get_group_from_path(folder_path),'pessoais'))
        results_df.to_csv(os.path.join(folder_path, f"{questionnaire_name}{CSV}"))


# -------------------------------------------------------------------------------------------------------------------- #
# private functions
# -------------------------------------------------------------------------------------------------------------------- #

def _get_design_escritorio_results():
    pass

def _get_equipamentos_results():
    pass

def _get_incapacidade_dor_results():
    pass