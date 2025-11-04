# # ------------------------------------------------------------------------------------------------------------------- #
# # imports
# # ------------------------------------------------------------------------------------------------------------------- #
# import pandas as pd
# import os
# from pathlib import Path
# from typing import Dict, Any
#
# # internal imports
# from .json_parser import filter_config_dict_by_id
#
# # ------------------------------------------------------------------------------------------------------------------- #
# # constants
# # ------------------------------------------------------------------------------------------------------------------- #
# LIKERT_SCALE = 'likert'
#
# # ------------------------------------------------------------------------------------------------------------------- #
# # public functions
# # ------------------------------------------------------------------------------------------------------------------- #
#
# def filter_results_dataframe(results_dict: Dict[str, pd.DataFrame], config_dict: Dict[str, Dict[str, Any]]) \
#         -> Dict[str, pd.DataFrame]:
#     """
#
#     Filters the dataframes with the questionnaire results based on the scale of the questions. The resulting dataframe
#     contains only the relevant information such as the id of the subject and the question values and can now be used
#     to calculate the questionnaire scores.
#
#     This function first extracts from the config file the information regarding the scale of the questions for each
#     questionnaire of the domain and then filters the columns of the dataframe with the results based on the scale of the questions.
#
#     :param results_dict: Dictionary with the results for all questionnaires of the given domain.
#     :param config_dict: Config json file for the given domain, loaded into a dictionary
#     :return: Dict[str, pd.DataFrame] where the keys are the questionnaire ids of the given domain and the values are
#             the dataframes with the filtered results.
#     """
#     # init dictionary to hold the filtered results
#     filtered_results_dict: Dict[str, pd.DataFrame] = {}
#
#     # iterate through the multiple questionnaire results
#     for questionnaire_id, results_df in results_dict.items():
#
#         # (2) filter config dict to have only the information for this questionnaire
#         questionnaire_info_dict = filter_config_dict_by_id(config_dict, questionnaire_id)
#
#         # get questionnaire topics
#         topics_dict = questionnaire_info_dict.get("topics", {})
#
#         # cycle over the dictionary with the questionnaire info
#         for subtopic_name, subtopic_items in topics_dict.items():
#
#             # cycle over the inner dict containing the matrix ('S1', 'S2'....) information
#             for matrix_id, question_info_dict in subtopic_items.items():
#
#                 # get the scale of the answers for the questions with the correspondent matrix id
#                 scale = question_info_dict['type']
#
#                 # Check if the DataFrame has a column containing the matrix_id (S1, S2 ....)
#                 matching_columns = [col for col in results_df.columns if matrix_id in col]
#
#                 # (3) Apply cleaning depending on the scale type
#                 for col in matching_columns:
#
#                     if scale == LIKERT_SCALE:
#
#                         # clean likert scale results
#                         results_df[col] = _clean_likert_results(results_df[col])
#
#                     else:
#                         # TODO implement other scales
#                         pass
#
#                 # add dataframe to the filtered results dict
#                 filtered_results_dict[questionnaire_id] = results_df
#
#     return filtered_results_dict
#
#
# # ------------------------------------------------------------------------------------------------------------------- #
# # private functions
# # ------------------------------------------------------------------------------------------------------------------- #
#
# def _clean_likert_results(df_column: pd.Series) -> pd.Series:
#     """
#
#     :param df_column:
#     :return:
#     """
#
#     # remove the prefix 'A' from
#     df_column = df_column.str.replace("^A", "", regex=True)
#
#     return df_column


