# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import json
from typing import Dict, Any
import os

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def load_json_file(json_path: str) -> Dict[Any, Any]:
    """
    Loads a json file.
    :param json_path: str
        Path to the json file
    :return: Dict[Any,Any]
    Dictionary containing the features from TSFEL
    """

    # read json file to a features dict
    with open(json_path, "r", encoding='utf-8') as file:
        json_dict = json.load(file)

    return json_dict


def create_dir(path, folder_name):
    """
    creates a new directory in the specified path
    :param path: the path in which the folder_name should be created
    :param folder_name: the name of the folder that should be created
    :return: the full path to the created folder
    """

    # join path and folder
    new_path = os.path.join(path, folder_name)

    # check if the folder does not exist yet
    if not os.path.exists(new_path):
        # create the folder
        os.makedirs(new_path)

    return new_path