# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import json
from typing import Dict, Any

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