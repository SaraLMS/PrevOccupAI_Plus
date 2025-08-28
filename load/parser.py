# -------------------------------------------------------------------------------------------------------------------- #
# imports
# -------------------------------------------------------------------------------------------------------------------- #
from pathlib import Path
from typing import List, Dict
import re
from constants import ACQUISITION_PATTERN


# -------------------------------------------------------------------------------------------------------------------- #
# public functions
# -------------------------------------------------------------------------------------------------------------------- #



# def filter_files_by_folder(folder_name: str, files: List[Path]) -> List[Path]:
#     """
#     Filters a list of file paths, keeping only those that contain a given folder name.
#
#     :param folder_name: The folder name to search for (e.g. "10-20-00", "2022-06-21")
#     :param files: List of Path objects
#     :return: List of Path objects that contain the folder name in their parents
#     """
#     # innit list to hold only the wanted paths
#     filtered: List[Path] = []
#
#     # cycle though the paths in the list
#     for file in files:
#
#         # if the path has a folder like folder_name, add to list
#         if folder_name in [p.name for p in file.parents]:
#             filtered.append(file)
#
#     return filtered
#
#
# def _get_unique_acquisition_times_from_paths(paths_dict):
#
#     acquisition_times: List[str] = []
#
#     # get cycle over the list with the paths for all devices
#     for paths_list in paths_dict.values():
#
#         # cycle over the paths in the list of paths
#         for path in paths_list:
#
#             # get time pattern to find in the folder names
#             acquisition_pattern = re.compile(ACQUISITION_PATTERN)
#
#             # find folders that have the pattern hh-mm-ss
#             match = acquisition_pattern.search(str(path.parent))
#
#             # add to list if it's not there already
#             if match.group(0) not in acquisition_times:
#                 acquisition_times.append(match.group(0))
#
#     return acquisition_times
# -------------------------------------------------------------------------------------------------------------------- #
# private functions
# -------------------------------------------------------------------------------------------------------------------- #