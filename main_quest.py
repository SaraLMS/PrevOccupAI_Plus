# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from questionnaire_processing.linear_score_calculator import calculate_linear_scores
from questionnaire_processing.personal_score_calculator import calculate_personal_scores

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
PROCESS_PSICOSSOCIAL = False
PROCESS_PESSOAIS = False
PROCESS_AMBIENTE = False

quest_path = "D:\\Backup PrevOccupAI data\\jan2023\\data\\group3\\questionnaires"
domain = "psicosocial"

# ------------------------------------------------------------------------------------------------------------------- #
# program starts here
# ------------------------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':


    if PROCESS_PSICOSSOCIAL:
        calculate_linear_scores(quest_path, domain='psicosocial')

    if PROCESS_AMBIENTE:
        calculate_linear_scores(quest_path, domain='ambiente')

    if PROCESS_PESSOAIS:
        calculate_personal_scores(quest_path)


