# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import questionnaire_processing as qp

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
PROCESS_PSICOSSOCIAL = False
PROCESS_PESSOAIS = False
PROCESS_AMBIENTE = False
PROCESS_BIOMECANICO = False
GENERATE_LS_RESULTS = True

quest_path = "D:\\Backup PrevOccupAI data\\jan2023\\data\\group3\\questionnaires"
ls_output_path = "C:\\Users\\srale\\Desktop\\TESTE"

# ------------------------------------------------------------------------------------------------------------------- #
# program starts here
# ------------------------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':


    if PROCESS_PSICOSSOCIAL:
        qp.calculate_linear_scores(quest_path, domain='psicosocial')

    if PROCESS_AMBIENTE:
        qp.calculate_linear_scores(quest_path, domain='ambiente')

    if PROCESS_PESSOAIS:
        qp.calculate_personal_scores(quest_path)

    if PROCESS_BIOMECANICO:
        qp.calculate_biomechanical_scores(quest_path)


    if GENERATE_LS_RESULTS:
        qp.generate_results_csv_files(ls_output_path)