import os
import sys

#
#  GLOBAL VARIABLES SECTION
#

TEST_VERSION: bool = True

IS_WIN_SYS: bool = sys.platform.find("win") >= 0
CWD: str = os.getcwd()

BASE_SIMULATION_DATA_DIR: str = None

# Directories prefixes/suffixes
SIMULATION_DIR_PREFIX: str = "sim_"
SIMULATION_RUN_PREFIX: str = "run_"
SIMULATION_REP_PREFIX: str = "rep_"


MF_DATA_TYPE_LOW_FIDELITY : int = 0
MF_DATA_TYPE_HIGH_FIDELITY: int = 1


TEST_PARAMETER_LR  : int = 0
TEST_PARAMETER_WD  : int = 1
TEST_PARAMETER_F   : int = 2
TEST_PARAMETER_MLR : int = 3
TEST_PARAMETER_NONE: int = 4
TEST_PARAM_STR: list[str] = ["LR", "WD", "F", "MLR"]


RMSE_USE_REDUCTION_SUM  : int = 0
RMSE_USE_REDUCTION_MEAN : int = 1
RMSE_REDUCTION_TYPE : dict = {
   RMSE_USE_REDUCTION_SUM: 'sum',
   RMSE_USE_REDUCTION_MEAN: 'mean'
}


def setBaseSimDir(d: str) -> None:
    global BASE_SIMULATION_DATA_DIR
    BASE_SIMULATION_DATA_DIR = d
    return


