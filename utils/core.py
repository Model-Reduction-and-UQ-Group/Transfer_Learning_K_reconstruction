import os

import torch
import numpy as np

from h5py import File

import time
from timeit import default_timer as timer

# Local imports
from .postproc import *
import utils.tlmf as tlmf
import utils.globals as glob


USE_LF_ONLY   : int = 0
USE_HF_ONLY   : int = 1
USE_HF_AND_LF : int = 2


__I_REP    : int  = 0
__I_RUN    : int  = 0
__N_REPS   : int  = 1
__N_RUNS   : int  = 1
__N_PHASES : int = None
__N_EPOCHS : np.ndarray = None


# NOTE: testing purposes only
__ENABLE_TEST: bool = False


# TODO: might be removed..
__PYT_LOG_INTERVAL : int  = 1  # PyTorch logging interval


__BATCH_SIZE : int = 40


__n_reps_forced:  bool = False
__test_param_set: bool = False
__TEST_PARAM_ID:  int  = -1


__data_usage_set : bool = False
__DATA_USAGE  : int = USE_HF_AND_LF


__N_REPS_LOAD : int = 1


def testParameter(i):
    global __test_param_set, __TEST_PARAM_ID
    if (i < TEST_PARAMETER_LR or i > TEST_PARAMETER_MLR):
        print(f"\n ERROR: invalid test parameter ID  {i}\n")
        return
    __test_param_set = True
    __TEST_PARAM_ID  = i
    return


def forceNReps(n):
    global __n_reps_forced, __N_REPS
    __N_REPS = n
    __n_reps_forced = True



__params_set : np.ndarray = None


def __setNReps():
    global __N_REPS, __params_set
    if (__test_param_set and __params_set is not None):
        __N_REPS = __params_set[__TEST_PARAM_ID].size
    else:
        __N_REPS = 1
    return


def setNRuns(n):
    global __N_RUNS
    __N_RUNS = n


def setEpochs(nepochs):
    global __N_EPOCHS
    __N_EPOCHS = np.array(nepochs, dtype=np.int32)


def setBatchSize(bs):
    global __BATCH_SIZE
    __BATCH_SIZE = bs


def setDataUsage(i):
    global __DATA_USAGE, __data_usage_set
    __data_usage_set = True
    __DATA_USAGE     = i


def setDataLoadRepetitions(n):
    global __N_REPS_LOAD
    __N_REPS_LOAD = n




hdf5_data_dir: str = ""


def setHDF5DataDirectory(d: str) -> None:
    global hdf5_data_dir
    hdf5_data_dir = d
    return



class DSData:
    _hdf5file: str = ""
    _field:    str = ""

    n_samp: int = 0
    dims:  tuple[int] = None

    # _data  = None
    _shape = None
    _handle_fct_ptr = None  

    def __init__(self, mf_specs, nsamp):
        global hdf5_data_dir
        self.n_samp    = nsamp
        self._hdf5file = hdf5_data_dir + mf_specs["fname"]
        self._field    = mf_specs["field"]
        return

    def set_dims(self, dims) -> None:
        self.dims = dims
        return

    def set_data_callback_fct(self, fct) -> None:
        self._handle_fct_ptr = fct
        return

    def modify_dataset_data(self, data):
        if (self._handle_fct_ptr is not None):
            data = self._handle_fct_ptr(data)
        # self._data = data
        self._shape = data.shape
        return data

    def get_size(self):
        return self._shape[2:4]



class SimData:
    input:  DSData = None
    target: DSData = None

    def __init__(self, inp_specs, inp_samp, tgt_specs, tgt_samp) -> None:
        self.input  = DSData(inp_specs, inp_samp)
        self.target = DSData(tgt_specs, tgt_samp)
        return


def getDataFromHDF5File(file=None, field=None):
    if (file is None or field is None):
        return None
    with File(file, "r") as f:
        field_data_ = np.transpose(f[field], (0, 1, 2, 3))
    return field_data_


def getDownscaledData(up_data, scale: int):
    if (up_data is None):
        return None
    # NOTE: kronecker product
    b = np.repeat(up_data, scale, axis=1)
    return np.repeat(b,    scale, axis=0)


def addDatasetToHDF5File(data=None, file=None, field=None) -> int:
    if (data is None or file is None or field is None):
        return 1

    field = "/" + field
    f = File(file, "r+")
    if field in f:
        del f[field]
        _ = f.create_dataset(field, data=data)
    else:
        _ = f.create_dataset(field, data=data)
    f.close()
    return 0

__OUT_RESULTS_DIR : str = None


def __setOutResultDirectory():
    global __OUT_RESULTS_DIR

    if (__RUN_DIR is not None):
        __OUT_RESULTS_DIR = __RUN_DIR
    return


__BASE_DIR : str = None
__RUN_DIR  : str = None


def setRunDir(d : str):
    global __BASE_DIR

    __BASE_DIR = d
    if (not os.path.isabs(__BASE_DIR)):
        __BASE_DIR = CWD + os.path.sep + __BASE_DIR
    if (not os.path.exists(__BASE_DIR)):
        os.makedirs(__BASE_DIR)
    return


def _setRunBaseDir():
    global __BASE_DIR
    if (__data_usage_set):
        if (__DATA_USAGE == USE_LF_ONLY):
            __BASE_DIR = __BASE_DIR + os.path.sep + "LF_training"
        elif (__DATA_USAGE == USE_HF_ONLY):
            __BASE_DIR = __BASE_DIR + os.path.sep + "HF_training"
        else:
            __BASE_DIR = __BASE_DIR + os.path.sep + "LF_HF_training"

        if (__test_param_set):
            __BASE_DIR = __BASE_DIR + os.path.sep + TEST_PARAM_STR[__TEST_PARAM_ID]
        else:
            __BASE_DIR = __BASE_DIR + os.path.sep + "default_run"


def __setRunBaseDir():
    global __RUN_DIR, __BASE_DIR, __I_REP, __I_RUN

    __RUN_DIR = __BASE_DIR

    # get to select right folder
    if (__I_REP == 0 and __I_RUN == 0):
        __I_RUN = 1
        __I_REP = 1
    else:
        __I_RUN += 1
        if (__I_RUN > __N_RUNS):
            __I_RUN  = 1
            __I_REP += 1

    if (__I_REP <= __N_REPS):
        __RUN_DIR = os.path.join(__RUN_DIR,
              SIMULATION_REP_PREFIX + "{:03d}".format(__I_REP),
              SIMULATION_RUN_PREFIX + "{:03d}".format(__I_RUN))

        print(f"Info:  Base run directory: {__RUN_DIR}.")
        if (not os.path.exists(__RUN_DIR)):
            os.makedirs(__RUN_DIR)
        __setOutResultDirectory()
    return




__do_save_models : bool = False


def setModelsSaving(b : bool):
    global __do_save_models
    __do_save_models = b
    return


def saveFigures(b : bool, f : dict) -> None:
    setSaveFigures(b)
    setFormatsFromDict(f)
    return


def showFigures(b : bool) -> None:
    setShowFigures(b)
    return


def bindParametersSet(pset):
    global __params_set, __N_REPS, __N_PHASES
    __params_set = pset
    __N_REPS     = pset.shape[0]
    n = pset.shape[1]
    return


def getParametersSet() -> dict | None:
    global __params_set
    return __params_set




__default_param_vals : dict[int, np.ndarray] = None


def bindDefaultParamValues(vals : dict[int, np.ndarray]) -> None:
    global __default_param_vals
    __default_param_vals = vals
    return



def setNPhases(n : int):
    global __N_PHASES
    if (__N_PHASES is not None and n != __N_PHASES):
        print("ERROR: n. of phases mismatch")
        exit(1)
    elif (__N_PHASES is None):
        __N_PHASES = n
    tlmf.setNPhases_(n)
    return



def __readDataFromHDF5File(dataset: DSData, reps: int = None):
    data   = None
    fn     = dataset._hdf5file
    field  = dataset._field
    f      = File(fn, mode="r")
    failed = False
    try:
        data_ = f[field][()]
    except:
        print(f"\nField  \"{field}\"  does not exist in  \"{fn}\".\n")
        failed = True
    if (not failed and data_ is not None):
        data = dataset.modify_dataset_data(data_)
    else:
        print(f"Error:  failed to read data from  {fn}.")
        exit(1)
    f.close()
    return data




__getModelOrig_ftcPtr = None


def setModelOrigCallback(fptr):
    global __getModelOrig_ftcPtr
    __getModelOrig_ftcPtr = fptr
    return


def __getModelOrig():
    global __getModelOrig_ftcPtr

    if (__getModelOrig_ftcPtr is None):
        return None
    return __getModelOrig_ftcPtr()



__getModelFromBase_fptr = None


def setGetBaseModelCallback(fptr):
    global __getModelFromBase_fptr
    __getModelFromBase_fptr = fptr


def __getModelFromBase(base_model, phase : int = None):
    global __N_PHASES, __getModelFromBase_fptr
    if (phase is None):
        print("\nERROR: Please specify a phase index\n")
        return None

    if (phase < 1):
        print("\nPhase index is lass than 1!\n")
        return None
    if (__N_PHASES is not None):
        if (phase > __N_PHASES):
            print(f"\nOnly {__N_PHASES} available!\n")
            return None

    if (__getModelFromBase_fptr is None):
        return None
    tlmf.setPhaseIndex(phase)
    return __getModelFromBase_fptr(base_model, phase)




__getModelReady_fptr : None


def setModelReadyCallback(fptr):
    global __getModelReady_fptr
    __getModelReady_fptr = fptr



__n_reps_epochs : int = 1


def trainModel(orig, tsl, trl, phase):
    phase_ = phase - 1
    return tlmf.trainModel_(
        model_orig=orig,
        test_loader=tsl,
        train_loader=trl,
        reps=__n_reps_epochs,
        n_epochs=__N_EPOCHS[phase_],
        lr=__default_param_vals[TEST_PARAMETER_LR][phase_],
        wd=__default_param_vals[TEST_PARAMETER_WD][phase_],
        factor=__default_param_vals[TEST_PARAMETER_F][phase_],
        min_lr=__default_param_vals[TEST_PARAMETER_MLR][phase_],
        log_interval=__PYT_LOG_INTERVAL)




# Important results per phase.
__TIMES_PHASES: np.ndarray = None
__RMSE_PHASES : np.ndarray = None


def __saveLossPlotData() -> None:
    data = tlmf.getLossData_()
    np.save(__OUT_RESULTS_DIR + os.path.sep + "loss_data", data)
    return


def __saveSimData() -> None:
    base = __BASE_DIR + os.path.sep
    np.save(base + "times.npy", __TIMES_PHASES)
    np.save(base + "rmse.npy",  __RMSE_PHASES)

    # TO avid unwanted overrides..
    f = "times_backup.npy"
    if (not os.path.exists(base + f)):
        np.save(base + f, __TIMES_PHASES)
    f = "rmse_backup.npy"
    if (not os.path.exists(base + f)):
        np.save(base + f,  __RMSE_PHASES)
    return




def add_downscaled_data_to_hdf5file(hdf5file, hdf5field, scale, fieldstr) -> None:
    global hdf5_data_dir
    f = hdf5_data_dir + hdf5file
    input_       = getDataFromHDF5File(f, hdf5field)
    target_down_ = getDownscaledData(input_, scale)
    if (0 != addDatasetToHDF5File(data=target_down_, file=f, field=fieldstr)):
        print("  [ERROR]  While adding dataset to file " + f)
        exit(1)
    else:
        print("Info:  Dataset correctly added to file " + f)
    return





def setReductionType(id : int) -> None:
    tlmf.setReductionTypeStr(RMSE_REDUCTION_TYPE[id])
    return



__save_sim_params: bool = True



def write_param_file(iphase, phase_specs, f) -> None:
    global __n_reps_epochs, __N_EPOCHS, __default_param_vals
    global TEST_PARAMETER_LR, TEST_PARAMETER_WD, TEST_PARAMETER_F, TEST_PARAMETER_MLR

    f.write(f"PHASE  {iphase+1}:\n")
    # f.write("   N. TEST:             {:7d}\n".format(mf_data_phase.input.n_samp))
    # f.write("   N. TRAIN:            {:7d}\n".format(mf_data_phase.target.n_samp))
    f.write("   - N. REPS:             {:7d}\n".format(__n_reps_epochs))
    f.write("   - N. EPOCHS:           {:7d}\n".format(__N_EPOCHS[iphase]))
    f.write("   - LEARNING RATE:       {:7g}\n".format(__default_param_vals[TEST_PARAMETER_LR][iphase]))
    f.write("   - WEIGHT DECAY:        {:7g}\n".format(__default_param_vals[TEST_PARAMETER_WD][iphase]))
    f.write("   - FACTOR:              {:7g}\n".format(__default_param_vals[TEST_PARAMETER_F][iphase]))
    f.write("   - MIN. LEARNING RATE:  {:7g}\n".format(__default_param_vals[TEST_PARAMETER_MLR][iphase]))
    f.write("\n")
    f.write("   Multi-fidelity data level:\n")
    f.write("      Train:\n")
    f.write("        - input:    {:s}\n".format(phase_specs[iphase]["train"]["inp"]))
    f.write("        - target:   {:s}\n".format(phase_specs[iphase]["train"]["tgt"]))
    f.write("      Test:\n")
    f.write("        - input:    {:s}\n".format(phase_specs[iphase]["test"]["inp"]))
    f.write("        - target:   {:s}\n".format(phase_specs[iphase]["test"]["tgt"]))
    f.write("\n")
    return



def run(phase_data, phase_specs):
    global __TIMES_PHASES, __RMSE_PHASES
    global __BASE_DIR, __RUN_DIR
    global __getModelReady_fptr

    if (__BASE_DIR is None):
        setRunDir(CWD)
    _setRunBaseDir()

    # Main loop
    __TIMES_PHASES = np.zeros((__N_REPS, __N_RUNS, __N_PHASES), dtype=np.float64)
    __RMSE_PHASES  = np.zeros((__N_REPS, __N_RUNS, __N_PHASES), dtype=np.float64)
    __model_best = None
    __rmse_best  = 10e10
    _data = {}

    f = None
    while (True):

        __setRunBaseDir()
        if (__save_sim_params and f is None):
            f = open(__RUN_DIR + os.path.sep + "params.txt", "w")

        if (__test_param_set and __params_set is not None and __I_RUN == 1):
            __default_param_vals[__TEST_PARAM_ID] = __params_set[__I_REP-1]
            pass

        if (__do_save_models):
            tlmf.saveModels(True)
            tlmf.setModelSaveBaseDir(__OUT_RESULTS_DIR)

        tlmf.initLossCurvesArr(__N_PHASES)

        print(f"Info:  initialising rep  {__I_REP},  run  {__I_RUN}:")

        # Before starting training, the "previous" model is
        # initialised to the original (base) one.
        model_previous_phase = __getModelOrig()
        for iphase in range(__N_PHASES):

            phase_id = iphase + 1
            tlmf.setPhaseIndex(phase_id)

            # Get HDF5 description 
            mf_data_phase = phase_data[iphase]

            if (__save_sim_params and f is not None):
                write_param_file(iphase, phase_specs, f)

            # Get HDF5 data for current phase
            # NOTE: depends on current phase Data-Fidelity level
            for tstr in ["train", "test"]:
                field = "input_" + tstr
                _data[field] = __readDataFromHDF5File(mf_data_phase[tstr].input, __N_REPS_LOAD)
                field = "target_" + tstr
                _data[field] = __readDataFromHDF5File(mf_data_phase[tstr].target, __N_REPS_LOAD)

            # Get current phase NN model
            model__ = __getModelFromBase(model_previous_phase, phase_id)

            # Get data loaders (data is only for current phase)
            train_loader, test_loader = tlmf.getDataLoader(_data, mf_data_phase, __BATCH_SIZE)

            # NOTE: here we can directly call the fptr, no wrapper needed.
            #       Get model ready to run !!
            __getModelReady_fptr(model__, phase_id)

            if (not __ENABLE_TEST):
                tstart = time.time()
                model_previous_phase, rmse_best = trainModel(model__,
                                                            test_loader, train_loader, phase_id)
                __TIMES_PHASES[__I_REP-1, __I_RUN-1, iphase] = time.time() - tstart
                __RMSE_PHASES [__I_REP-1, __I_RUN-1, iphase] = rmse_best
                print(f"         - Phase  {phase_id}:  RMSE BEST = {rmse_best}")

        # NOTE: here loop on phases is done.
        #       So, we save base RMSE of last phase only (over different reps and runs!)
        if (rmse_best < __rmse_best):
            __rmse_best  = rmse_best
            __model_best = model_previous_phase

        __saveLossPlotData()

        if (__I_REP == __N_REPS and __I_RUN == __N_RUNS):
            break
        pass

    if (__save_sim_params and f is not None):
        f.close()

    if (not __ENABLE_TEST):
        __saveSimData()
        torch.save(__model_best.state_dict(), __OUT_RESULTS_DIR + os.path.sep + 'ModelBest')
    return




def postProc(d : str) -> None:
    setPostprocBaseDir(d)
    bindParamSetPostproc(__params_set)
    generatePlots()
    if (isShowing()):
        plt.show()
    pass



__n_sims_gen : int = 0


def setNumOfGenerations(n : int) -> None:
    global __n_sims_gen
    __n_sims_gen = n


__GEN_DEFAULT_RUN_ENABLED: bool = False


def generationEnableDefaultRun(b: bool) -> None:
    global __GEN_DEFAULT_RUN_ENABLED
    __GEN_DEFAULT_RUN_ENABLED = b
    return

