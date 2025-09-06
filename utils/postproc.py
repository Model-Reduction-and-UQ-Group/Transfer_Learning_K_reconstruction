from typing import Any
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.globals import *


FORMAT_PNG : int = 0
FORMAT_PDF : int = 1
FORMAT_SVG : int = 2

FIG_ERROR_ID : int = 0
FIG_LOSS_ID  : int = 1


__N_REPS : int = 0
__N_RUNS : int = 0


__err_plot_show_index : int = 0

__do_show_figs : bool = False
__do_save_figs : bool = False


def setShowFigures(b : bool):
    global __do_show_figs
    __do_show_figs = b


def isShowing() -> bool:
    return __do_show_figs


def setSaveFigures(b : bool):
    global __do_save_figs
    __do_save_figs = b


__base_resdir : str = ""   
__figures_dir : str = ""   


def setPostprocBaseDir(d : str) -> None:
    global __base_resdir
    __base_resdir = d
    return



__saving_formats : dict[int, dict[str, Any]] = {
    FORMAT_PNG: { "on": False, "str": ".png"},
    FORMAT_PDF: { "on": False, "str": ".pdf"},
    FORMAT_SVG: { "on": False, "str": ".svg"}
}


def setFormatsFromDict(formats : dict[int, bool]) -> None:
    for _, v in enumerate(formats):
        __saving_formats[v]["on"] = formats[v]



def enableSaveFormat(id : int) -> None:
    if (id < 0 or id > FORMAT_SVG):
        print(f"ERROR: invalid format id  {id}.")
        return
    __saving_formats[id]["on"] = True



def __saveFigureWithFormats(fig, name : str) -> None:
    for k, v in enumerate(__saving_formats):
        if (__saving_formats[v]["on"]):
            fig.savefig(name + __saving_formats[v]["str"])



def saveFigure(fig : Figure, name : str = None) -> None:
    global __figures_dir
    if (fig is None):
        return
    if (name is None):
        name_ = f"Figure{fig.number}"
    else:
        name_ = name
    if (__figures_dir is None):
        __figures_dir = __base_resdir + os.path.sep + "figures"
    if (not os.path.exists(__figures_dir)):
        os.mkdir(__figures_dir)
    name_ = __figures_dir + os.path.sep + name_
    __saveFigureWithFormats(fig, name_)



def __lossPlot(d : str):
    data    = np.load(d + os.path.sep + "loss_data.npy", allow_pickle=True)
    figname = "Loss"
    f       = plt.figure(figname, visible=True)

    n_phases = data.shape[0]
    for i in range(n_phases):
        N = data[i, 0].shape[0]
        x = np.arange(1, N+1)
        plt.plot(x, data[i, 0], \
                 linewidth=1, label=f"Phase {i} (train)")
        plt.plot(x, data[i, 1], '--', \
                 linewidth=1, label=f"Phase {i} (test)")

    plt.draw()
    plt.legend()
    if (__do_save_figs):
        saveFigure(f, figname)
    plt.close(f)
    return



__i_rep    : int = 0
__i_run    : int = 0
__m_coeffs : np.ndarray = None
__b_coeffs : np.ndarray = None


def __errorPlot(d : str, index : int = None) -> None:
    global __m_coeffs, __b_coeffs

    i = 1
    while (True): 
        if (not os.path.exists(d + os.path.sep + 'TargetTest_Phase{:d}.pth'.format(i))):
            i -= 1
            break
        i += 1
    OutputTest = torch.load(d + os.path.sep + 'OutputTest_Phase{:d}.pth'.format(i), map_location=torch.device('cpu'))
    TargetTest = torch.load(d + os.path.sep + 'TargetTest_Phase{:d}.pth'.format(i), map_location=torch.device('cpu'))


    if (index is not None):
        id = index
    else:
        id = __err_plot_show_index

    t_index = 0

    figname = "Output"
    f = plt.figure(figname, visible=True)
    plt.imshow(np.squeeze((OutputTest[id][t_index][:][:])), cmap='coolwarm') 
    plt.colorbar()
    plt.draw()
    if (__do_save_figs):
        saveFigure(f, figname)
    plt.close(f)

    figname = "Target"
    f = plt.figure(figname, visible=True)
    plt.imshow(np.squeeze((TargetTest[id][t_index][:][:])), cmap='coolwarm')
    plt.colorbar()
    plt.draw()
    if (__do_save_figs):
        saveFigure(f, figname)
    plt.close(f)

    TargetTest_ = TargetTest
    n_elems = np.prod(TargetTest.shape)
    TargetTest_ = np.reshape(TargetTest_, (n_elems))
    OutputTest_ = OutputTest[:, :, :, :]
    OutputTest_ = np.reshape(OutputTest_, (n_elems))

    corr_matrix = np.corrcoef(TargetTest_, OutputTest_)
    corr        = corr_matrix[0, 1]
    R_sq        = corr**2

    m, b = np.polyfit(TargetTest_, OutputTest_, 1)
    __m_coeffs[__i_rep, __i_run] = m
    __b_coeffs[__i_rep, __i_run] = b


    figname = "Regression"
    f = plt.figure(figname, visible=True)
    plt.scatter(TargetTest[id, :, :, :], OutputTest[id, :, :, :], \
                s=1.5, c= 'k')
    plt.plot(TargetTest_, m*TargetTest_ + b,
            color = 'darkcyan', alpha = 0.6, linewidth = 2.0,
            label = 'Regression Line: y = {:.2f}x + {:.2f}'.format(m, b))

    maxtarget_ = np.max(TargetTest[id, :, :, :]) * 1.1
    bisect_    = [0, maxtarget_]
    plt.plot(bisect_, bisect_, 'plum',
            alpha = 0.9, linewidth=2.0, label = 'Bisector')
    plt.xlim([0, plt.gca().get_xlim()[1]])
    plt.ylim([0, plt.gca().get_xlim()[1]])
    plt.xlabel('Target - Test')
    plt.ylabel('Output - Test')
    plt.legend()
    stats = (f'$R^2$ = {R_sq:.2f}\n')
    bbox  = dict(boxstyle='round', fc='blanchedalmond', ec='brown', alpha=0.5, fill =True)
    plt.text(0.95, 0.07, stats, fontsize=12, bbox=bbox,
            transform=plt.gca().transAxes, horizontalalignment='right')
    plt.draw()
    if (__do_save_figs):
        saveFigure(f, figname)
    plt.close(f)
    return




def __plot(base : str) -> None:
    global __figures_dir, __i_rep, __i_run

    for item in os.listdir(base):
        if ((os.path.isdir(base + os.path.sep + item) and (item.startswith("rep") or item.startswith("run")))):
            if (item.startswith("rep")):
                irep = int(item[4 : ]) - 1
                if (not irep == __i_rep):
                    __i_run = 0
                __i_rep = irep
            __plot(base + os.path.sep + item)
            __i_run += 1
        else:
            if (os.path.isdir(base + os.path.sep + item)):
                continue
            if (not (os.path.exists(base + os.path.sep + "loss_data.npy"))):
                continue
            __figures_dir = base + os.path.sep + "figures"
            if (not os.path.exists(__figures_dir)):
                os.mkdir(__figures_dir)
            print(f"INFO: Plotting results in  {__figures_dir}")
            __errorPlot(base)
            __lossPlot(base)
            break



__param_set : dict = None


def bindParamSetPostproc(pset : dict) -> None:
    global __param_set
    __param_set = pset
    return



def __getLastNumAsStr(line : str) -> str:
    i = len(line)-1
    while (not line[i : i+1] == ' '):
        i -= 1
    return line[i+1 : ]


def __readParamsTxtFile(fn : str) -> float:
    with open(fn, "r") as f:
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        line = f.readline()
        if (__param_id == TEST_PARAMETER_LR):
            substr = __getLastNumAsStr(line)
            return float(substr)
        line = f.readline()
        if (__param_id == TEST_PARAMETER_WD):
            substr = __getLastNumAsStr(line)
            return float(substr)
        line = f.readline()
        if (__param_id == TEST_PARAMETER_F):
            substr = __getLastNumAsStr(line)
            return float(substr)
        line = f.readline()
        if (__param_id == TEST_PARAMETER_MLR):
            substr = __getLastNumAsStr(line)
            return float(substr)

def __getParamsFromReps():
    global __N_REPS, __N_RUNS

    if (__param_id is None):
        return None
    res : np.ndarray = np.zeros(__N_REPS)
    i_rep = 0
    for item in os.listdir(__base_resdir):
        if (item.startswith("rep_") and (os.path.isdir(__base_resdir + os.path.sep + item))):
            res[i_rep] = __readParamsTxtFile(
                os.path.join(__base_resdir, item, "run_001", "params.txt"))
            i_rep += 1
    return res


__param_str : str = None
__param_id  : int = 0


def __inferParamFromDirName() -> None:
    global __param_id, __param_str

    i = len(__base_resdir)-1
    while (True):
        i -= 1
        if (__base_resdir[i:i+1] == '/'):
            break
    par_str : str  = __base_resdir[i+1 : ]
    for i in range(len(TEST_PARAM_STR)):
        if (par_str == TEST_PARAM_STR[i]):
            __param_str = par_str
            __param_id  = i
            return
    print(f"WARNING: could not infer test parameter from directory name {__base_resdir}.")


__box_plot_yscale : bool = False


def __box_plots() -> None:
    global __param_set, __box_plot_yscale, __m_coeffs, __b_coeffs

    rmse : np.ndarray = np.load(__base_resdir + os.path.sep + "rmse.npy")
    rmse = rmse

    phase_id = rmse.shape
    n_reps   = phase_id[0]
    phase_id = phase_id[len(phase_id)-1] - 1

    par_str = ""
    if (__param_str is not None):
        par_str = __param_str

    params = __getParamsFromReps()

    x_labels = []
    if (params is not None):
        for i in range(n_reps):
            val : float = params[i]
            x_labels.append("{:.1E}".format(val))
    else:
        for i in range(n_reps):
            x_labels.append("{:.1E}".format(i))  

    name  = "Rmse"
    f     = plt.figure(name, visible=True)
    x     = rmse[:, :, phase_id].transpose()

    flierprops  = dict(marker='x', markerfacecolor='none', markersize=3, markeredgecolor='k')
    medianprops = dict(linestyle='-',  linewidth=2.5, color='#029386')
    meanprops   = dict(linestyle='--', linewidth=1.8, color='orange')

    plt.boxplot(x, labels=x_labels, medianprops=medianprops, flierprops=flierprops, \
                showmeans=True, meanline=True, meanprops=meanprops)
    plt.ylabel("RMSE")
    plt.xlabel(par_str)
    if (__box_plot_yscale):
        plt.yscale("log")
    plt.draw()
    if (__do_save_figs):
        saveFigure(f, name)
    plt.close(f)
    return


def __inferNumsFromDirStructure() -> None:
    global __N_REPS, __N_RUNS

    for item in os.listdir(__base_resdir):
        if (item.startswith("rep_") and (os.path.isdir(__base_resdir + os.path.sep + item))):
            __N_REPS += 1
    # TODO: SET MANUALLY.
    __N_RUNS = 20


def generatePlots() -> None:
    global __figures_dir, __base_resdir
    global __m_coeffs, __b_coeffs

    if (not __do_show_figs):
        plt.ioff()

    __inferNumsFromDirStructure()
    __inferParamFromDirName()

    __m_coeffs = np.zeros((__N_REPS, __N_RUNS), dtype=float)
    __b_coeffs = np.zeros((__N_REPS, __N_RUNS), dtype=float)

    __plot(__base_resdir)

    __figures_dir = __base_resdir
    if (not os.path.exists(__figures_dir)):
        os.mkdir(__figures_dir)
    __box_plots()
    return


