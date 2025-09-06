import sys
import utils.core as core

'''
"Accelerated training of neural networks via multi-fidelity simulations"
    "https://github.com/Model-Reduction-and-UQ-Group"

CNN surrogates trained on multiple levels of data
Sample code to train model on 128x128(HF), 64x64(LF) and 32x32(VLF) data. 
Final model will be saved.

The training script for the CNN surrogate levarges ideas and methodology inspired by: 
https://github.com/DDMS-ERE-Stanford/Transfer_Learning_on_Multi_Fidelity_Data.git

University of Bologna, Model Reduction and UQ Group

Alessia Chiofalo (alessia.chiofalo3@unibo.it)
Jan 2025
'''


# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************
#
#   General Data
#
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************


MODEL_TYPE_STR: str = "reconstruction"
MODEL_TYPE_DIR: str = core.CWD  + "/" + MODEL_TYPE_STR
MODEL_BASE_DIR: str = MODEL_TYPE_DIR + "/MODEL/"
# HDF5_DIR      : str = MODEL_BASE_DIR + "HDF5_DATA/"
HDF5_DIR      : str = "MFdata/"


#  ######################
#  Model parameters
#  ######################

__device = core.tlmf.getTorchAvailableDevice()

MODEL_NUM_INPUT_CHANNELS   = 16
MODEL_NUM_OUTPUT_CHANNELS  = 1
MODEL_BLOCKS_NUM_LAYERS    = (7, 12, 7)  # n. of layers for each denseED block.
MODEL_GROWTH_RATE          = 40
MODEL_DROP_RATE            = 0
MODEL_USE_BOTTLENECKS      = False
MODEL_BOTTLENECK_SIZE      = 8
MODEL_NUM_INITIAL_FEATURES = 64



#  ######################
#  RUN
#  ######################

NN_INPUT_DIMS:  tuple[int] = (16, 128, 128)
NN_TARGET_DIMS: tuple[int] = (1)

TEST_PARAMETER : int = core.TEST_PARAMETER_NONE
REDUCTION_TYPE : int = core.RMSE_USE_REDUCTION_SUM

N_RUNS : int = 1
N_REPS : int = 1

BATCH_SIZE  : int = 40

TRANSPOSE_IMAGE: bool = False
if (TRANSPOSE_IMAGE):
   DATA_INPUT_TRANSPOSE:   tuple[int, int, int, int] = (3, 2, 0, 1)
   DATA_OUTPUT_TRANSPOSE:  tuple[int, int, int, int] = (3, 2, 0, 1)
else:
   DATA_INPUT_TRANSPOSE:   tuple[int, int, int, int] = (3, 2, 0, 1)
   DATA_OUTPUT_TRANSPOSE:  tuple[int, int, int, int] = (3, 2, 0, 1)



# Multi-fidelity levels specifications
MF_SPECS = {
    "vlf": {
        "train": {
            "nsamp": 2,
            "input": {
                "fname": "vlf_dataset.h5",
                "field": "ss_down",
            },
            "target": {
                "fname": "hf_dataset.h5",
                "field": "kx",
            },
        },
        "test": {
            "nsamp": 1,
            "input": {
                "fname": "vlf_dataset.h5",
                "field": "ss_down",
            },
            "target": {
                "fname": "hf_dataset.h5",
                "field": "kx",
            },
        },
        "downscale": {
            "file": "vlf_dataset.h5",
            "inpfield": "ss",
            "scale": 4,
            "tgtfield": "ss_down"
        }
    },
    "lf": {
        "train": {
            "nsamp": 2,
            "input": {
                "fname": "lf_dataset.h5",
                "field": "ss_down",
            },
            "target": {
                "fname": "hf_dataset.h5",
                "field": "kx",
            },
        },
        "test": {
            "nsamp": 1,
            "input": {
                "fname": "lf_dataset.h5",
                "field": "ss_down",
            },
            "target": {
                "fname": "hf_dataset.h5",
                "field": "kx",
            },
        },
        "downscale": {
            "file": "lf_dataset.h5",
            "inpfield": "ss",
            "scale": 2,
            "tgtfield": "ss_down"
        }
    },
    "hf": {
        "train": {
            "nsamp": 2,
            "input": {
                "fname": "hf_dataset.h5",
                "field": "ss",
            },
            "target": {
                "fname": "hf_dataset.h5",
                "field": "kx",
            },
        },
        "test": {
            "nsamp": 1,
            "input": {
                "fname": "hf_dataset.h5",
                "field": "ss",
            },
            "target": {
                "fname": "hf_dataset.h5",
                "field": "kx",
            },
        },
    },
}


USE_DATA: int = core.USE_HF_AND_LF

# Phases specifications (NOTE: please specify for each phase!)
PHASES_SPECS = [
    {
        "train": {"inp": "vlf", "tgt": "vlf"},
        "test":  {"inp": "vlf", "tgt": "vlf"},
        "data": {
            "nepochs": 200,
            "lr":  1e-04,
            "wd":  1e-04,
            "f":   0.1,
            "mlr": 1e-07
        }
    },
    {
        "train": {"inp": "lf", "tgt": "lf"},
        "test":  {"inp": "lf", "tgt": "lf"},
        "data": {
            "nepochs": 200,
            "lr":  1e-04,
            "wd":  1e-04,
            "f":   0.1,
            "mlr": 1e-07
        }
    },
    {
        "train": {"inp": "lf", "tgt": "lf"},
        "test":  {"inp": "lf", "tgt": "lf"},
        "data": {
            "nepochs": 200,
            "lr":  1e-04,
            "wd":  1e-04,
            "f":   0.1,
            "mlr": 1e-07
        }
    },
    {
        "train": {"inp": "hf", "tgt": "hf"},
        "test":  {"inp": "hf", "tgt": "hf"},
        "data": {
            "nepochs": 200,
            "lr":  0.5e-04,
            "wd":  1e-04,
            "f":   0.1,
            "mlr": 1e-07
        }
    },
    {
        "train": {"inp": "hf", "tgt": "hf"},
        "test":  {"inp": "hf", "tgt": "hf"},
        "data": {
            "nepochs": 200,
            "lr":  0.5e-04,
            "wd":  1e-04,
            "f":   0.1,
            "mlr": 1e-07
        }
    },
]


# NOTE: do not modify
if (USE_DATA == core.USE_HF_AND_LF):
    N_PHASES = len(PHASES_SPECS)
else:
    N_PHASES = 1 # LF/HF only



__default_parameters : dict[int, core.np.ndarray] = {
    core.TEST_PARAMETER_LR:   core.np.zeros(N_PHASES, dtype=float),
    core.TEST_PARAMETER_WD:   core.np.zeros(N_PHASES, dtype=float),
    core.TEST_PARAMETER_F:    core.np.zeros(N_PHASES, dtype=float),
    core.TEST_PARAMETER_MLR:  core.np.zeros(N_PHASES, dtype=float),
    core.TEST_PARAMETER_NONE: core.np.zeros(N_PHASES, dtype=float), 
}

def __populate_def_params() -> None:
    global __default_parameters

    for i in range(N_PHASES):
        __default_parameters[core.TEST_PARAMETER_LR][i]   = PHASES_SPECS[i]["data"]["lr"]
        __default_parameters[core.TEST_PARAMETER_WD][i]   = PHASES_SPECS[i]["data"]["wd"]
        __default_parameters[core.TEST_PARAMETER_F][i]    = PHASES_SPECS[i]["data"]["f"]
        __default_parameters[core.TEST_PARAMETER_MLR][i]  = PHASES_SPECS[i]["data"]["mlr"]
        __default_parameters[core.TEST_PARAMETER_NONE][i] = PHASES_SPECS[i]["data"]["nepochs"]
    return


#  ######################
#  POSTPROCESS

POSTPROC_BASE_DIR : str = core.CWD + MODEL_TYPE_STR + "/MODEL/results/LF_HF_training/default_run"

SAVE_FIGS : bool = True
SHOW_FIGS : bool = False

SAVE_FORMATS : dict[int, bool] = {
   core.FORMAT_PNG: True,
   core.FORMAT_PDF: True,
   core.FORMAT_SVG: False
}

# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************
#
#   Network Architecture
#
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



class _DenseLayer(nn.Sequential):  
   
   def __init__(self, in_features, growth_rate, drop_rate=0, bn_size=4,
               bottleneck=False):
      super(_DenseLayer, self).__init__()
      if bottleneck and in_features > bn_size * growth_rate:
         self.add_module('norm1', nn.BatchNorm2d(in_features))
         self.add_module('relu1', nn.ReLU(inplace=True))
         self.add_module('conv1', nn.Conv2d(in_features,
                                            bn_size * growth_rate,
                                            kernel_size=1, stride=1, bias=False))
         self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
         self.add_module('relu2', nn.ReLU(inplace=True))
         self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                           kernel_size=3, stride=1, padding=1, bias=False))
      else:
         self.add_module('norm1', nn.BatchNorm2d(in_features))
         self.add_module('relu1', nn.ReLU(inplace=True))
         self.add_module('conv1', nn.Conv2d(in_features, growth_rate,
                           kernel_size=3, stride=1, padding=1, bias=False))
      self.drop_rate = drop_rate

   def forward(self, x):
      y = super(_DenseLayer, self).forward(x)
      if self.drop_rate > 0:
         y = F.dropout2d(y, p=self.drop_rate, training=self.training)
      z = torch.cat([x, y], 1)
      return z


class _DenseBlock(nn.Sequential):  
   def __init__(self, num_layers, in_features, growth_rate, drop_rate,
               bn_size=4, bottleneck=False):
      super(_DenseBlock, self).__init__()
      for i in range(num_layers):
         layer = _DenseLayer(in_features + i * growth_rate, growth_rate,
                             drop_rate=drop_rate, bn_size=bn_size,
                             bottleneck=bottleneck)
         self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):  
   def __init__(self, in_features, out_features, encoding=True, drop_rate=0.,
               last=False, out_channels=3, outsize_even=True):
      super(_Transition, self).__init__()
      self.add_module('norm1', nn.BatchNorm2d(in_features))
      self.add_module('relu1', nn.ReLU(inplace=True))
      if encoding:
         self.add_module('conv1', nn.Conv2d(in_features, out_features,
                                            kernel_size=1, stride=1,
                                            padding=0, bias=False))
         if drop_rate > 0:
            self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
         self.add_module('norm2', nn.BatchNorm2d(out_features))
         self.add_module('relu2', nn.ReLU(inplace=True))
         self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                            kernel_size=3, stride=2,
                                            padding=1, bias=False))
         if drop_rate > 0:
            self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
      else:
         # decoding, transition up
         if last:
            ks = 4
            out_convt = nn.ConvTranspose2d(out_features, out_channels,
                              kernel_size=ks, stride=2, padding=1, bias=False)
         else:
            out_convt = nn.ConvTranspose2d(out_features,
                                           out_features,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1,
                                           output_padding=0,
                                           bias=False)

         self.add_module('conv1', nn.Conv2d(in_features, out_features,
                                            kernel_size=1, stride=1,
                                            padding=0, bias=False))
         if drop_rate > 0:
            self.add_module('dropout1', nn.Dropout2d(p=drop_rate))

         self.add_module('norm2', nn.BatchNorm2d(out_features))
         self.add_module('relu2', nn.ReLU(inplace=True))
         self.add_module('convT2', out_convt)
         if drop_rate > 0:
            self.add_module('dropout2', nn.Dropout2d(p=drop_rate))



class DenseED(nn.Module):  
   def __init__(self, in_channels, out_channels, blocks, growth_rate=16,
               num_init_features=64, bn_size=4, drop_rate=0, outsize_even=True,
               bottleneck=False):
      """
      Args:
         in_channels (int): number of input channels
         out_channels (int): number of output channels
         blocks: list (of odd size) of integers
         growth_rate (int): K
         num_init_features (int): the number of feature maps after the first
               conv layer
         bn_size: bottleneck size for number of feature maps (not useful...)
         bottleneck (bool): use bottleneck for dense block or not
         drop_rate (float): dropout rate
         outsize_even (bool): if the output size is even or odd (e.g.
               65 x 65 is odd, 64 x 64 is even)

      """
      super(DenseED, self).__init__()
      self.out_channels = out_channels

      if len(blocks) > 1 and len(blocks) % 2 == 0:
         ValueError('length of blocks must be an odd number, but got {}'
                    .format(len(blocks)))
      enc_block_layers = blocks[: len(blocks) // 2]
      dec_block_layers = blocks[len(blocks) // 2:]
      self.features = nn.Sequential()
      # First convolution ================
      self.features.add_module('in_conv',
                  nn.Conv2d(in_channels, num_init_features,
                           kernel_size=5, stride=2, padding=2, bias=False))

      # Encoding / transition down ================
      num_features = num_init_features
      for i, num_layers in enumerate(enc_block_layers):
         block = _DenseBlock(num_layers=num_layers,
                             in_features=num_features,
                             bn_size=bn_size, growth_rate=growth_rate,
                             drop_rate=drop_rate, bottleneck=bottleneck)
         self.features.add_module('encblock%d' % (i + 1), block)
         num_features = num_features + num_layers * growth_rate

         trans = _Transition(in_features=num_features,
                             out_features=num_features // 2,
                             encoding=True, drop_rate=drop_rate)
         self.features.add_module('down%d' % (i + 1), trans)
         num_features = num_features // 2

      # Decoding / transition up ==============
      for i, num_layers in enumerate(dec_block_layers):
         block = _DenseBlock(num_layers=num_layers,
                             in_features=num_features,
                             bn_size=bn_size, growth_rate=growth_rate,
                             drop_rate=drop_rate, bottleneck=bottleneck)
         self.features.add_module('decblock%d' % (i + 1), block)
         num_features += num_layers * growth_rate

         # if this is the last decoding layer is the output layer
         last_layer = True if i == len(dec_block_layers) - 1 else False

         trans = _Transition(in_features=num_features,
                             out_features=num_features // 2,
                             encoding=False, drop_rate=drop_rate,
                             last=last_layer, out_channels=out_channels,
                             outsize_even=outsize_even)
         self.features.add_module('up%d' % (i + 1), trans)
         num_features = num_features // 2

   def forward(self, x):
      y = self.features(x)
      y = F.softplus(y.clone(), beta=5)
      return y

   def _num_parameters_convlayers(self):
      n_params, n_conv_layers = 0, 0
      for name, param in self.named_parameters():
         if 'conv' in name:
            n_conv_layers += 1
         n_params += param.numel()
      return n_params, n_conv_layers

   def _count_parameters(self):
      n_params = 0
      for name, param in self.named_parameters():
         print(name)
         print(param.size())
         print(param.numel())
         n_params += param.numel()
         print('num of parameters so far: {}'.format(n_params))

   def reset_parameters(self, verbose=False):
      for module in self.modules():
         # pass self, otherwise infinite loop
         if isinstance(module, self.__class__):
            continue
         if 'reset_parameters' in dir(module):
            if callable(module.reset_parameters):
               module.reset_parameters()
               if verbose:
                  print("Reset parameters in {}".format(module))


class DenseED_phase1(nn.Module):
   def __init__(self, model):
      '''
      Modifies the DenseED model to remove the "up2" layer and
      adds a temporary layer to force output to match
      dimensions of IFS [64x64]

      Input:
         model: model as described by DenseED
      '''
      super(DenseED_phase1, self).__init__()
      self.features = nn.Sequential()
      self.features.add_module('in_conv', model.features.in_conv)
      self.features.add_module('encblock1', model.features.encblock1)
      self.features.add_module('down1', model.features.down1)
      self.features.add_module('decblock1', model.features.decblock1)
      self.features.add_module('up1', model.features.up1)
      self.features.add_module('decblock2', model.features.decblock2)
      self.features.add_module('up2', model.features.up2)


   def forward(self, x):
      y = self.features(x)
      y = F.softplus(y.clone(), beta=5)
      return y

   def _num_parameters_convlayers(self):
      n_params, n_conv_layers = 0, 0
      for name, param in self.named_parameters():
         if 'conv' in name:
            n_conv_layers += 1
         n_params += param.numel()
      return n_params, n_conv_layers

   def _count_parameters(self):
      n_params = 0
      for name, param in self.named_parameters():
         print(name)
         print(param.size())
         print(param.numel())
         n_params += param.numel()
         print('num of parameters so far: {}'.format(n_params))

   def reset_parameters(self, verbose=False):
      for module in self.modules():
         # pass self, otherwise infinite loop
         if isinstance(module, self.__class__):
            continue
         if 'reset_parameters' in dir(module):
            if callable(module.reset_parameters):
               module.reset_parameters()
               if verbose:
                  print("Reset parameters in {}".format(module))




# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************
#
#   Main Local Functions and API calls
#
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************


def __getModelOrig():
   '''
   This function DEFINES the original model from which
   all the other ones (different phases) are derived
   '''
   return DenseED(MODEL_NUM_INPUT_CHANNELS,
                  MODEL_NUM_OUTPUT_CHANNELS,
                  blocks            = MODEL_BLOCKS_NUM_LAYERS,
                  growth_rate       = MODEL_GROWTH_RATE,
                  drop_rate         = MODEL_DROP_RATE,
                  bn_size           = MODEL_BOTTLENECK_SIZE,
                  num_init_features = MODEL_NUM_INITIAL_FEATURES,
                  bottleneck        = MODEL_USE_BOTTLENECKS).to(__device)


def __getModelAtPhaseN(base_model, phase_id : int):
   '''
   This function DEFINES how models are constructed at each PHASE.
   '''
   dense_obj = None
   if (phase_id == 1):
      dense_obj = DenseED_phase1(base_model).to(__device)
   elif (phase_id == 2):
      dense_obj = base_model
   elif (phase_id == 3):
      dense_obj = base_model
   elif (phase_id == 4):
      dense_obj = base_model
   elif (phase_id == 5):
      dense_obj = base_model   
   else:
      pass
   return dense_obj


def __getModelReadyToTrain(model, phase_id : int):
   if (phase_id == 1):
      return

   elif (phase_id == 2):
      for param in model.parameters():
         # If you want to freeze part of your model and train the rest,
         # you can set requires_grad of the parameters you want to freeze to False.
         param.requires_grad = False
      for param in model.features.decblock2.parameters():
         param.requires_grad = True
      for param in model.features.up2.parameters():
         param.requires_grad = True

   elif (phase_id == 3):
      for param in model.parameters():  # Unfreeze all weights
         param.requires_grad = True
   
   elif (phase_id == 4):
      for param in model.parameters():
         param.requires_grad = False
      for param in model.features.decblock2.parameters():
         param.requires_grad = True
      for param in model.features.up2.parameters():
         param.requires_grad = True

   elif (phase_id == 5):
      for param in model.parameters():  # Unfreeze all weights
         param.requires_grad = True
   return



def handle_hdf5_input_data(data: core.np.ndarray):
    return core.np.transpose(data, DATA_INPUT_TRANSPOSE)


def handle_hdf5_output_data(data: core.np.ndarray):
    newdata = core.np.transpose(data, DATA_OUTPUT_TRANSPOSE)
    newdata *= 10**14
    return newdata


def _run(rundir : str = None) -> None:
    
    core.setHDF5DataDirectory(HDF5_DIR)

    # NOTE: before proceeding, add downscaled data to HDF5 files if necessary
    if False:
        for k in MF_SPECS.keys():
           if ("downscale" in MF_SPECS[k]):
                 down_data = MF_SPECS[k]["downscale"]
                 core.add_downscaled_data_to_hdf5file(
                    down_data["file"],
                    down_data["inpfield"],
                    down_data["scale"],
                    down_data["tgtfield"]
                 )

    # Construct array of HDF5 data for each phase
    # NOTE: here the N of PHASES is implicitly defined
    phase_data = []

    for iphase in range(N_PHASES):
        mf_lev_train_inp = PHASES_SPECS[iphase]["train"]["inp"]
        mf_lev_train_tgt = PHASES_SPECS[iphase]["train"]["tgt"]
        mf_lev_test_inp  = PHASES_SPECS[iphase]["test"]["inp"]
        mf_lev_test_tgt  = PHASES_SPECS[iphase]["test"]["tgt"]
        
        data = {}
        data["train"] = core.SimData(MF_SPECS[mf_lev_train_inp]["train"]["input"],
                                     MF_SPECS[mf_lev_train_inp]["train"]["nsamp"],
                                     MF_SPECS[mf_lev_train_tgt]["train"]["target"],
                                     MF_SPECS[mf_lev_train_tgt]["train"]["nsamp"])
        data["train"].input.set_dims(NN_INPUT_DIMS)
        data["train"].target.set_dims(NN_TARGET_DIMS)
        data["train"].input.set_data_callback_fct(handle_hdf5_input_data)
        data["train"].target.set_data_callback_fct(handle_hdf5_output_data)

        data["test"] = core.SimData(MF_SPECS[mf_lev_test_inp]["test"]["input"],
                                    MF_SPECS[mf_lev_test_inp]["test"]["nsamp"],
                                    MF_SPECS[mf_lev_test_tgt]["test"]["target"],
                                    MF_SPECS[mf_lev_test_tgt]["test"]["nsamp"])
        data["test"].input.set_dims(NN_INPUT_DIMS)
        data["test"].target.set_dims(NN_TARGET_DIMS)
        data["test"].input.set_data_callback_fct(handle_hdf5_input_data)
        data["test"].target.set_data_callback_fct(handle_hdf5_output_data)

        phase_data.append(data)


    core.setReductionType(REDUCTION_TYPE)
    core.tlmf.setProblemDimensions(NN_INPUT_DIMS, NN_TARGET_DIMS)

    core.setModelOrigCallback(__getModelOrig)
    core.setGetBaseModelCallback(__getModelAtPhaseN)
    core.setModelReadyCallback(__getModelReadyToTrain)

    core.setBatchSize(BATCH_SIZE)
    if (rundir is None or rundir=="src"):
        core.setRunDir(MODEL_BASE_DIR + "results")
    else:
        core.setRunDir(MODEL_TYPE_DIR + "/" + rundir + "/results")
    core.setDataUsage(USE_DATA)

    __populate_def_params()
    core.bindDefaultParamValues(__default_parameters)

    core.setNPhases(N_PHASES)
    core.setEpochs(__default_parameters[core.TEST_PARAMETER_NONE])

    if (TEST_PARAMETER == core.TEST_PARAMETER_NONE):
        core.forceNReps(N_REPS)
    else:
        core.testParameter(TEST_PARAMETER)
    core.setNRuns(N_RUNS)
    core.setModelsSaving(True)

    core.run(phase_data, PHASES_SPECS)
    pass



def _postproc():
   core.showFigures(SHOW_FIGS)
   core.saveFigures(SAVE_FIGS, SAVE_FORMATS)
   core.postProc(POSTPROC_BASE_DIR)



def main():
   argc = len(sys.argv)

   # If called only as "main.py" assume "RUN" subcommand
   if (argc < 2):
      _run()
   else:
      command_str = sys.argv[1]
      if (command_str == "-run"):
         if (argc == 2):
            _run()
         else:
            _run(sys.argv[2])
      elif (command_str == "-postprocess"):
         _postproc()
      else:
         print(f"\n  -- [ERROR]:  unonkwn command  \"{command_str}\"")
         exit(-1)
   return


if __name__ == '__main__':
   main()


