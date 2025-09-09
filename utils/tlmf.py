import os
import numpy as np
import torch
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import copy



def getTorchAvailableDevice():
   '''
   Sets torch device to GPU NVIDIA CUDA if avaliable.
   Otherwise, uses CPU (not as performant as GPUs for this kind of computations).
   '''
   USE_GPU = True
   if USE_GPU and torch.cuda.is_available():
      device = torch.device('cuda')
   else:
      device = torch.device('cpu')
   return device


__device = getTorchAvailableDevice()


__input_dims  = None
__target_dims = None


def setProblemDimensions(input, target) -> None:
   global __input_dims, __target_dims
   __input_dims  = input
   __target_dims = target




def getDataLoader(data, mf_data, bs):
    """
    data [n_sims, n_channels, Nx, Ny]
    """

    input_name_test   = 'input_test'
    input_name_train  = 'input_train'
    target_name_test  = 'target_test'
    target_name_train = 'target_train'

    nx = mf_data["train"].target.get_size()[0]
    ny = mf_data["train"].target.get_size()[1]

    # Create a randperm for the entire n. of simulations
    n   = len(data[input_name_train])
    idx = np.array(torch.randperm(n))

    # Then, take first [0:n_train] indexes for train, [n_train:n_train+n_test] for test
    num_train = mf_data["train"].target.n_samp
    num_test  = mf_data["test"].target.n_samp
    idx_train = idx[0 : num_train]
    idx_test  = idx[num_train : num_train+num_test]

    input_data_train  = data[input_name_train][idx_train]
    target_data_train = data[target_name_train][idx_train]
    input_data_test   = data[input_name_test][idx_test]
    target_data_test  = data[target_name_test][idx_test]

    input_data_train  = torch.Tensor(np.reshape(input_data_train,  (num_train, *mf_data["train"].input.dims)))
    input_data_test   = torch.Tensor(np.reshape(input_data_test,   (num_test,  *mf_data["test"].input.dims)))
    target_data_train = torch.Tensor(np.reshape(target_data_train, (num_train, mf_data["train"].target.dims, nx, ny)))
    target_data_test  = torch.Tensor(np.reshape(target_data_test,  (num_test,  mf_data["test"].target.dims, nx, ny)))

    train_dataset = torch.utils.data.TensorDataset(input_data_train, target_data_train)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=bs) 
    test_dataset  = torch.utils.data.TensorDataset(input_data_test, target_data_test)
    test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=bs)

    return train_loader, test_loader




__N_PHASES : int = 0


def setNPhases_(n):
   global __N_PHASES
   __N_PHASES = n
   return


__iphase : int = 1


def setPhaseIndex(idx : int) -> None:
   global __iphase
   __iphase = idx
   return


__do_save_models      : bool = False
__model_save_base_dir : str  = None


def saveModels(flag):
   global __do_save_models
   __do_save_models = flag
   return



def setModelSaveBaseDir(d):
   global __model_save_base_dir
   __model_save_base_dir = d + os.path.sep
   if (not os.path.exists(__model_save_base_dir)):
      os.mkdir(__model_save_base_dir)
   return


__do_write_test_model: bool = False
__loss_curves: np.ndarray = None


def initLossCurvesArr(nphases : int) -> None:
   global __loss_curves
   __loss_curves = np.empty_like(np.empty((nphases, 2)), dtype=object)
   return



__reduction_type_str: str = 'sum'  # By default we SUM
__is_reduction_sum: bool  = True


def setReductionTypeStr(reduction_str : str) -> None:
   global __reduction_type_str, __is_reduction_sum

   __reduction_type_str = reduction_str
   if (__reduction_type_str != 'sum'):
      __is_reduction_sum = False
   return


__DO_SAVE_TEST_ONLY: bool = True
__DO_SAVE_LAST_BATCH_ONLY: bool = True

__LAST_NUM_EPOCHS_SAVE: int = 1
__epoch_threshold: int = 100000    # NOTE: big number so that if not modified, conditions fail
__i_epoch_save: int = 0

__target_train_saved: np.ndarray = None
__output_train_saved: np.ndarray = None
__target_test_saved:  np.ndarray = None
__output_test_saved:  np.ndarray = None



def testModel_(model, test_loader, epoch):
   global __do_plot_model_test_error
   global __i_epoch_save, __epoch_threshold
   global __target_test_saved
   global __output_test_saved

   n_dataset_        = len(test_loader.dataset)
   n_out_pixels_test = n_dataset_ * test_loader.dataset[0][1].numel()

   model.eval()

   loss  = 0.
   offst = 0
   for _, (input, target) in enumerate(test_loader):
      input, target = input.to(__device), target.to(__device)

      with torch.no_grad():
         output = model(input)

      loss += F.mse_loss(output, target, reduction=__reduction_type_str).item() 

      if (not __DO_SAVE_LAST_BATCH_ONLY and epoch > __epoch_threshold):
          offst_end = offst + target.shape[0]
          __target_test_saved[offst : offst_end, :, :, :, __i_epoch_save] = target.cpu().detach().numpy()
          __output_test_saved[offst : offst_end, :, :, :, __i_epoch_save] = output.cpu().detach().numpy()
          offst = offst_end

   if (__DO_SAVE_LAST_BATCH_ONLY and epoch > __epoch_threshold):
      torch.save(target.cpu().detach().numpy(), __model_save_base_dir + 'TargetTest_Phase{:d}.pth'.format(__iphase))
      torch.save(output.cpu().detach().numpy(), __model_save_base_dir + 'OutputTest_Phase{:d}.pth'.format(__iphase))

   if (__is_reduction_sum):
      rmse_test = np.sqrt(loss / n_out_pixels_test)
   else:
      rmse_test = np.sqrt(loss)
   return rmse_test




def trainModel_(train_loader, test_loader, \
      reps, n_epochs, log_interval, model_orig, lr, wd, factor, min_lr):

   global __i_epoch_save, __epoch_threshold
   global __target_train_saved
   global __output_train_saved
   global __target_test_saved
   global __output_test_saved
   global __LAST_NUM_EPOCHS_SAVE


   # Not only we save last batch, but also last epoch as well.
   if (__DO_SAVE_LAST_BATCH_ONLY):
       __LAST_NUM_EPOCHS_SAVE = 1


   __epoch_threshold = n_epochs - __LAST_NUM_EPOCHS_SAVE


   if (not __DO_SAVE_LAST_BATCH_ONLY):
      single_dataset_shape = train_loader.dataset[0][1].shape # [95, 60, 7]

      output_target_save_shape = np.concatenate(
              ([len(test_loader.dataset)], single_dataset_shape, np.array([__LAST_NUM_EPOCHS_SAVE])))
      __target_test_saved  = np.zeros(output_target_save_shape, dtype=np.float64)
      __output_test_saved  = np.zeros(output_target_save_shape, dtype=np.float64)

      if (not __DO_SAVE_TEST_ONLY):
         output_target_save_shape = np.concatenate(
                 ([len(train_loader.dataset)], single_dataset_shape, np.array([__LAST_NUM_EPOCHS_SAVE])))
         __target_train_saved = np.zeros(output_target_save_shape, dtype=np.float64)
         __output_train_saved = np.zeros(output_target_save_shape, dtype=np.float64)


   n_out_pixels_train = len(train_loader.dataset) * train_loader.dataset[0][1].numel()


   rmse_best = 10**6  # initial values, just have to be large.
   for rep in range(reps):

      rmse_train = []
      rmse_test  = []
      model      = copy.deepcopy(model_orig)
      optimizer  = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
      scheduler  = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=10,
                                     verbose=True, threshold=0.0001, threshold_mode='rel',
                                     cooldown=0, min_lr=min_lr, eps=1e-08)

      __i_epoch_save = 0
      for epoch in range(1, n_epochs+1):

         model.train() 

         mse   = 0.
         offst = 0
         for _, (input, target) in enumerate(train_loader):

            input, target = input.to(__device), target.to(__device)

            model.zero_grad(set_to_none=False)
            output = model(input)
            loss   = F.l1_loss(output, target, reduction=__reduction_type_str)
            loss.backward(gradient=None, retain_graph=None, create_graph=False, inputs=None) 
            optimizer.step()  

            mse += F.mse_loss(output, target, reduction=__reduction_type_str).item() 

            if (not __DO_SAVE_LAST_BATCH_ONLY and epoch > __epoch_threshold):
                offst_end = offst + target.shape[0]
                __target_train_saved[offst : offst_end, :, :, :, __i_epoch_save] = target.cpu().detach().numpy()
                __output_train_saved[offst : offst_end, :, :, :, __i_epoch_save] = output.cpu().detach().numpy()
                offst = offst_end


         if (__is_reduction_sum):
            rmse = np.sqrt(mse / n_out_pixels_train)
         else:
            rmse = np.sqrt(mse)

         scheduler.step(rmse)

         if (epoch % log_interval == 0 or epoch == n_epochs):
            rmse_train.append(rmse)
            rmse_t = testModel_(model, test_loader, epoch)
            rmse_test.append(rmse_t)

         if (not __DO_SAVE_LAST_BATCH_ONLY and epoch > __epoch_threshold):
            __i_epoch_save += 1

      rmse_test_last_10_values_      = rmse_test[-10:]
      rmse_test_last_10_values_mean_ = np.mean(rmse_test_last_10_values_)

      if (rmse_test_last_10_values_mean_ < rmse_best):
         model_best = copy.deepcopy(model)
         rmse_best  = rmse_test_last_10_values_mean_

   if (__do_save_models and __model_save_base_dir is not None):
      if (__DO_SAVE_LAST_BATCH_ONLY):
         if (not __DO_SAVE_TEST_ONLY):
            torch.save(target.cpu().detach().numpy(), __model_save_base_dir + 'TargetTrain_Phase{:d}.pth'.format(__iphase))
            torch.save(output.cpu().detach().numpy(), __model_save_base_dir + 'OutputTrain_Phase{:d}.pth'.format(__iphase))
      else:
         if (not __DO_SAVE_TEST_ONLY):
            torch.save(__target_train_saved, __model_save_base_dir + 'TargetTrain_Phase{:d}.pth'.format(__iphase))
            torch.save(__output_train_saved, __model_save_base_dir + 'OutputTrain_Phase{:d}.pth'.format(__iphase))
         torch.save(__target_test_saved,  __model_save_base_dir + 'TargetTest_Phase{:d}.pth'.format(__iphase))
         torch.save(__output_test_saved,  __model_save_base_dir + 'OutputTest_Phase{:d}.pth'.format(__iphase))

   __loss_curves[__iphase-1, 0] = np.array(rmse_train)
   __loss_curves[__iphase-1, 1] = np.array(rmse_test)

   return model_best, rmse_best


def getLossData_():
   return __loss_curves




