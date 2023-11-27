import argparse
import numpy as np
import time
import torch

from data_preprocess import system_samplesize, seed_generator, DataPreproc
from args import list_of_param_dicts
from model import Transperf
from dataset import PerfDataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation


if __name__ == '__main__':

    # Get system name from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("system_name",
                        help="name of system to be evaluated: Apache, LLVM, x264, BDBC, BDBJ, SQL, Dune, hipacc, hsmgp, javagc, sac",
                        type=str)
    args = parser.parse_args()

    # System to be evaluated:
    sys_name = args.system_name
    print(sys_name)

    data_gen = DataPreproc(sys_name)
    src_sample = data_gen.get_train_valid_samples(1, 1)
    src_shape = src_sample[0].shape[1]
    nhead = 8
    model = Transperf(input_size=src_shape, nhead=nhead, load_model="models\\" + sys_name + '\\model.pt')
    if src_shape % nhead != 0:
        d_model = (src_shape // nhead)*nhead + nhead
    else:
        d_model = src_shape
    
    batch_size = 1
    print('Batch size is: {}'.format(batch_size))

    # Set seed
    seed = seed_generator(1)

    torch.manual_seed(123)
    np.random.seed(123)

    x_all_shape = data_gen.X_all.shape
    print('x_all_shape: {}'.format(x_all_shape))
    
    # Split whole dataset
    x_train, y_train, x_valid, y_valid, _ = data_gen.get_train_valid_samples(x_all_shape[0], seed)
    # x_all = ConcatDataset([x_train, x_valid])

    x_train = PerfDataset(x_train, y_train, model.d_model).x
    x_valid = PerfDataset(x_valid, y_valid, model.d_model).x

    x_train_shape = x_train.shape
    print('x_train_shape: {}'.format(x_train_shape))
    x_valid_shape = x_valid.shape
    print('x_valid_shape: {}'.format(x_valid_shape))

    ig = IntegratedGradients(model)
    ig_nt = NoiseTunnel(ig)
    dl = DeepLift(model)
    gs = GradientShap(model)
    fa = FeatureAblation(model)

    model.eval()

    ig_attr_test = ig.attribute(x_valid, n_steps=50)
    ig_nt_attr_test = ig_nt.attribute(x_valid)
    dl_attr_test = dl.attribute(x_valid)
    gs_attr_test = gs.attribute(x_valid, x_train)
    fa_attr_test = fa.attribute(x_valid)

    result = dict()
    result["IG"] = ig_attr_test
    result["IG_NT"] = ig_nt_attr_test
    result["DL"] = dl_attr_test
    result["GS"] = gs_attr_test
    result["FA"] = fa_attr_test
    result_arr = np.asarray(result)

    print("\n")
    print('Finish feature attribution for system {}.'.format(sys_name))
    print("\n")


    # Save the result statistics to a csv file after each sample
    # Save the raw results to an .npy file
    print('Save results to the results directory ...')
    filename = 'results/feature_attribution_' + sys_name + '.csv'
    np.savetxt(filename, result_arr, fmt="%f", delimiter=",", header="IG, IG_NT, DL, GS, FA")
    print('Save the statistics to file ' + filename + ' ...')