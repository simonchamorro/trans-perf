import argparse
import numpy as np
import time
import torch

from data_preprocess import system_samplesize, seed_generator, DataPreproc
from args import list_of_param_dicts
from model import Transperf
from model_runner import ModelRunner
from dataset import PerfDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':

    # Get system name from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("system_name",
                        help="name of system to be evaluated: Apache, LLVM, x264, BDBC, BDBJ, SQL, Dune, hipacc, hsmgp, javagc, sac",
                        type=str)
    parser.add_argument("-ne", "--number_experiment",
                        help="number of experiments per sample size (integer)",
                        type=int)
    parser.add_argument("-ss", "--sample_size",
                        help="sample size to be evaluated (integer)",
                        type=int)
    args = parser.parse_args()

    # System to be evaluated:
    sys_name = args.system_name
    print(sys_name)

    # Number of experiments per sample size
    if args.number_experiment is not None:
        n_exp = int(args.number_experiment)
    else:
        n_exp = 5

    # The sample size to be evaluated
    if args.sample_size is not None:
        sample_size_all = []
        sample_size_all.append(int(args.sample_size))
    else:
        sample_size_all = list(system_samplesize(sys_name))

    data_gen = DataPreproc(sys_name)
    src_sample = data_gen.get_train_valid_samples(1, 1)
    src_shape = src_sample[0].shape[1]
    nhead = 8
    model = Transperf(input_size=src_shape, nhead=nhead)
    if src_shape % nhead != 0:
        d_model = (src_shape // nhead)*nhead + nhead
    else:
        d_model = src_shape
    
    batch_size = 256 if sys_name == 'javagc' else 32
    print('Batch size is: {}'.format(batch_size))
    runner = ModelRunner(data_gen, model, batch_size=batch_size)
    result_sys = []

    # Sample sizes need to be investigated
    for idx in range(len(sample_size_all)):
        N_train = sample_size_all[idx]
        rel_error_mean = []

        if (N_train >= data_gen.all_sample_num):
            raise AssertionError("Sample size can't be larger than whole data")

        for m in range(1, n_exp+1):
            print("Experiment: {}".format(m))

            start = time.time()

            config = dict(
                input_dim = [data_gen.config_num],
                gnorm = [False, True],
                lr = [0.001],
                epochs = [1000],
            )
            config_list = list_of_param_dicts(config)
            
            abs_error_val_min = float('inf')
            best_config = None
            for con in config_list:
                abs_error_train, abs_error_val = runner.train(con, N_train, n_exp, m)
                if abs_error_val_min > abs_error_val:
                    abs_error_val_min = abs_error_val
                    best_config = con
            
            print(best_config)
            
            mean_error, rel_error = runner.test(best_config, N_train, n_exp, m)
            rel_error_mean.append(rel_error)

        result = dict()
        result["N_train"] = N_train
        result["rel_error_mean"] = rel_error_mean
        result_sys.append(result)


        # Compute some statistics: mean, confidence interval
        result = []
        for i in range(len(result_sys)):
            temp = result_sys[i]
            sd_error_temp = np.sqrt(np.var(temp['rel_error_mean'], ddof=1))
            ci_temp = 1.96*sd_error_temp/np.sqrt(len(temp['rel_error_mean']))
            result_exp = [temp['N_train'], np.mean(temp['rel_error_mean']), ci_temp]
            result.append(result_exp)
        result_arr = np.asarray(result)

        print('Finish experimenting for system {} with sample size {}.'.format(sys_name, N_train))
        print('Mean prediction relative error (%) is: {:.2f}, Margin (%) is: {:.2f}'.format(np.mean(rel_error_mean), ci_temp))

        # Save the result statistics to a csv file after each sample
        # Save the raw results to an .npy file
        print('Save results to the results directory ...')
        filename = 'results/result_' + sys_name + '.csv'
        np.savetxt(filename, result_arr, fmt="%f", delimiter=",", header="Sample size, Mean, Margin")
        print('Save the statistics to file ' + filename + ' ...')

        filename = 'results/result_' + sys_name + '_AutoML_veryrandom.npy'
        np.save(filename, result_sys)
        print('Save the raw results to file ' + filename + ' ...')