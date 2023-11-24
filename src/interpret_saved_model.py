import argparse
import numpy as np
import time
import torch

from data_preprocess import system_samplesize, seed_generator, DataPreproc
from args import list_of_param_dicts
from model import Transperf
from interpretable_model_runner import InterpretableModelRunner
from dataset import PerfDataset
from torch.utils.data import DataLoader

from captum.attr import IntegratedGradients


if __name__ == '__main__':

    # Get system name from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("system_name",
                        help="name of system to be evaluated: Apache, LLVM, x264, BDBC, BDBJ, SQL, Dune, hipacc, hsmgp, javagc, sac",
                        type=str)
    parser.add_argument("-ne", "--number_experiment",
                        help="number of experiments per sample size (integer)",
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

    data_gen = DataPreproc(sys_name)
    src_sample = data_gen.get_train_valid_samples(1, 1)
    src_shape = src_sample[0].shape[1]
    nhead = 8
    model = Transperf(input_size=src_shape, nhead=nhead, load_model="models\\" + sys_name + '\\model.pt')
    if src_shape % nhead != 0:
        d_model = (src_shape // nhead)*nhead + nhead
    else:
        d_model = src_shape
    
    batch_size = 256 if sys_name == 'javagc' else 32
    print('Batch size is: {}'.format(batch_size))
    runner = InterpretableModelRunner(data_gen, model, batch_size=batch_size)
    result_sys = []
    all_attributions = []
    all_convergence_deltas = []

    for m in range(1, n_exp+1):
        print("Experiment: {}".format(m))

        start = time.time()

        model.eval()

        ig = IntegratedGradients(model)
        input = x_test
        baseline = torch.zeros_like(input)
        attributions, convergence_delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
        print('IG Attributions:', attributions)
        print('Convergence Delta:', convergence_delta)

        all_attributions.append(attributions)
        all_convergence_deltas.append(convergence_delta)

    result = dict()
    result["attributions_mean"] = all_attributions / n_exp
    result["convergence_deltas_mean"] = all_convergence_deltas / n_exp
    result_sys.append(result)


    # Compute some statistics: mean, confidence interval

    print('Finish experimenting for system {} with {} experiments.'.format(sys_name, n_exp))
    print('IG Attributions Mean:', result["attributions_mean"])
    print('Convergence Deltas Mean:', result["convergence_deltas_mean"])

    # Save the result statistics to a csv file after each sample
    # Save the raw results to an .npy file
    # print('Save results to the results directory ...')
    # filename = 'results/result_' + sys_name + '.csv'
    # np.savetxt(filename, result_arr, fmt="%f", delimiter=",", header="Sample size, Mean, Margin")
    # print('Save the statistics to file ' + filename + ' ...')

    # filename = 'results/result_' + sys_name + '_AutoML_veryrandom.npy'
    # np.save(filename, result_sys)
    # print('Save the raw results to file ' + filename + ' ...')