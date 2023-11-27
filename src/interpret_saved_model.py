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
    
    batch_size = 1
    print('Batch size is: {}'.format(batch_size))

    # Set seed
    seed_init = seed_generator(data_gen.sys_name, n_exp)
    seed = seed_init * n_exp

    torch.manual_seed(123)
    np.random.seed(123)

    x_train, y_train, x_valid, y_valid, _ = data_gen.get_train_valid_samples(n_exp, seed)

    train_dataset = PerfDataset(x_train, y_train, model.d_model)
    valid_dataset = PerfDataset(x_valid, y_valid, model.d_model)
    combined_dataset = ConcatDataset([train_dataset, valid_dataset])

    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    result_sys = []
    ig_all_attributions = []
    ig_all_convergence_deltas = []
    dl_all_attributions = []
    exp_counter = 0

    ig = IntegratedGradients(model)
    dl = DeepLift(model)

    for batch_idx, (x_test, y_test) in enumerate(dataloader):
        exp_counter += 1
        if exp_counter > n_exp:
            break
        print("Experiment: {}".format(exp_counter))

        start = time.time()

        model.eval()
        prediction = model.forward(x_test)
        print('Model Prediction:', prediction)

        # Calculate model error
        error = torch.abs(prediction - y_test)
        print('Model Error:', error)

        input = x_test
        baseline = torch.zeros_like(input)
        
        ig_attributions, ig_convergence_delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
        print('IG Attributions:', ig_attributions)
        print('IG Convergence Delta:', ig_convergence_delta)

        dl_attributions = dl.attribute(input, baseline, target=0, return_convergence_delta=True)
        print('DL Attributions:', dl_attributions)

        ig_all_attributions.append(ig_attributions)
        ig_all_convergence_deltas.append(ig_convergence_delta)

        dl_all_attributions.append(ig_attributions)


    ig_attributions_mean = sum(ig_all_attributions) / n_exp
    ig_convergence_deltas_mean = sum(ig_all_convergence_deltas) / n_exp

    dl_attributions_mean = sum(dl_all_attributions) / n_exp

    result = dict()
    result["IG"] = ig_attributions_mean
    result["DL"] = dl_attributions_mean
    result_arr = np.asarray(result)

    print("\n")
    print('Finish experimenting for system {} with {} experiments.'.format(sys_name, n_exp))
    print('IG Attributions Mean:', ig_all_attributions)
    print('IG Convergence Deltas Mean:', ig_convergence_deltas_mean)
    print('DL Attributions Mean:', dl_attributions_mean)
    print("\n")


    # Save the result statistics to a csv file after each sample
    # Save the raw results to an .npy file
    print('Save results to the results directory ...')
    filename = 'results/interpret_' + sys_name + '.csv'
    np.savetxt(filename, result_arr, fmt="%f", delimiter=",", header="IG, DL")
    print('Save the statistics to file ' + filename + ' ...')