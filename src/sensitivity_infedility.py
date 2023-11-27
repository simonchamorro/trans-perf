import argparse
from matplotlib import pyplot as plt
import numpy as np
import time
import torch

from data_preprocess import system_samplesize, seed_generator, DataPreproc
from args import list_of_param_dicts
from model import Transperf
from dataset import PerfDataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from captum.attr import Saliency
from captum.metrics import sensitivity_max, infidelity
import csv


if __name__ == '__main__':

    # Get system name from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("system_name",
                        help="name of system to be evaluated: Apache, LLVM, x264, BDBC, BDBJ, SQL, Dune, hipacc, hsmgp, javagc, sac",
                        type=str)
    parser.add_argument("-ss", "--sample_size",
                        help="sample size to be evaluated (integer)",
                        type=int)
    args = parser.parse_args()

    # System to be evaluated:
    sys_name = args.system_name
    print(sys_name)

    # The sample size to be evaluated
    if args.sample_size is not None:
        sample_size = int(args.sample_size)
    else:
        sample_size = 100

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
    seed = seed_generator(sys_name,1)

    torch.manual_seed(123)
    np.random.seed(123)

    x_all_shape = data_gen.X_all.shape
    print('x_all_shape: {}'.format(x_all_shape))
    
    # Split whole dataset
    x_train, y_train, x_valid, y_valid, _ = data_gen.get_train_valid_samples(sample_size, seed)
    # x_train, y_train, x_valid, y_valid, _ = data_gen.get_train_valid_samples(x_all_shape[0], seed)
    # x_all = ConcatDataset([x_train, x_valid])

    x_train = PerfDataset(x_train, y_train, model.d_model).x
    x_valid = PerfDataset(x_valid, y_valid, model.d_model).x

    x_train_shape = x_train.shape
    print('x_train_shape: {}'.format(x_train_shape))
    x_valid_shape = x_valid.shape
    print('x_valid_shape: {}'.format(x_valid_shape))

    input = x_train

    model.eval()

    saliency = Saliency(model)
    saliency_attribution = saliency.attribute(input)

    def perturb_fn(inputs):
       noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float()
       return noise, inputs - noise
    
    # Computes infidelity score for saliency maps
    infid = infidelity(model, perturb_fn, input, saliency_attribution)
    sens = sensitivity_max(saliency.attribute, input, target = 3)

    result = dict()

    result["infidelity"] = infid
    result["sensitivity"] = sens

    print("\n")
    print('Finish feature attribution for system {}.'.format(sys_name))
    print("\n")


    # Save the result statistics to a csv file after each sample
    # Save the raw results to an .npy file
    print('Save results to the results directory ...')
    # filename = 'results/feature_attribution_' + sys_name
    filename = 'results/sens_infid_' + sys_name + '.csv'
    # np.savetxt(filename, np.asarray(result), fmt="%f", delimiter=",", header="IG, IG_NT, DL, GS, FA")
    # np.save(filename, result)


    # Save the statistics to a csv file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.keys())
        writer.writerows(zip(*result.values()))

    print("\n")
    print('Finish feature attribution for system {}.'.format(sys_name))
    print("\n")


    # Save the result statistics to a csv file after each sample
    # Save the raw results to an .npy file
    print('Save results to the results directory ...')
    # filename = 'results/feature_attribution_' + sys_name
    filename = 'results/feature_attribution_' + sys_name + '_' + str(sample_size) + '.csv'
    # np.savetxt(filename, np.asarray(result), fmt="%f", delimiter=",", header="IG, IG_NT, DL, GS, FA")
    # np.save(filename, result)
    

    # Save the statistics to a csv file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.keys())
        writer.writerows(zip(*result.values()))

    print('Statistics saved to file: ' + filename)
    

    def plot_dataset_results(data, model, dataset_name, feature_names, src_shape):
        
        x_axis_data = np.arange(src_shape)
        x_axis_data_labels = list(map(lambda idx: feature_names[idx], x_axis_data))

        ig_attr_test_sum = data["infidelity"].detach().numpy().sum(0)
        ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)

        ig_nt_attr_test_sum = data["sensitivity"].detach().numpy().sum(0)
        ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / np.linalg.norm(ig_nt_attr_test_sum, ord=1)

        width = 0.14
        legends = ['Int Grads', 'Int Grads w/SmoothGrad','DeepLift', 'GradientSHAP', 'Feature Ablation', 'Weights']
        # legends = ['Int Grads', 'Int Grads w/SmoothGrad','DeepLift', 'GradientSHAP', 'Feature Ablation']

        plt.figure(figsize=(20, 10))

        ax = plt.subplot()
        ax.set_title('Comparing input feature importances across multiple algorithms')
        ax.set_ylabel('Attributions')

        FONT_SIZE = 16
        plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
        plt.rc('axes', titlesize=FONT_SIZE)       # fontsize of the axes title
        plt.rc('axes', labelsize=FONT_SIZE)       # fontsize of the x and y labels
        plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

        ax.bar(x_axis_data, ig_attr_test_norm_sum, width, align='center', alpha=0.8, color='#eb5e7c')
        ax.bar(x_axis_data + width, ig_nt_attr_test_norm_sum, width, align='center', alpha=0.7, color='#A90000')
        ax.autoscale_view()
        plt.tight_layout()

        ax.set_xticks(x_axis_data + 0.5)
        ax.set_xticklabels(x_axis_data_labels)

        plt.legend(legends, loc=3)
        # plt.show()
        plt.savefig('plots/sens_infid_' + dataset_name + '_' + str(sample_size) + '.png')

    feature_names = data_gen.get_feature_names()
    for i in range(model.d_model - src_shape):
        pad_string = 'padding' + str(i)
        feature_names = np.append(feature_names, pad_string)

    plot_dataset_results(result, model, sys_name, feature_names, model.d_model)
