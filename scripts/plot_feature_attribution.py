import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from src.data_preprocess import DataPreproc
from src.model import Transperf


def load_file(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)

        # Skipping the header row
        next(csv_reader)

        # Initializing lists to store data
        ig_attr_test = []
        ig_nt_attr_test = []
        dl_attr_test = []
        gs_attr_test = []
        fa_attr_test = []

        # Reading data from CSV
        for row in csv_reader:
            ig_attr_test.append(float(row[0]))
            ig_nt_attr_test.append(float(row[1]))
            dl_attr_test.append(float(row[2]))
            gs_attr_test.append(float(row[3]))
            fa_attr_test.append(float(row[4]))

        data = dict()
        data["IG"] = ig_attr_test
        data["IG_NT"] = ig_nt_attr_test
        data["DL"] = dl_attr_test
        data["GS"] = gs_attr_test
        data["FA"] = fa_attr_test

    return data
    

def plot_dataset_results(data, model, dataset_name, feature_names, src_shape):
    
    x_axis_data = np.arange(src_shape.shape[1])
    x_axis_data_labels = list(map(lambda idx: feature_names[idx], x_axis_data))

    ig_attr_test_sum = data["IG"].pop.detach().numpy().sum(0)
    ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)

    ig_nt_attr_test_sum = data["IG_NT"].pop.detach().numpy().sum(0)
    ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / np.linalg.norm(ig_nt_attr_test_sum, ord=1)

    dl_attr_test_sum = data["DL"].pop.detach().numpy().sum(0)
    dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)

    gs_attr_test_sum = data["GS"].pop.detach().numpy().sum(0)
    gs_attr_test_norm_sum = gs_attr_test_sum / np.linalg.norm(gs_attr_test_sum, ord=1)

    fa_attr_test_sum = data["FA"].pop.detach().numpy().sum(0)
    fa_attr_test_norm_sum = fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)

    lin_weight = model.lin1.weight[0].detach().numpy()
    y_axis_lin_weight = lin_weight / np.linalg.norm(lin_weight, ord=1)

    width = 0.14
    legends = ['Int Grads', 'Int Grads w/SmoothGrad','DeepLift', 'GradientSHAP', 'Feature Ablation', 'Weights']

    plt.figure(figsize=(20, 10))

    ax = plt.subplot()
    ax.set_title('Comparing input feature importances across multiple algorithms and learned weights')
    ax.set_ylabel('Attributions')

    FONT_SIZE = 16
    plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
    plt.rc('axes', titlesize=FONT_SIZE)       # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE)       # fontsize of the x and y labels
    plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

    ax.bar(x_axis_data, ig_attr_test_norm_sum, width, align='center', alpha=0.8, color='#eb5e7c')
    ax.bar(x_axis_data + width, ig_nt_attr_test_norm_sum, width, align='center', alpha=0.7, color='#A90000')
    ax.bar(x_axis_data + 2 * width, dl_attr_test_norm_sum, width, align='center', alpha=0.6, color='#34b8e0')
    ax.bar(x_axis_data + 3 * width, gs_attr_test_norm_sum, width, align='center',  alpha=0.8, color='#4260f5')
    ax.bar(x_axis_data + 4 * width, fa_attr_test_norm_sum, width, align='center', alpha=1.0, color='#49ba81')
    ax.bar(x_axis_data + 5 * width, y_axis_lin_weight, width, align='center', alpha=1.0, color='grey')
    ax.autoscale_view()
    plt.tight_layout()

    ax.set_xticks(x_axis_data + 0.5)
    ax.set_xticklabels(x_axis_data_labels)

    plt.legend(legends, loc=3)
    plt.show()
    plt.savefig('plots/feature_attribution_' + dataset_name + '.png')


def main():
    # Get system name from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("system_name",
                        help="name of system to be evaluated: Apache, LLVM, x264, BDBC, BDBJ, SQL, Dune, hipacc, hsmgp, javagc, sac",
                        type=str)
    args = parser.parse_args()

    # System to be evaluated:
    sys_name = args.system_name
    print(sys_name)

    path = 'results/feature_attribution_' + sys_name + '.csv'
    data = load_file(path, sys_name)
    
    data_gen = DataPreproc(sys_name)
    src_sample = data_gen.get_train_valid_samples(1, 1)
    src_shape = src_sample[0].shape[1]
    nhead = 8
    model = Transperf(input_size=src_shape, nhead=nhead, load_model="models\\" + sys_name + '\\model.pt')

    feature_names = data_gen.get_feature_names()
    
    # Plot data
    plot_dataset_results(data, model, sys_name, feature_names, src_shape)
    

if __name__ == '__main__':
    main()
