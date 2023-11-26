import csv
import matplotlib.pyplot as plt

# List of CSV file paths
folders = ['replication/DeepPerf/Data/DeepPerf_results/',
           'replication/HINNPerf/results_old/',
           'results/',
           'results-seq/']
labels = ['DeepPerf', 
          'HINNPerf', 
          'TransPerf',
          'TransPerf-Seq']
datasets = ['x264',
            'Dune',
            'hipacc',
            'javagc']
colors = ['C0', 'C1', 'C2', 'C3'] 


def load_file(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)

        # Skipping the header row
        next(csv_reader)

        # Initializing lists to store data
        sample_sizes = []
        means = []
        margins = []

        # Reading data from CSV
        for row in csv_reader:
            # Assuming the structure: Sample size, Mean, Margin
            sample_sizes.append(float(row[0]))
            means.append(float(row[1]))
            margins.append(float(row[2]))
    return sample_sizes, means, margins


def load_all_files(folders, dataset_name, labels):
    data = {}
    for idx, folder in enumerate(folders):
        path = folder + 'result_' + dataset_name + '.csv'
        sample_sizes, means, margins = load_file(path)
        data[labels[idx]] = (sample_sizes, means, margins)
    return data
    

def plot_dataset_results(data, dataset_name):
    
    fig, ax = plt.subplots()
    bar_width = 0.15
    
    # We will use these lists to set the x-axis labels
    x_ticks = []
    x_labels = data[labels[-1]][0]
    
    for model_idx, model in enumerate(labels):
        value = data[model]
        y_values = value[1]
        y_margins = value[2]
        num_tests = len(y_values)
        for i in range(num_tests):
            if i == 0:
                ax.bar([i + (model_idx * bar_width)], y_values[i], width=bar_width, 
                       yerr=y_margins[i], capsize=5, label=labels[model_idx], color=colors[model_idx])
            else:
                ax.bar([i + (model_idx * bar_width)], y_values[i], width=bar_width, yerr=y_margins[i], 
                       capsize=5, color=colors[model_idx])
            # Collect x-tick labels
            if model == labels[-1]:  # Only need to do this once
                x_ticks.append(i + 2*bar_width)
    
    # Setting custom x-tick labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    
    plt.grid()
    plt.xlabel('Sample Size')
    plt.ylabel('Mean Relative Error (%)')
    plt.title(f'Mean Relative Error for Dataset: {dataset_name}')
    # Add legend
    plt.legend()
    plt.savefig('plots/' + dataset_name + '.png')


def main():
    for dataset_name in datasets:
        # Load data from CSV files
        data = load_all_files(folders, dataset_name, labels)
        
        # Plot data
        plot_dataset_results(data, dataset_name)
    

if __name__ == '__main__':
    main()
