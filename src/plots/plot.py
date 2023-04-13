import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


def show_everything(path, filename, n_experiments):
    """Show all the measurements for debugging purposes."""
    data = []
    for experiment in range(0, n_experiments):
        path_load = os.path.join(path, str(experiment), filename)
        data.append(pd.read_csv(path_load))

    df = pd.concat(data)
    df = df.drop("id", axis=1)
    df = df.drop("seed", axis=1)
    df = df.drop("target_label", axis=1)

    # Print the average values and the standard deviation for debugging.
    print(f"{10*'='} Average Values {10*'='}")
    print(df.groupby(["dataset", "model", "epsilon"]).mean().reset_index())
    print(f"{10*'='} Standard Deviation {10*'='}")
    print(df.groupby(["dataset", "model", "epsilon"]).std().reset_index())


def plot(attack, debug=False):
    """Plot the results for SSBA attack."""
    sns.set()
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams["font.family"] = "Times New Roman"

    # Read the reuslts from the csv file
    path = f'../../experiments/{attack}/'
    path_save = './'
    filename = 'results.csv'

    models = ['vgg', 'resnet', 'googlenet', 'alexnet']
    legend_labels = ['VGG', 'ResNet', 'GoogleNet', 'AlexNet']
    # TODO: TinyImageNet is not correct yet so plot only CIFAR10.
    datasets = ['CIFAR10']
    epsilon_values = [0.005, 0.01, 0.015, 0.02]
    markers = ['o', 's', 'D', 'X']
    linestyles = ['-', '--', '-.', ':']

    if attack == "SSBA":
        n_experiments = 5
    elif attack == "WaNet":
        n_experiments = 3

    fig, axs = plt.subplots(nrows=len(datasets), ncols=1, figsize=(12, 8),
                            sharex=True, sharey=True)

    if debug:
        show_everything(path, filename, n_experiments)

    idx = 0
    dataset = datasets[idx]
    for model in models:
        list_eps_asr = []
        for epsilon in epsilon_values:
            list_asr = []
            for experiment in range(0, n_experiments):
                # Read the results from the csv file
                path_load = os.path.join(
                    path, str(experiment), filename)
                df = pd.read_csv(path_load)

                # Filter the results
                df = df[(df['dataset'] == dataset) &
                        (df['model'] == model) &
                        (df['epsilon'] == epsilon)]

                list_asr.append(df['bk_acc'].values[0]*100)

            list_eps_asr.append(list_asr)

        # if we use boxplot, we can here replace the mean with median value,
        # and the error to be the median-min, max-median, (all values are
        # calculated without the outliers)
        mean = np.mean(list_eps_asr, axis=1)
        std = np.std(list_eps_asr, axis=1)
        min = np.min(list_eps_asr, axis=1)
        max = np.max(list_eps_asr, axis=1)

        #err = np.array([mean - min, max - mean])
        err = np.array([std, std])
        axs.errorbar(epsilon_values, mean, yerr=err,
                     marker=markers[models.index(model)], alpha=0.8,
                     markersize=8, label=model,
                     linestyle=linestyles[models.index(model)])

    axs.set_title(f"Dataset {datasets[0]}")
    handles, labels = axs.get_legend_handles_labels()

    fig.legend(handles, legend_labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.08), fancybox=False, shadow=False,
               ncol=len(models))

    # Set the x and y labels
    fig.supxlabel(r'$\epsilon$', y=0.001)
    fig.supylabel('ASR (%)')

    axs.set_xticks(epsilon_values)
    axs.set_yticks(np.arange(0, 110, 20))
    axs.set_ylim(0, 101)

    # Set the grid
    sns.despine(left=True)
    plt.tight_layout()
    path_save = os.path.join(path_save, f'{attack}.pdf')
    plt.savefig(path_save)


if __name__ == '__main__':
    plot("SSBA")
    plot("WaNet")
