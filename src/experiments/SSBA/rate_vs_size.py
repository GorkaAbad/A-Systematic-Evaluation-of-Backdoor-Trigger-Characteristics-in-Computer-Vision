import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


def main():

    sns.set()
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams["font.family"] = "Times New Roman"

    # Read the reuslts from the csv file
    path = './'
    path_save = './figs/'
    filename = 'results.csv'

    models = ['vgg', 'resnet', 'googlenet', 'alexnet']
    legend_labels = ['VGG', 'ResNet', 'GoogleNet', 'AlexNet']
    datasets = ['CIFAR10', 'TinyImageNet']
    epsilon_values = [0.005, 0.01, 0.015, 0.02]
    markers = ['o', 's', 'D', 'X']
    linestyles = ['-', '--', '-.', ':']
    n_experiments = 5

    fig, axs = plt.subplots(nrows=len(datasets), ncols=1, figsize=(12, 8),
                            sharex=True, sharey=True)
    column = 0
    row = 0
    for idx, ax in enumerate(axs.flat):
        #if column >= len(trigger_size_values):
        #    column = 0
        #    row += 1

        dataset = datasets[row]

        for model in models:
            list_eps_asr = []
            for epsilon in epsilon_values:
                list_asr = []
                for experiment in range(0, n_experiments):
                    # Read the results from the csv file
                    path_load = os.path.join(
                        str(experiment), filename)
                    df = pd.read_csv(path_load)

                    # Filter the results
                    df = df[(df['dataset'] == dataset) &
                            (df['model'] == model) &
                            (df['epsilon'] == epsilon)]

                    list_asr.append(df['bk_acc'].values[0]*100)

                list_eps_asr.append(list_asr)

            mean = np.mean(list_eps_asr, axis=1) # if we use boxplot, we can here replace the mean with median value, \
                                                 # and the error to be the median-min, max-median, (all values are calculated without the outliers)
            std = np.std(list_eps_asr, axis=1)
            min = np.min(list_eps_asr, axis=1)
            max = np.max(list_eps_asr, axis=1)

            #err = np.array([mean - min, max - mean])
            err = np.array([std, std])
            ax.errorbar(epsilon_values, mean, yerr=err,
                        marker=markers[models.index(model)], alpha=0.8,
                        markersize=8, label=model,
                        linestyle=linestyles[models.index(model)])

        column += 1

    # Set the labels
    #for ax, col in zip(axs[0], trigger_size_values):
    #    ax.set_title('Trigger size = {}'.format(col))

    #import ipdb; ipdb.set_trace()
    #for ax, row in zip(axs[:, 0], datasets):
    #    ax.set_ylabel(row.upper(), rotation=90, size='large')

    # Set the legend showing the models with the corresponding marker
    #handles, labels = axs[0, 0].get_legend_handles_labels()
    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(handles,
               legend_labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.1), fancybox=False, shadow=False, ncol=len(models))

    # Set the x and y labels
    fig.supxlabel(r'$\epsilon$')
    fig.supylabel('ASR (%)')

    # Set the ticks
    for ax in axs.flat:
        ax.set_xticks(epsilon_values)
        ax.set_yticks(np.arange(0, 110, 20))
        ax.set_ylim(0, 100)

    # Set the grid
    sns.despine(left=True)
    plt.tight_layout()
    path_save = os.path.join(
        path_save, f'rate_vs_size_{args.trigger_color}_{args.pos}.pdf')
    plt.savefig(path_save)
    # plt.show()


if __name__ == '__main__':
    main()
