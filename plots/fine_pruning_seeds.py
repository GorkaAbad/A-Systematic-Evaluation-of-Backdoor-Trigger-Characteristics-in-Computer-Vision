import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set()
sns.set_theme(style="whitegrid", font_scale=1)
plt.rcParams["font.family"] = "Times New Roman"

SEED = 2
DATASET = 'CIFAR10'
ATTACK = 'WaNet'
PATH = f'./experiments/FinePruning/{ATTACK}'
filename = 'results.csv'

taus = [0.0, 0.3, 0.6, 0.9]
# models = df['model'].unique()
models = ['vgg', 'resnet', 'googlenet', 'alexnet']

print(models)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

markers = ['o', 's', 'v', 'd']
linestyles = ['-', '--', '-.', ':']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
legend_labels = ['VGG', 'ResNet', 'GoogleNet', 'AlexNet']
n_experiments = 3
for model in models:
    list_pr_asr = []
    list_pr_acc = []
    for pr in taus:
        list_acc = []
        list_asr = []
        for experiment in range(n_experiments):
            # Read the results from the csv file
            path_load = os.path.join(
                PATH, str(experiment), filename)
            df = pd.read_csv(path_load)

            # Filter the results
            df = df[(df['model'] == model)]
            df = df[(df['pruning_rate'] == pr)]
            list_acc.append(df['fine-pruned_clean_acc'].values[0])

            list_asr.append(df['fine-pruned_bk_acc'].values[0])
        list_pr_asr.append(list_asr)
        list_pr_acc.append(list_acc)
    
    mean_asr = np.mean(list_pr_asr, axis=1)
    std_asr = np.std(list_pr_asr, axis=1)
    min_asr = np.min(list_pr_asr, axis=1)
    max_asr = np.max(list_pr_asr, axis=1)

    mean_acc = np.mean(list_pr_acc, axis=1)
    std_acc = np.std(list_pr_acc, axis=1)
    min_acc = np.min(list_pr_acc, axis=1)
    max_acc = np.max(list_pr_acc, axis=1)

    err_asr = np.array([std_asr, std_asr])
    err_acc = np.array([std_acc, std_acc])
    ax.errorbar(taus, mean_asr, yerr=err_asr,
                    marker=markers[models.index(model)], alpha=0.8,
                    markersize=8, label=model, color=colors[models.index(model)],
                    linestyle=linestyles[0])

    ax.errorbar(taus, mean_acc, yerr=err_acc,
                    marker=markers[models.index(model)], alpha=0.8,
                    markersize=8, color=colors[models.index(model)],
                    linestyle=linestyles[1])

    print(model)
    print(err_asr, err_acc)
    print(
        df[['pruning_rate', 'fine-pruned_clean_acc', 'fine-pruned_bk_acc']])

plt.xticks(taus)
plt.ylim(0.0, 1.01)
sns.despine(left=True, right=True)
plt.xlabel('Pruning Rate')
plt.ylabel('Accuracy')
#plt.legend()
handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, legend_labels, loc='upper center',
            bbox_to_anchor=(0.5, 0.02), fancybox=False, shadow=False,
            ncol=len(models))

plt.savefig(f'./plots/{ATTACK}_{DATASET}_fine_pruning.pdf',
            bbox_inches='tight')
