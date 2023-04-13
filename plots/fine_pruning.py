import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()
sns.set_theme(style="whitegrid", font_scale=1)
plt.rcParams["font.family"] = "Times New Roman"

SEED = 0
DATASET = 'CIFAR10'
ATTACK = 'BADNETS'
PATH = f'./experiments/FinePruning/{SEED}/results.csv'

df = pd.read_csv(PATH)
df = df[df['dataset'] == DATASET]
df = df[df['seed'] == SEED]

taus = df['pruning_rate'].unique()
models = df['model'].unique()

print(models)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

markers = ['o', 's', 'v', 'd']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for model in models:
    df_model = df[df['model'] == model]
    # Sort by pruning rate
    df_model = df_model.sort_values(by='pruning_rate')

    marker = markers.pop()
    color = colors.pop()

    print(model)
    print(
        df_model[['pruning_rate', 'fine-pruned_clean_acc', 'fine-pruned_bk_acc']])
    # Plot the fine_pruned_clean_acc and fine_pruned_bk_acc with the same color and marker but different line style
    ax.plot(df_model['pruning_rate'], df_model['fine-pruned_clean_acc'],
            label=f'{model.upper()}', linestyle='-', marker=marker, color=color)
    ax.plot(df_model['pruning_rate'],
            df_model['fine-pruned_bk_acc'],  linestyle='--', marker=marker, color=color)

plt.xticks(taus)
sns.despine(left=True, right=True)
plt.xlabel('Pruning Rate')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig(f'./plots/{ATTACK}_{DATASET}_fine_pruning.pdf',
            bbox_inches='tight')
