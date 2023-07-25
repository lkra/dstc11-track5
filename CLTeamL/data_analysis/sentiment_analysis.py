import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(font_scale=0.6)

COLUMNS_TO_PLOT = ['ref_know_avg_sentiment', 'ref_know_std_sentiment',
                   'ref_response_summary_sentiment',
                   ]


def plot_distribution():
    # read in data frame
    df_train = pd.read_csv(f'./output/analysis_train.csv')
    df_val = pd.read_csv(f'./output/analysis_val.csv')
    df_test = pd.read_csv(f'./output/analysis_test.csv')
    df = pd.concat([df_train, df_val, df_test])

    # Filter
    df = df[df['target']]
    df = df[COLUMNS_TO_PLOT]
    df.columns = ['average knowledge sentiment', 'std knowledge sentiment', 'response sentiment']

    # plot
    sns.violinplot(data=df, palette='coolwarm')
    plt.subplots_adjust(bottom=0.4, left=0.3)
    plt.savefig(f'./output/distributions_sentiment.png', bbox_inches='tight')


def plot_correlations(dataset='train'):
    # read in data frame
    df = pd.read_csv(f'./output/analysis_{dataset}.csv')

    # change data types from object to int
    df = df.infer_objects()
    df = df.astype({'user_utterance_dialogue_act': "category"})

    # Report overall statistics
    general_stats = df.describe()
    general_stats.to_csv(f'./output/stats_{dataset}.csv')

    # Filter
    df = df[df['target']]
    df = df[COLUMNS_TO_PLOT]

    # calculate correlations
    matrix = df.corr().round(2)

    # plot heatmap
    m = sns.heatmap(matrix, vmax=1, vmin=-1, center=0, cmap="Blues", annot=True, annot_kws={"fontsize": 6})
    plt.subplots_adjust(bottom=0.4, left=0.3)
    plt.savefig(f'./output/correlations_{dataset}.png', bbox_inches='tight')

    return matrix


def plot_side_by_side(matrix1, matrix2, matrix3):
    # Create axis
    fig, axs = plt.subplots(ncols=4, gridspec_kw=dict(width_ratios=[6, 6, 6, 0.4]))

    # Plot both matrix
    m1 = sns.heatmap(matrix1, ax=axs[0], cbar=False,
                     vmax=1, vmin=-1, center=0, cmap="Blues", annot=True, annot_kws={"fontsize": 6})
    m1.set_yticklabels(m1.get_yticklabels(), rotation=90, va='center', size=5)
    sns.heatmap(matrix2, ax=axs[1], cbar=False, yticklabels=False,
                vmax=1, vmin=-1, center=0, cmap="Blues", annot=True, annot_kws={"fontsize": 6})
    sns.heatmap(matrix3, ax=axs[2], cbar=False, yticklabels=False,
                vmax=1, vmin=-1, center=0, cmap="Blues", annot=True, annot_kws={"fontsize": 6})

    # Plot color bar
    fig.colorbar(axs[1].collections[0], cax=axs[3])

    # Prettify
    plt.subplots_adjust(bottom=0.4, left=0.3)
    plt.savefig(f'./output/correlations_both.png', bbox_inches='tight')


def main():
    matrix_train = plot_correlations('train')
    matrix_val = plot_correlations('val')
    matrix_test = plot_correlations('test')
    plot_side_by_side(matrix_train, matrix_val, matrix_test)

    plot_distribution()


if __name__ == "__main__":
    main()
