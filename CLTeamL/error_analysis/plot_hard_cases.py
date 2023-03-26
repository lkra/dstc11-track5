import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main(args):
    approach = args.prediction_file.rsplit(".", 1)[0]
    ref_columns = ['turn_nr',
                   'ref_know_nr', 'ref_know_avg_sentiment',
                   'ref_response_length', 'ref_response_sent_nr',
                   'ref_response_summary_sentiment',
                   # 'case_type'
                   ]

    # read in data frame
    df = pd.read_csv(f'./output/errors_{approach}.csv')
    df = df[df['target']]

    # Report overall statistics
    general_stats = df.describe()
    general_stats.to_csv(f'./output/stats_{approach}.csv')

    # Select best/worse 50
    worse = df.sort_values(by=['bleu']).head(50)
    best = df.sort_values(by=['bleu'], ascending=False).head(50)

    worse_stats = worse.describe().loc['mean']
    best_stats = best.describe().loc['mean']
    general_stats = general_stats.loc['mean']

    compare_df = pd.concat([best_stats[ref_columns], general_stats[ref_columns], worse_stats[ref_columns]], axis=1)
    compare_df.columns = ['best', 'general', 'worse']
    compare_df = compare_df.T

    # Plot
    axes = compare_df.plot.line(subplots=True)

    # flatten the array
    axes = axes.flat  # .ravel() and .flatten() also work

    # extract the figure object to use figure level methods
    fig = axes[0].get_figure()
    fig.subplots_adjust(hspace=1)

    # iterate through each axes to use axes level methods
    for ax in axes:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 2), frameon=False)

    plt.savefig(f'./output/error_trend_{approach}.png')

    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', default='baseline.rg.bart-base.json', type=str,
                        help="Which file with predictions to compare to the ground truth")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
