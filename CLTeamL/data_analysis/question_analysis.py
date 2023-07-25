import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(font_scale=0.6)


def plot_question_distribution(df1, df2, df3):
    df = pd.concat([df1, df2, df3])
    groups = df.groupby(['ref_response_question']).agg(num_samples=('ref_response_question', 'count')).sort_values(
        by='num_samples', ascending=False)
    groups.reset_index(inplace=True)
    groups = groups[0:50]

    m = sns.barplot(data=groups, x='ref_response_question', y='num_samples', palette='colorblind')
    m.set_xticklabels(m.get_xticklabels(), rotation=90, va='center_baseline', size=5)
    plt.subplots_adjust(bottom=0.4, left=0.3)
    plt.savefig(f'./output/distributions_questions.png', bbox_inches='tight')
    plt.show()

    # m = sns.histplot(data=df, x='ref_response_question',
    #                 # stat="percent", discrete=True , kde=True, element='step'
    #                 )
    # sns.kdeplot(data=df1, x='num_samples')
    # sns.rugplot(data=groups)
    # plt.subplots_adjust(bottom=0.4, left=0.3)
    # plt.savefig(f'./output/distributions_questions.png', bbox_inches='tight')
    # plt.show()


def get_unique_questions(dataset='train'):
    # read in data frame
    df = pd.read_csv(f'output/analysis_{dataset}.csv')
    df = df[df['target']]

    # Find questions
    df['ref_response_question'] = df['ref_response_question'].str.lower()
    groups = df.groupby(['ref_response_question']).agg(num_samples=('ref_response_question', 'count')).sort_values(
        by='num_samples', ascending=False)

    groups.to_csv(f"output/ref_question_{dataset}.csv")

    return df


def main():
    df_train = get_unique_questions('train')
    df_val = get_unique_questions('val')
    df_test = get_unique_questions('test')
    plot_question_distribution(df_train, df_val, df_test)


if __name__ == "__main__":
    main()
