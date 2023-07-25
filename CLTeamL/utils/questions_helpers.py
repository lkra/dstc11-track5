from pathlib import Path

import pandas as pd


def most_frequent_question():
    print(Path(f'./../error_analysis/output/scores_per_ref_question_baseline.rg.bart-base.csv').resolve())
    full_questions = pd.read_csv(f'./../error_analysis/output/scores_per_ref_question_baseline.rg.bart-base.csv')

    full_questions.sort_values(by="num_samples", ascending=False, inplace=True)

    return full_questions.loc[full_questions.index[0], 'ref_response_question']
