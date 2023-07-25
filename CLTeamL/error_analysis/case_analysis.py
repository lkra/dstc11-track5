import argparse

import pandas as pd

from utils.helpers import group_metrics_by


def main(args):
    approach = args.prediction_file.rsplit(".", 1)[0]

    # read in data frame
    df = pd.read_csv(f'./output/error_analysis_{approach}.csv')
    df = df[df['target']]

    # group by USER UTTERANCE dialogue act
    groups = group_metrics_by(df, 'user_utterance_dialogue_act')
    groups.to_csv(f"output/scores_per_act_{approach}.csv")

    # group by REFERENCE RESPONSE optional questions
    df['ref_response_question'] = df['ref_response_question'].str.lower()
    groups = group_metrics_by(df, 'ref_response_question')
    groups.to_csv(f"output/scores_per_ref_question_{approach}.csv")


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
