import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(font_scale=0.6)

DATASET = 'train'

# read in data frame
df = pd.read_csv(f'./output/analysis_{DATASET}.csv')
df = df[df['target']]

# change data types from object to int
df = df.infer_objects()
df = df.astype({'user_utterance_dialogue_act': "category",
                # 'pred_know_nr': float,
                # 'pred_know_avg_sentiment': float,
                # 'pred_response_length': float
                })

# calculate correlations
matrix = df.corr().round(2)

# plot heatmap
sns.heatmap(matrix, vmax=1, vmin=-1, center=0, cmap="Blues", annot=True, annot_kws={"fontsize": 6})
plt.subplots_adjust(bottom=0.4, left=0.3)
plt.savefig(f'./output/correlations_{DATASET}.png')

# create pair plot
# sns.pairplot(df, hue='prediction_domain', size=2.5)
# print('save visualsss')
# plt.savefig('prediction_domain.png')
