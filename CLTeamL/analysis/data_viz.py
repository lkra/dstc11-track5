import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=0.6)

# read in data frame
df = pd.read_pickle('output/analysis.pkl')

# change data types from object to int
print(df.dtypes)
df = df.infer_objects()
print(f'new data types: {df.dtypes}')
df = df.astype({'question_dialogue_act': object})
print(f'question dialogue act type: {df["question_dialogue_act"].dtypes}')

# calculate correlations
matrix = df.corr().round(2)

# plot heatmap
sns.heatmap(matrix, vmax=1, vmin=-1, center=0, cmap="Blues", annot=True, annot_kws={"fontsize":8})
plt.subplots_adjust(bottom=0.28, left=0.18)
plt.show()
plt.savefig('correlations.png')

# create pair plot
# sns.pairplot(df, hue='prediction_domain', size=2.5)
# print('save visualsss')
# plt.savefig('prediction_domain.png')