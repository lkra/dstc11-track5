import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# read in data frame
df = pd.read_pickle("output/analysis.pkl")

print("hi :))")

# create pair plot
sns.pairplot(df, hue='prediction_domain', size=2.5)
# print('save visualsss')
# plt.savefig('prediction_domain.png')