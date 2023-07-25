import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tiktoken

DATASET = "val"  # val or train

# read in data frame
df = pd.read_csv(f'output/analysis_{DATASET}.csv')
df = df[df['target']]

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

df['ref_response_tokens'] = df['ref_response'].apply(lambda x: len(encoding.encode(x)))

avg_tokens = df['ref_response_tokens'].mean()

print(f'Average tokens per response:{avg_tokens}')

print(df['ref_response_tokens'].describe())

sns.kdeplot(data=df, x="ref_response_tokens")
plt.show()
