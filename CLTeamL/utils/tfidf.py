import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_tfidf(trigrams, keep_top=0.05):
    trigrams.sort_values(by="count", ascending=False, inplace=True)

    # Take only the most frequent
    cutoff = trigrams.quantile(1 - keep_top)["count"]
    trigrams = trigrams[trigrams["count"] > cutoff]

    # Calculate tfidf
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(trigrams['trigram'])
    df = pd.DataFrame(vectors.todense().tolist(), columns=vectorizer.get_feature_names())
