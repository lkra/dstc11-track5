import itertools

import pandas as pd
from pytorch_beam_search import autoregressive


def train_question_beam_search(full_questions, ngram_size=5):
    # Create vocabulary and examples
    full_questions['question_tokens'] = full_questions['reference_response_question'].apply(
        lambda x: [f"{t} " for t in x.split(" ")])
    corpus = list(itertools.chain(*full_questions['question_tokens'].to_list()))

    # An Index object represents a mapping from the vocabulary to integers (indices) to feed into the models
    index = autoregressive.Index(corpus)
    n_grams = [corpus[i:ngram_size + i] for i in range(len(corpus))[:-ngram_size + 1]]

    # Create and train the model
    X = index.text2tensor(n_grams)
    model = autoregressive.TransformerEncoder(index)  # just a PyTorch model
    model.fit(X)  # basic method included

    return model, index


def beam_search_question(model, index, phrase_to_complete="would you like"):
    # Generate new predictions
    new_examples = [[f"{t} " for t in phrase_to_complete.split(" ")]]
    X_new = index.text2tensor(new_examples)

    # every element in predictions is the list of candidates for each example
    predictions, log_probabilities = autoregressive.beam_search(model, X_new)
    output = [index.tensor2text(p) for p in predictions]
    print(output)

    return output


def most_frequent_question():
    full_questions = pd.read_csv(f'./../data_analysis/output/scores_per_reference_question_train.csv')

    full_questions.sort_values(by="num_samples", ascending=False, inplace=True)

    return full_questions.loc[full_questions.index[0], 'reference_response_question']
