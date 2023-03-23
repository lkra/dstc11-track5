# CLTL Team @ DSTC11 2023 Shared Task (Track 5)

See description of the task [here](https://dstc11.dstc.community/tracks).

We are a team of awesome [CLTL](http://www.cltl.nl/) PhDs:

- [Lea Krause](https://lkra.github.io/)
- [Selene Báez Santamaría](https://selbaez.github.io/)
- [Michiel van der Meer](https://liacs.leidenuniv.nl/~meermtvander/)
- [Urja Khurana](https://urjakh.github.io/)

## Installation

Install packages with `pip install -r requirements.txt`. This file may or may not be complete/up-to-date..

## Usage

You will find all scripts and resources related to our team's approaches for this task in the ```CLTeamL``` folder.
Scripts related to either data analysis or error analysis can be found under ```analysis```.

### Data analysis

We process responses to separate it into two parts:

- summaries: this part of the response generally focuses on processing the knowledge items to reply to the last user
  utterance
- questions: this part relates more to dialogue management, where the system may offer help in continuing with the task
  at hand (e.g. reserve a restaurant)

We enrich the dataset with

- dialogue act on each of the utterances in the dialogue context
- sentiment analysis on each knowledge item, and the average across knowledge items
- sentiment analysis on the ground truth responses (only the summary part)

The analysis outputs a csv file under ```CLTeamL/data_analysis/output``` called `analysis_$DATASET$` To run, please do:

```bash
python dataset_analysis.py 
 --dataset train
```

#### Summary analysis

Our intuition is that the summary part of the response must follow a similar sentiment as the knowledge items selected
for creating such response. To calculate the correlations between the average sentiment of the knowledge
items `ref_know_avg_sentiment` and the `ref_response_summary_sentiment`, please run:

```bash
python ngram_analysis.py 
```

#### Question analysis

To extract ngrams and look at the patterns in the questions part of the responses, please run:

```bash
python ngram_analysis.py 
```

### Approaches

#### Prompting

First, please create an account with [OpenAI](https://auth0.openai.com/u/signup). Get your API key and set it as an
environmental variable. For running the *best* approach using GPT3, run:

```bash
python prompting.py 
 --test_set True
 --prompt_style 1
```

#### Manipulate questions

We experimented on removing the question part of the response, or adding it in case it was not produced by the baseline.
In the case of addition, we use the most frequent question, as selected from the train dataset.
To try these approaches run:

```bash
python manipulate_questions.py 
 --test_set True
 --question_style 1
 --question_selection 0
```

### Evaluation

#### Automatic metrics

We use the original scripts for evaluation to get an aggregated performance overview. For this you may run something
like this:

```bash
cd scripts
python scores.py
 --dataset val
 --dataroot data/
 --outfile pred/val/baseline.rg.prompt-style0.json
 --scorefile pred/val/baseline.rg.prompt-style0.score.json
```

#### Error analysis

To compare the output of a specific approach to the ground truth, on an item by item basis, you may run:

```bash
python error_analysis.py
 --dataset val
 --prediction_file baseline.rg.prompt-style0.json
```

To compare the output of a specific type of dialogue act, you may run:

```bash
python case_analysis.py
```
