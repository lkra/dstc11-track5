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

### GPT3 models

First, please create an account with [OpenAI](https://auth0.openai.com/u/signup). Get your API key and set it as an
environmental variable. For running the *best* approach using GPT3 (for example on the test set), run:

```bash
python prompting.py --n_shot 4 --prompt_style 5
```

To evaluate a file with predictions, you can do:

```python
from baseline import evaluate_from_file

evaluate_from_file("predictions/dump_prompt_5_results.json")
```


