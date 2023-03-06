import dataclasses
from typing import Optional, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from baseline.analysis.utils.constants import _LABELS, _LABEL2ID


@dataclasses.dataclass
class DialogueAct:
    """
    Information about a Dialogue Act.
    """
    type: str
    value: str
    confidence: Optional[float]


class MidasDialogTagger(object):
    def __init__(self, model_path):
        self._device = torch.device('cpu')

        self._tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self._model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(_LABELS))
        self._model.load_state_dict(torch.load(model_path, map_location=self._device))
        self._model.to(self._device)

        self._label2id = _LABEL2ID
        self._id2label = _LABELS
        self._dialog = [""]  ### initialise with an empty string to get started

    def _tokenize(self, strings):
        return self._tokenizer(strings, padding=True, return_tensors='pt').to(self._device)

    def _encode_labels(self, labels):
        for label in labels:
            if label not in self._label2id:
                self._label2id[label] = len(self._label2id)
                self._id2label[len(self._id2label)] = label

    # Only needed for training
    def fit(self, data, epochs=4, batch_size=32, lrate=1e-5):
        # Preprocess turns and index labels
        strings = [t0 + self._tokenizer.sep_token + t1 for t0, t1, _ in data]
        labels = [l for _, _, l in data]

        X = [self._tokenize(strings[i:i + batch_size]) for i in range(0, len(strings), batch_size)]
        y = [self._encode_labels(labels[i:i + batch_size]) for i in range(0, len(labels), batch_size)]

        # Setup optimizer and objective function
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lrate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            losses = []

            for X_batch, y_batch in tqdm(zip(X, y)):
                y_pred = self._model(**X_batch)
                loss = criterion(y_pred.logits, y_batch)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(np.mean(losses))

    def extract_dialogue_act(self, utterance: str) -> List[DialogueAct]:
        if not utterance:
            return []

        turn0 = self._dialog[-1]
        self._dialog.append(utterance)
        string = turn0 + self._tokenizer.sep_token + utterance
        X = self._tokenize([string])
        y = self._model(**X).logits.cpu().detach().numpy()
        label = self._id2label[np.argmax(y[0])]
        score = y[0][np.argmax(y[0])]
        dialogueAct = DialogueAct(type="MIDAS", value=label, confidence=float(score))
        ### Trying to normalize the scores, any ideas?
        # max = np.max(y[0])
        # min = np.min(y[0])
        # scaled_scores = np.array([(x-min)/(max-min) for x in y[0]])
        return [dialogueAct]
