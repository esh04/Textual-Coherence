import torch
import os
import argparse
from torch.utils.data import Dataset, DataLoader
import torchtext
from collections import Counter
import numpy as np
import pandas as pd
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--file', help='Data to use for training', required=True)
# parser.add_argument('--tokenizer-threshold',
#                     help='Word Count below which the token is treated as other', type=int, default=15)
# parser.add_argument(
#     '--batch-size', help='Training batch size', type=int, default=32)
# parser.add_argument(
#     '--num-workers', help='Workers to use in dataloader', type=int, default=5)
# parser.add_argument(
#     '--num-epochs', help='Number of training epochs', type=int, default=40)
# parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
# parser.add_argument('--embedding-dim',
#                     help='Size of embedding dim', type=int, default=256)
# parser.add_argument(
#     '--hidden-dim', help='Size of lstm hidden dim', type=int, default=512)
# parser.add_argument('--dropout', help='dropout value', type=float, default=0.2)
# parser.add_argument(
#     '--num-layers', help='Number of LSTM layers', type=int, default=4)
# parser.add_argument(
#     '--seq-len', help='Numbers of tokens in each example', type=int, default=25)
# parser.add_argument('--grad-clip', help='Gradient Clipping',
#                     type=float, default=0.5)
# parser.add_argument('--checkpoint-file-name',
#                     help='Name of checkpoint file', default='best_language_model.pth')
# parser.add_argument('--lr-drop-factor',
#                     help='Factor by which to reduce lr', default=0.5, type=float)
# parser.add_argument('--language', help='Tokenizer language',
#                     default='en', type=str)


class Tokenizer:
    def __init__(self, file, threshold=5):
        self.file = file
        self.data = pd.read_csv(file)
        self.threshold = threshold

    def preprocess(self):
        tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en')
        tokens = []
        for text in self.data['text'].tolist():
            tokens.append(tokenizer(text))
        counter = Counter()
        for line in tokens:
            for word in line:
                counter[word] += 1
        # print(len(counter.items()), len(counter.most_common()))

        # remove all words that have frequency less than threshold
        # counter_threshold = {k:v for k,v in counter.items() if v >= self.threshold}

        # create mappings
        # mapper = {word:idx+1 for idx,word in enumerate(counter_threshold.keys())}
        # inverse_mapper = {idx+1:word for idx,word in enumerate(counter_threshold.keys())}

        # sos_idx = len(counter_threshold.keys())
        # eos_idx = len(counter_threshold.keys()) + 1
        # other_idx = len(counter_threshold.keys()) + 2

        # mapped_tokens = []

        # for line in tokens:
        #     mapped_line = [sos_idx]
        #     for word in line:
        #       # map words to their mappings and to other otherwise
        #         mapped_line.append(mapper.get(word, other_idx))
        #     mapped_line.append(eos_idx)
        #     mapped_tokens.append(mapped_line)

        # inverse_mapper[other_idx] = "__OTHER__"
        # inverse_mapper[sos_idx] = "__SOS__"
        # inverse_mapper[eos_idx] = "__EOS__"
        # inverse_mapper[0] = "__PADDING__"

        mapper = {word[0]: idx+1 for idx,
                  word in enumerate(counter.most_common())}
        inverse_mapper = {idx+1: word[0] for idx,
                          word in enumerate(counter.most_common())}

        # sos_idx = len(counter_threshold.keys())
        # eos_idx = len(counter_threshold.keys()) + 1
        other_idx = len(counter.keys())

        mapped_tokens = []

        for line in tokens:
            mapped_line = []
            for word in line:
              # map words to their mappings and to other otherwise
                mapped_line.append(mapper.get(word, other_idx))
            mapped_tokens.append(mapped_line)

        # inverse_mapper[other_idx] = "__OTHER__"
        # inverse_mapper[sos_idx] = "__SOS__"
        # inverse_mapper[eos_idx] = "__EOS__"
        # inverse_mapper[0] = "__PADDING__"

        return mapped_tokens, inverse_mapper

    def save_tokens(self):
        mapped_tokens, inverse_mapper = self.preprocess()

        self.data.to_pickle(self.file+".pkl")
        with open('mapped_tokens_' + self.file + '.pkl', 'wb') as f:
            pickle.dump(mapped_tokens, f)
        with open('inverse_mapper' + self.file + '.pkl', 'wb') as f:
            pickle.dump(inverse_mapper, f)

        return


if __name__ == '__main__':
    args = parser.parse_args()
    t = Tokenizer(args.file)
    t.save_tokens()
