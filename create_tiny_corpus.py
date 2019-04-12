import numpy as np


corpus_file = 'librispeech-lm-norm.txt'
probability = 0.01
tiny_corpus_file = 'tiny_corpus.txt'

tiny_corpus = []
with open(corpus_file, 'r') as f:
    for line in f:
        if np.random.rand() < probability:
            tiny_corpus.append(line)

with open(tiny_corpus_file, 'w') as f:
    f.write(''.join(tiny_corpus))