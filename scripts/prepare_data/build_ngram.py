import pandas as pd
from tqdm import tqdm

from src_new.utils.vocab import Vocab

vocab = Vocab('testing_vocab.model')

import os

root = 'data/filterd_manifest/'
csvs = os.listdir(root)
csvs = [i for i in csvs if i.endswith('.csv')]

count = 0
with open('lm_corpus.all', 'w') as writer:
    for csv in tqdm(csvs):
        df = pd.read_csv(os.path.join(root, csv))
        target = df.target
        for t in target:
            line = t.strip()
            line = vocab.str2token(line)
            if len(line) >= 3:
                count += 1
                line = ' '.join(line)
                writer.write(line + '\n')
print(f'{count} line written')