import sys
import os
sys.path.append(os.getcwd())
import sentencepiece as spm
from tqdm import tqdm
from src_new.utils.corpus_tokenizer import is_chinese
from collections import Counter


if __name__ == '__main__':
    chinese_word_list = []
    counter = Counter()
    with open('data/corpus/all_2.combined') as reader, open('data/corpus/test_corpus.combined', 'w') as writer:
        data = reader.readlines()
        for i in tqdm(data):
            writer.write(i.strip() + '\n')
            for j in i:
                if is_chinese(j):
                    counter.update(j)

    # write chinese chars
    chinese_chars = ''
    with open('data/chinese_char_2.txt', 'w') as writer:
        for i in counter.most_common(4200):
            writer.write(i[0].strip() + '\n')
            chinese_chars += i[0].strip() + ','
    english_corpus = ['data/corpus/libri_100.txt', 'data/corpus/libri_360.txt', 'data/corpus/libri_500.txt']
    for i in english_corpus:
        with open(i) as reader, open('data/corpus/english_2.corpus', 'w') as writer:
            for i in reader:
                writer.write(i.strip() + '\n')



    config_string = '--split_by_whitespace=1 --normalization_rule_name=nmt_nfkc_cf --add_dummy_prefix=1 --model_type=bpe --input=data/corpus/english_2.corpus --model_prefix=testing_vocab --vocab_size=6000 --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
    config_string += ' --user_defined_symbols=[S],[B],[N],[T],[P],[FIL],[SPK],\'s,' + chinese_chars[:-1]
    print(config_string)
    vocab_trainer = spm.SentencePieceTrainer.Train(config_string)




