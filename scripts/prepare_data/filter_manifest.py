import sys
import os
sys.path.append(os.getcwd())
from src_new.utils.corpus_tokenizer import is_chinese
import pandas as pd


with open('data/chinese_char.txt') as reader:
    data = reader.readlines()

word_list = [i.strip() for i in data]

def cal_coverage(input_string, word_list, ratio):
    val_count = 0
    good_count = 0
    for i in input_string:
        if is_chinese(i):
            val_count += 1
            if i in word_list:
                good_count += 1
            else:
                pass
    if val_count == 0:
        return True

    if good_count / val_count >= ratio:
        return True
    else:
        # print(input_string)
        return False


files = [
    # 'data/manifest/ce_200.csv',
    #      'data/manifest/c_500.csv',
    #      'data/manifest/AISHELL-2.csv',
    #      'data/manifest/data_aishell.csv',
    #      'data/manifest/aidatatang_200zh.csv',
    #      'data/manifest/magic_data_train.csv',
    #      'data/manifest/prime.csv',
    #      'data/manifest/stcmds.csv',
         'data/manifest/magic_data_test.csv',
         'data/manifest/magic_data_dev.csv'
         ]

for i in files:
    df = pd.read_csv(i)
    print(len(df))
    df = df[df.target.apply(lambda x: cal_coverage(x, word_list, 0.95))]
    print(len(df))
    print('---')
    file_name = i.replace('manifest', 'filterd_manifest')
    df.to_csv(file_name,index=False)




