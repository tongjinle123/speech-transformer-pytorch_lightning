import os
import tarfile
import zipfile
from os.path import join
import time
import pandas as pd
import torchaudio as ta
from src_new.utils import tokenize, combine


class Logger(object):
    def __init__(self, func):
        self.func = func
        self.dilimit = '=' * 5 + '*' * 10 + '=' * 5

    def __call__(self, *args, **kwargs):
        print(self.dilimit)
        print(f'start {self.func.__name__} with args {args}')
        start = time.time()
        output = self.func(*args, **kwargs)
        end = time.time()
        print(f'done in {round(end-start, 5)}.')
        print(self.dilimit)
        return output


def search_folder_for_post_fix_file_list(root, post_fix):
    """
    search folder recurrnetly for files with post_fix
    :param root: path
    :param post_fix: post_fix
    :return: file list
    """
    targets = []
    files = [join(root, i) for i in os.listdir(root)]
    for i in files:
        if i.endswith(post_fix):
            targets.append(i)
        if os.path.isdir(i):
            for j in search_folder_for_post_fix_file_list(i, post_fix):
                targets.append(j)
    return targets


def extract_file(file_to_extract, folder_extract_to, type):
    """
    extract zip of tar.gz file
    :param file_to_extract:
    :param folder_extract_to:
    :param type:
    :return:
    """
    assert type in ['zip', 'tar']
    tools = {'zip': zipfile.ZipFile, 'tar': tarfile}
    if type == 'tar':
        with tools[type].open(file_to_extract) as file:
            file.extractall(folder_extract_to)
    else:
        with tools[type](file_to_extract) as file:
            file.extractall(folder_extract_to)
    print('extract_file done')


def nested_extract(extracted_folder, type):
    """
    extract zip or tar.gz files which deep in a folder
    :param extracted_folder:
    :param type:
    :return:
    """
    assert type in ['zip', 'tar']
    current_folder = extracted_folder
    current_files = [join(current_folder, i) for i in os.listdir(current_folder)]

    for file in current_files:
        if os.path.isdir(file):
            nested_extract(file, type)
        if file.endswith('.tar.gz'):
            extract_file(file, current_folder, type)


def extract_nested_file(file_to_extract, folder_extract_to, type):
    """
    extract file that have zipped file in zip file
    :param file_to_extract:
    :param folder_extract_to:
    :param type:
    :return:
    """
    assert type in ['zip', 'tar']
    extract_file(file_to_extract, folder_extract_to, type)
    nested_extract(folder_extract_to, type)
    print('extract_nested_file done ')


def extract_corpus_from_target_dict(target_dict, write_to):
    with open(write_to, 'w', encoding='utf8') as writer:
        for name, target in target_dict.items():
            writer.write(combine(tokenize(target.strip())) + '\n')
    print('done')


def merge(wav_list, target_dict, extract_name_fn, manifest_csv_path):
    wav_df = pd.DataFrame(wav_list, columns=['wav_file'])
    wav_df.index = wav_df.wav_file.apply(extract_name_fn)
    target_df = pd.DataFrame.from_dict(target_dict, orient='index', columns=['target'])
    merged_df = pd.merge(left=wav_df, right=target_df, left_index=True, right_index=True)
    merged_df['duration'] = merged_df['wav_file'].apply(cal_duration)
    merged_df['target'] = merged_df['target'].apply(lambda x: combine(tokenize(x)))
    try:
        merged_df.to_csv(manifest_csv_path, encoding='utf8')
    except:
        merged_df.to_csv(manifest_csv_path)
    print(f'manifest saved to {manifest_csv_path}')
    return 'done'


def extract_name_fn(path):
    return path.split('/')[-1].split('.')[0]


def cal_duration(file):
    signal, sr = ta.load(file)
    second = len(signal[0]) / sr
    return second

