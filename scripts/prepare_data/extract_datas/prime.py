import sys
import os
sys.path.append(os.getcwd())
from scripts.utils.folder_tool import *
import yaml
import json



config = yaml.safe_load(open('src/configs/default_config.yaml'))
corpus_root = config['data']['data_corpus_root']
extracted_root = config['data']['data_extracted_root']
manifest_root = config['data']['data_manifest_root']
raw = 'data/raw/prime.tar.gz'
prefix = raw.split('/')[-1].split('.')[0]
extracted_to = join(extracted_root, prefix)
manifest_csv_path = join(manifest_root, prefix + '.csv')
corpus_path = join(corpus_root, prefix + '.txt')


def extract_target(file):
    name2target = {}
    data = json.load(open(txt_list[0], encoding='utf8'))
    for line in data:
        name2target[line['file'].split('.')[0]] = ''.join(line['text'].split(' '))
    return name2target


def extract_name_fn(path):
    return path.split('/')[-1].split('.')[0]


if __name__ == '__main__':
    #extract_nested_file(raw, extracted_to, 'tar')
    wav_list = search_folder_for_post_fix_file_list(extracted_to ,'.wav')
    txt_list = search_folder_for_post_fix_file_list(extracted_to,'.json')
    target_dict = extract_target(txt_list[0])
    extract_corpus_from_target_dict(target_dict, corpus_path)
    merge(wav_list, target_dict, extract_name_fn, manifest_csv_path)
    print('all done')