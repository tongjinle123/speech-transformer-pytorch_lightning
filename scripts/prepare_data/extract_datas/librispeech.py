import sys
import os
sys.path.append(os.getcwd())
from scripts.utils.folder_tool import *
import yaml
from tqdm import tqdm

def extract_target(file_list):
    name2target = {}
    for file in tqdm(file_list, desc='filelist'):
        with open(file, encoding='utf8') as reader:
            data = reader.readlines()
            for line in data:
                name = line.split(' ')[0]
                target = ' '.join(line.strip().split(' ')[1:]).lower()
                name2target[name] = target
    return name2target



def extract_name_fn(path):
    return path.split('/')[-1].split('.')[0]

if __name__ == '__main__':
    for raw_i in ['libri_500.tar.gz','libri_360.tar.gz','libri_100.tar.gz']:
        config = yaml.safe_load(open('src/configs/default_config.yaml'))
        corpus_root = config['data']['data_corpus_root']
        extracted_root = config['data']['data_extracted_root']
        manifest_root = config['data']['data_manifest_root']
        raw = os.path.join('data/raw/', raw_i)
        #raw = 'data/raw/libri_500.tar.gz'
        prefix = raw.split('/')[-1].split('.')[0]
        extracted_to = join(extracted_root, prefix)
        manifest_csv_path = join(manifest_root, prefix + '.csv')
        corpus_path = join(corpus_root, prefix + '.txt')

        #extract_nested_file(raw, extracted_to, 'tar')
        wav_list = search_folder_for_post_fix_file_list(extracted_to, '.flac')
        txt_list = search_folder_for_post_fix_file_list(extracted_to, '.txt')
        target_dict = extract_target(txt_list)
        extract_corpus_from_target_dict(target_dict, corpus_path)
        merge(wav_list, target_dict, extract_name_fn, manifest_csv_path)

    print('all done')