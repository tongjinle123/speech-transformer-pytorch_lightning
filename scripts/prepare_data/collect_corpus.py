import os
from os.path import join
from tqdm import tqdm
import sys

sys.path.append(os.getcwd())


def get_dir_files(path, post_fix):
    return [join(path, i) for i in os.listdir(path) if i.endswith(post_fix)]


def get_combined_segmented_text(file_list, write_to):
    with open(write_to, 'w', encoding='utf8') as writer:
        for file in file_list:
            print(file)
            with open(file, encoding='utf8') as reader:
                for line in tqdm(reader.readlines()):
                    line = line.strip()
                    writer.write(line.strip() + '\n')
                print(' '.join(line))
            print('----------')


if __name__ == '__main__':

    # files = get_dir_files('data/corpus/', '.txt')
    # get_combined_segmented_text(files, 'data/corpus/all.combined')
    files = get_dir_files('data/corpus/', '.txt')
    files = [i for i in files if 'c_500' in i or 'ce_200' in i]
    print(files)
    get_combined_segmented_text(files, 'data/corpus/all_2.combined')
