from tfrecord.writer import TFRecordWriter
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
from src.bak.data_loader import Featurizer
from tqdm import tqdm



manifest_list = None
tfrecord_root = None


def build_name(manifest_file, tfrecord_root):
    file_name = manifest_file.split('/')[-1]
    file_name = file_name.replace('.csv', '.tfrecord')
    file_name = os.path.join(tfrecord_root, file_name)
    return file_name

def filter_df(df, max_duration=10, min_duration=1):
    df = df[(df.duration > min_duration) & (df.duration < max_duration)]
    df = df.reset_index()
    df = df[['wav_file', 'target']]
    print(f'total_length: {len(df)}')
    return df


def build_tfrecord(manifest_file, tfrecord_root, featurizer):
    count = 0
    tfrecord_file = build_name(manifest_file, tfrecord_root)
    tfrecord_writer = TFRecordWriter(tfrecord_file)
    df = pd.read_csv(manifest_file)
    df = filter_df(df, min_duration=1, max_duration=12)
    df = df.to_dict('index')
    for i in tqdm(range(len(df))):
        line = df[i]
        wav_file, target = line['wav_file'], line['target']
        feature, feature_shape, target_id, target_length = featurizer(wav_file, target)
        tfrecord_writer.write({
            'feature': (feature.tobytes(), 'byte'),
            'feature_shape': (feature_shape, 'int'),
            'target_id': (target_id, 'int'),
            'target_length': ([target_length], 'int')
        })
        count += 1
    if count == 1:
        print(f'sample: {feature.shape}, \n{feature_shape}, \n{target_id}, \n{target_length}')
    file_name = tfrecord_file.replace('.tfrecord', f'_{count}.tfrecord')
    os.rename(tfrecord_file, file_name)
    print(f'done {manifest_file}')


if __name__ == '__main__':
    featurizer = Featurizer(vocab_path='testing_vocab_2.model')
    build_tfrecord('data/filterd_manifest/data_aishell_test.csv', 'data/tfrecords/', featurizer)