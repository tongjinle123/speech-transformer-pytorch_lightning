import gc
import os
import sys
sys.path.append(os.getcwd())
from src.bak.data_loader import build_tfrecord
from src.bak.data_loader import Featurizer



if __name__ == '__main__':
    tfrecord_root = 'data/tfrecords/'
    manifest_list = [
        # 'data/filterd_manifest/data_aishell_train.csv',
        # 'data/filterd_manifest/c_500_train.csv',
        # 'data/filterd_manifest/ce_200.csv',
        # 'data/filterd_manifest/AISHELL-2.csv',
        # 'data/filterd_manifest/magic_data_train.csv',
        # 'data/filterd_manifest/magic_data_test.csv',
        # 'data/filterd_manifest/magic_data_dev.csv',
        #
        # 'data/filterd_manifest/data_aishell_test_small.csv',
        # 'data/filterd_manifest/magic_data_test_small.csv',
        # 'data/filterd_manifest/c_500_test_small.csv',
        'data/filterd_manifest/c_500_test.csv',
        # 'aishell2_testing/manifest1.csv',
        'data/manifest/ce_20_dev_small.csv'
    ]
    featurizer = Featurizer(n_mel=80, left_frames=3, right_frames=0, skip_frames=2, vocab_path='testing_vocab.model')

    for i in manifest_list:
        build_tfrecord(i, tfrecord_root, featurizer)
        gc.collect()