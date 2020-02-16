import os
import sys
import gc
sys.path.append(os.getcwd())

if __name__ == '__main__':
    record_root = 'data/tfrecords/'
    index_root = 'data/tfrecord_index/'
    records = [os.path.join(record_root, i) for i in os.listdir(record_root)]
    indexs = [os.path.join(index_root, i.split('/')[-1].replace('.tfrecord', '.index')) for i in records]
    for i in zip(records, indexs):
        cmd = f'python -m tfrecord.tools.tfrecord2idx {i[0]} {i[1]}'
        os.system(cmd)
        print(f'{cmd} exed')
        gc.collect()
