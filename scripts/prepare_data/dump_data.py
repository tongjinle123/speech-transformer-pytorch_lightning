from tqdm import tqdm

from src_reshaped.loader.dataloader.audio_loader import build_data_loader_dump


def build_data(record_root, manifest_list):
    def build_sub_folder(path):
        os.mkdir(path)
    def get_name(path):
        return path.split('/')[-1].split('.')[0]
    def dump_files(root, dump_loaders):
        rates = [0.9, 1.0, 1.1]
        count = 0
        for dump_loader, rate in zip(dump_loaders, rates):
            for i in tqdm(dump_loader):
                for line in i:
                    count += 1
                    name = os.path.join(root,str(count)+f'_{rate}.t')
                    t.save(line, name)
        print('done')
    for manifest in manifest_list:
        print(manifest)
        name = get_name(manifest)
        print(name)
        sub_folder_path = os.path.join(record_root, name)
        build_sub_folder(sub_folder_path)
        loaders = [build_data_loader_dump(manifest_list=[manifest],given_rate=rate) for rate in [0.9,1.0,1.1]]
        dump_files(sub_folder_path, loaders)
        print(f'{manifest} done')

record_root = 'data/records/'
manifest_list = ['data/filterd_manifest/ce_200.csv', 'data/filterd_manifest/ce_20_dev.csv']

build_data(record_root, manifest_list)