from pathlib import Path
import os.path as osp
import random, json


def create_data(root, stereo=False):
    splits = [s for s in root.iterdir() if s.name.startswith('20') and s.is_dir()]
    all_imgs = []
    all_index = []
    for split in splits:
        seqs = [s for s in split.iterdir() if s.is_dir()]

        for seq in seqs:
            valid_files = get_valid_index(seq / 'image_02' / 'data',
                                          seq / 'proj_depth' / 'groundtruth' / 'image_02')
            if len(valid_files) > 0:
                imgs = [json.dumps(dict(img=file, sequence=seq.name, split=split.name)) for file in valid_files]
                index = list(range(len(all_imgs) + 1, len(all_imgs) + len(imgs) - 1))
                all_imgs.extend(imgs)
                all_index.extend(map(str, index))

    
    return all_imgs, all_index


def get_valid_index(dir_1, dir_2, filter=True):
    files_1 = dir_1.glob('*')
    files_1 = [osp.splitext(f.name)[0] for f in files_1]

    files_2 = dir_2.glob('*')
    files_2 = [osp.splitext(f.name)[0] for f in files_2]

    if filter is True:
        finals_all = [f for f in files_1 if f in files_2]
    else:
        finals_all = files_1
    
    finals_all = sorted(finals_all, key=lambda f: int(f))

    return finals_all


if __name__ == '__main__':
    root_dir = Path('data/kitti_raw')
    all_imgs, all_index = create_data(root_dir)
    
    random.seed(10)
    random.shuffle(all_index)

    split_train_index = int(len(all_index) * 0.9)
    train_index = all_index[0:split_train_index]
    split_val_index = int(len(all_index) * 0.05)
    val_index = all_index[0:split_val_index]
    test_index = all_index[split_train_index:]

    with open(osp.join(root_dir, 'data_splits', 'custom_all_imgs.txt'), 'w') as f:
        f.writelines(img + '\n' for img in all_imgs)
    with open(osp.join(root_dir, 'data_splits', 'custom_train_index.txt'), 'w') as f:
        f.writelines(index + '\n' for index in train_index)
    with open(osp.join(root_dir, 'data_splits', 'custom_val_index.txt'), 'w') as f:
        f.writelines(index + '\n' for index in val_index)
    with open(osp.join(root_dir, 'data_splits', 'custom_test_index.txt'), 'w') as f:
        f.writelines(index + '\n' for index in test_index)
