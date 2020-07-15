from pathlib import Path
import os.path as osp

def create_data(root, stereo=False):
    splits = [s for s in root.iterdir() if s.name.startswith('20') and s.is_dir()]
    all_index = []
    for split in splits:
        seqs = [s for s in (root/split).iterdir() if s.is_dir()]

        for seq in seqs:
            valid_files = get_valid_index(seq/'image_02'/'data',
                                          seq/'proj_depth'/'ground_depth'/'image_02')
            img = [dict(img=file, sequence=seq, split=split) for file in valid_files]
            all_index.append(img)

    return all_index


def get_valid_index(dir_1, dir_2):
    files_1 = dir_1.glob()
    files_1 = [osp.splitext(files_1)[0] for f in files_1]

    files_2 = dir_2.glob()
    files_2 = [osp.splitext(files_2)[0] for f in files_2]

    return [f for f in files_1 if f in files_2]

if __name__ == '__main__':
    root_dir = Path('data/kitti_depth')
    all_index = create_data(root_dir)

    with open(root_dir+'all_index.txt', 'w') as f:
        f.writelines(all_index)