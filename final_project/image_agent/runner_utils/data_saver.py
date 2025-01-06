import numpy as np
import shutil
import os

class DataSaver:
    def __init__(self):
        self.data = []

        # For inclusion criteria to td
        self.max_puck_size = np.inf
        self.min_puck_size = 20
    
    def inclusion_criteria(self, coor, mask) -> bool:
        if np.any(np.abs(coor) == np.array([1., 1.])):
            return False
        
        if not self.min_puck_size < np.count_nonzero(mask) < self.max_puck_size:
            return False
            
        return True

    def give(self, vision):
        pass
    
def create_dirs(path_training_dir):
    # Check whether the specified path exists or not
    isDataDirExists, isMasksExists = os.path.exists(path_training_dir)

    if isDataDirExists:
        shutil.rmtree(path_training_dir)
    os.makedirs(path_training_dir)


def get_default_folder():
    from glob import glob

    root_folder = './hockey_training_data/'

    def get_counter(fn):
        return int(fn[-3:])
    
    def complete_counter(cnt):
        cnt = str(cnt)
        assert len(cnt) <= 3
        return '0' * (3 - len(cnt)) + cnt
    
    counter = 0
    for fn in sorted(glob(root_folder + 'td_*')):
        current_counter = get_counter(fn)
        if counter != current_counter:
            break
        counter += 1
    
    return root_folder + 'td_' + complete_counter(counter) + '/'

def save_td(entries):
    import os

    counter = 0
    with_count = lambda s: s + '_' + str(counter)

    folder = get_default_folder()
    os.makedirs(folder)
    # create_dirs(folder)

    im_folder = folder + 'im'
    mask_folder = folder + 'mask'
    coor_folder = folder + 'coor'

    for vision, coor, mask in entries:
        # Save vision
        from PIL import Image
        vision_fn = with_count(im_folder) + '.png'
        Image.fromarray(vision).save(vision_fn)

        # Save coor
        coor_fn = with_count(coor_folder) + '.csv'
        with open(coor_fn, 'w') as f:
            f.write('%f,%f' % tuple(coor))

        # Save mask
        mask_fn = with_count(mask_folder) + '.png'
        Image.fromarray(mask).save(mask_fn)

        counter += 1