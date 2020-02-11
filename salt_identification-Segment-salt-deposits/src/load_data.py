import os
import numpy as np
import pandas as pd
from torch.utils import data
from data_preprocess import load_image


class Dataset(data.Dataset):
    def __init__(self, root_path, file_list, is_test=False):
        self.is_test = is_test
        self.root_path = root_path  # 'input/root_path'
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]

        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")

        mask_folder = os.path.join(self.root_path, "masks")
        mask_path = os.path.join(mask_folder, file_id + ".png")

        image = load_image(image_path)  # load_image():func <- data_preprocess.py

        if self.is_test:
            return (image, )
        else:
            mask = load_image(mask_path, mask=True)
            return image, mask


def define_paths(directory):
    depths_df = pd.read_csv(os.path.join(directory, 'train.csv'))
    train_path = os.path.join(directory, 'train')
    file_list = list(depths_df['id'].values)
    return train_path, file_list
