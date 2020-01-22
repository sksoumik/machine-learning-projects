import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from data_load import load_data_and_fix_map


class DatasetProcessing:
    def __init__(self, data, target, transform=None):
        self.transform = transform
        self.data = data.reshape((-1, 64, 64)).astype(np.float32)[:, :, :,
                                                                  None]
        self.target = torch.from_numpy(target).long()

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.target[index]

    def __len__(self):
        return len(list(self.data))


def transform_data():
    x_train, x_test, y_train, y_test = load_data_and_fix_map()
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.ToTensor()])
    dset_train = DatasetProcessing(x_train, y_train, transform)
    train_loader = torch.utils.data.DataLoader(dset_train,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=4)
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.ToTensor()])
    dset_train = DatasetProcessing(x_train, y_train, transform)
    train_loader = torch.utils.data.DataLoader(dset_train,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=1)
    dset_test = DatasetProcessing(x_test, y_test, transform)
    test_loader = torch.utils.data.DataLoader(dset_test,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=1)

    plt.figure(figsize=(16, 4))
    for num, x in enumerate(x_train[0:6]):
        plt.subplot(1, 6, num + 1)
        plt.axis('off')
        plt.imshow(x)
        plt.title(y_train[num])
        plt.savefig('../static/' + 'datasample.png')

    return train_loader, test_loader
