import argparse
import cv2
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='A script for reading data.')
    parser.add_argument('--path',
                        required=True,
                        type=str,
                        help='data directory')
    args = parser.parse_args()
    load_image(args.path)


def load_image(path, mask=False):
    # read image
    img = cv2.imread(str(path))
    # convert images to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # extract H, W, C
    height, width, _ = img.shape
    # pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)
    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    # v2.copyMakeBorder(src, top, bottom, left, right, borderType)
    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad,
                             cv2.BORDER_REFLECT_101)
    if mask:
        # Convert mask to 0 and 1 format
        img = img[:, :, 0:1] // 255
        return torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))
    else:
        img = img / 255.0
        return torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))


if __name__ == "__main__":
    main()
