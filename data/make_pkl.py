import pickle
import glob
import os
from scipy.misc import imread
import numpy as np


def main():
    root = '/home/chaehuny/ext/chaehuny/chaehun/bayesian/CamVid'

    train = os.path.join(root, 'train')
    trainannot = os.path.join(root, 'trainannot')
    val = os.path.join(root, 'val')
    valannot = os.path.join(root, 'valannot')
    test = os.path.join(root, 'test')
    testannot = os.path.join(root, 'testannot')

    path_list = [train, trainannot, val, valannot, test, testannot]
    path_names = ['train', 'trainannot',
                  'val', 'valannot', 'test', 'testannot']

    for i, path in enumerate(path_list):
        imgs = []
        for img_path in glob.glob(os.path.join(path, '*.png')):
            img = imread(img_path)
            imgs.append(img)
        imgs = np.array(imgs)

        with open(path_names[i]+'.pickle', 'wb') as f:
            pickle.dump(imgs, f)

        print(path_names[i])
        print('Array shape is ', imgs.shape)
        print(np.max(imgs))
        print(np.min(imgs))
        print(' ')


if __name__ == '__main__':
    main()
