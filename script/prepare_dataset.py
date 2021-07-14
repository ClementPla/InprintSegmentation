import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tqdm

EXTENSION = 'png'


def imshow(img, vmin=None, vmax=None, title=None):
    plt.imshow(img, vmin=vmin, vmax=vmax)
    if title is not None:
        plt.title(title)
    plt.show()


def create_label_map(list_images):
    for i, l in enumerate(list_images):
        img = cv2.imread(l, cv2.IMREAD_UNCHANGED) #BGR
        H, W, c = img.shape
        if i==0 :
            label_map = np.zeros((H, W), dtype=np.uint8)
        alpha = img[:,:,3]
        # print(alpha.min(), alpha.max())
        # label_map[img[:,:,0]>55] = 1
        # label_map[img[:,:,2]>55] = 2
        label_map[alpha>0]=2*(img[:,:,0][alpha>0]>127)+1*(img[:,:,2][alpha>0]>127)
    return label_map


def list_label_files(folder):
    list_files = os.listdir(folder)
    return [os.path.join(folder, l) for l in list_files if ("Composite" not in l) and ("Background" not in l) and l.endswith(EXTENSION)]


def get_img_filepath(folder):
    list_files = os.listdir(folder)
    return [os.path.join(folder, l) for l in list_files if ("Background" in l)][0]


if __name__ == '__main__':
    ROOT = '../extern/'
    SUBS = ['train'] # List in case we add a proper validation set
    for sub in SUBS:
        path = os.path.join(ROOT, sub)
        os.makedirs(os.path.join(path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(path, 'gt'), exist_ok=True)

        list_imgs = [l for l in os.listdir(path) if l!='images' and l!='gt']
        for filename in tqdm.tqdm(list_imgs):
            img_folder = os.path.join(path, filename)
            img_filepath = get_img_filepath(img_folder)
            img = cv2.imread(img_filepath)
            label = create_label_map(list_label_files(img_folder))
            cv2.imwrite(os.path.join(path, 'images', filename+'.png'), img)
            cv2.imwrite(os.path.join(path, 'gt', filename+'.png'), label)


