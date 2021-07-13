import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
EXTENSION = 'png'


def imshow(img):
    plt.imshow(img)
    plt.show()


def create_label_map(list_images):
    label_map = None
    for l in list_images:
        img = cv2.imread(l, cv2.IMREAD_COLOR) #BGR
        print(img.shape)
        imshow(img)
        H, W, c = img.shape
        if label_map is None:
            label_map = np.zeros((H, W), dtype=np.uint8)
        label_map[img[:,:,0]>0] = 1
        label_map[img[:,:,2]>0] = 2
    return label_map

    return color_labels


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
        for filename in list_imgs:
            img_folder = os.path.join(path, filename)
            img_filepath = get_img_filepath(img_folder)
            img = cv2.imread(img_filepath)
            label = create_label_map(list_label_files(img_folder))

            print(label.dtype)
            imshow(label)
            break


