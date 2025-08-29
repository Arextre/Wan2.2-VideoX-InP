import numpy as np 
import imageio 
from skimage.transform import resize
import glob
import os 
from PIL import Image
import matplotlib.pyplot as plt

import torch
# from torchvision.transforms import v2
import torchvision.transforms as T

torch.manual_seed(0)

# 
# applier = T.RandomApply(transforms=[T.RandomCrop(size=(256, 256))], p=0.5)
affine_transfomer = T.RandomAffine(degrees=(45, 45), translate=(0.0, 0.0), scale=(0.5, 0.5))

# Parameters
size = 512

# Load the image
list_category = ['bed', 'bottle', 'car', 'chair', 'desk lamp', 'person']

for category in list_category:
    list_img = glob.glob(category + '_sil/*.png')
    print("For {}, {} silhouettes found".format(category, len(list_img)))
    save_path = category + '_mask/'
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    for img_path in list_img:
        img = imageio.imread(img_path) / 255.0
        mask = img[:, :, 0] # Only use one channel
        mask = np.where(mask > 0.5, 0, 1) # object being white
        mask = resize(mask, (size, size))
        # print("Image shape: {}".format(img.shape))
        mask_Image = Image.fromarray(mask)
        transformed_Image = affine_transfomer(mask_Image)
        transformed = np.array(transformed_Image).squeeze()
        print("Transformed shape: {}".format(transformed.shape))
        before_after = np.hstack((mask, transformed))
        plt.imshow(before_after)
        plt.savefig(save_path + os.path.basename(img_path))


        break
    break




