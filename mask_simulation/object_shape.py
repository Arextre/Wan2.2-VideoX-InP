import numpy as np
import cv2
import glob
import os
try:
    from .transforms import get_random_affine_matrix, apply_affine_transform
except:
    from transforms import get_random_affine_matrix, apply_affine_transform

import time
import random
time.sleep(random.uniform(0, 3))  # Sleeps for 0.0–9.999... seconds
mask_dir = "./mask_simulation/image_dataset_6obj_200eg/"
print(f"{mask_dir=}, {os.path.exists(mask_dir)=}")
mask_object_list = [mask_dir + p.strip() for p in open(mask_dir + 'mask_list.txt', 'r').readlines()]
mask_images = [cv2.imread(mask_path) for mask_path in mask_object_list]
assert len(mask_images) == 1200


def get_object_shapes(width=512, height=512, server="local", random_affine=False):
    global mask_images

    idx = np.random.randint(len(mask_images))
    mask = mask_images[idx]  # object/person masks are 4ch
    if mask.ndim >= 3:
        mask = mask[:, :, 0]  # Only use one channel

    mask = cv2.resize(mask, (width, height))
    mask = np.where(mask > 128, 0, 255).astype(np.uint8)  # object being white, crucial to cast it to uint8!

    if random_affine:
        # augment the shape with random affine transformation
        affine_matrix = get_random_affine_matrix(width=width, height=height, tx_max=0.25, ty_max=0.25, angle_max=np.pi * 0.25, scale_min=0.1, scale_max=0.5, aspect_ratio_min=0.5, aspect_ratio_max=2.0, shear_max=0.1)
        mask3ch = np.stack((mask, mask, mask), axis=-1)
        mask3ch = apply_affine_transform(mask3ch, matrix=affine_matrix)
        mask = mask3ch[:, :, 0]

    return mask


if __name__ == "__main__":
    for i in range(20):
        mask = get_object_shapes(width=512, height=512, random_affine=True)
        cv2.imwrite("samples/mask_object_%d.png" % i, mask)
