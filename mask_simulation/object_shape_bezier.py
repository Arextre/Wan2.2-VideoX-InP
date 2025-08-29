import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

try:
    from .transforms import get_random_affine_matrix, apply_affine_transform
except:
    from transforms import get_random_affine_matrix, apply_affine_transform

mask_object_dir = "datasets/mask_simulation/image_dataset_6obj_200eg"  # local server
mask_object_list = sorted(glob.glob(mask_object_dir + "/*_sil/*.png"))
if len(mask_object_list) == 0:
    mask_object_dir = "./image_dataset_6obj_200eg"  # local server
    mask_object_list = sorted(glob.glob(mask_object_dir + "/*_sil/*.png"))


def get_object_shapes(width=512, height=512, server="local", random_affine=False):
    idx = np.random.randint(len(mask_object_list) - 1)
    mask_path = mask_object_list[idx]
    mask = plt.imread(mask_path)  # object/person masks are 4ch
    # from remote_pdb import set_trace; set_trace()
    if mask.ndim >= 3:
        mask = mask[:, :, 0]  # Only use one channel

    mask = cv2.resize(mask, (width, height))
    mask = np.where(mask > 128, 0, 255).astype(np.uint8)  # object being white, crucial to cast it to uint8!

    if random_affine:
        # augment the shape with random affine transformation
        angle_max = np.pi * 0.25
        angle = np.random.uniform(-angle_max, angle_max)
        tx = np.random.uniform(-width * 0.25, width * 0.25)
        ty = np.random.uniform(-height * 0.25, height * 0.25)
        affine_matrix = get_random_affine_matrix(
            width=width,
            height=height,
            tx=tx,
            ty=ty,
            angle=angle,
            scale_min=0.1,
            scale_max=0.5,
            aspect_ratio_min=0.5,
            aspect_ratio_max=2.0,
            shear_max=0.1,
        )
        mask3ch = np.stack((mask, mask, mask), axis=-1)
        mask3ch = apply_affine_transform(mask3ch, matrix=affine_matrix)
        mask = mask3ch[:, :, 0]

    return mask


if __name__ == "__main__":
    for i in range(20):
        mask = get_object_shapes(width=512, height=512, random_affine=True)
        cv2.imwrite("samples/mask_object_%d.png" % i, mask)
