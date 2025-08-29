# pip install pycocotools
import numpy as np
import cv2
from torchvision import datasets


dir_coco = "/nfs/flash/VideoAlg/AIGC/datasets/public/SHARE/coco"


dataset_coco = CocoDetection("{}/train2017/".format(dir_coco), "{}/annotations/instances_train2017.json".format(dir_coco))
dataset_coco = datasets.wrap_dataset_for_transforms_v2(dataset_coco)
n_coco = len(dataset_coco)


def get_coco_shapes(width=512, height=512):
    found_person = False

    while not found_person:
        idx = np.random.randint(n_coco - 1)
        image_coco_PIL, target = dataset_coco[idx]
        if "masks" not in target.keys():
            continue
        masks = target["masks"]
        boxes = target["boxes"]
        mask_union_np = np.zeros((height, width), dtype=np.float32)

        count_mask_person = 0
        for i in range(len(masks)):
            # if it is person (label=1)
            if target["labels"][i] == 1:
                count_mask_person += 1
                mask_np = np.array(masks[i])
                x0, y0, x1, y1 = boxes[i]
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                mask_crop_np = mask_np[y0:y1, x0:x1]
                # resize the mask if it's too large
                if max(mask_crop_np.shape) >= min(height, width):
                    scale_factor = min(height, width) / max(mask_crop_np.shape) * 0.75
                    mask_crop_np = cv2.resize(mask_crop_np, (0, 0), fx=scale_factor, fy=scale_factor)
                h_crop, w_crop = mask_crop_np.shape
                # randomly paste the person crop to the scene image
                y0p = np.random.randint(low=0, high=height - h_crop)
                x0p = np.random.randint(low=0, high=width - w_crop)
                # mask_tmp_np: the temporary mask for the instance of a person
                mask_tmp_np = np.zeros((height, width), dtype=np.float32)
                mask_tmp_np[y0p : y0p + h_crop, x0p : x0p + w_crop] = mask_crop_np
                mask_union_np = np.where(mask_tmp_np > 0, 1, mask_union_np)
        if count_mask_person > 0:
            found_person = True

        mask = np.where(mask_union_np > 0, 255, 0).astype(np.uint8)

    return mask


if __name__ == "__main__":
    for i in range(20):
        mask = get_coco_shapes(width=512, height=512)
        cv2.imwrite("samples/mask_coco_human_%d.png" % i, mask)
