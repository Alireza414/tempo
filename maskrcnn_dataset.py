import os
import numpy as np
import torch
import torch.utils.data
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from random import randrange

def display_img_and_mask(img,mask):
    f = plt.figure(figsize=(12, 12))
    f.add_subplot(1,2, 1)
    plt.imshow(img,cmap='gray')
    f.add_subplot(1,2, 2)
    plt.imshow(mask, cmap='gray')
    plt.show(block=True)

class AutoCTScanDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.imgs_array_path = ['./Numpy/6572_original_numpy.npy','./Numpy/6573_original_numpy.npy']
        self.masks_path = ['./NumpyMasks/6572_mask_numpy.npy','./NumpyMasks/6572_mask_numpy.npy']

        self.imgs_test_array_path = ['./Numpy/6575_original_numpy.npy']

    def __getitem__(self, idx):
        # load 1 image and its' masks
        file_num = randrange(len(self.imgs_array_path))
        img_npz = np.load(self.imgs_array_path[file_num])
        mask_npz = np.load(self.masks_path[file_num])

        mask_npz = mask_npz.astype(int)
        img_npz = img_npz.astype(int)
        img_grayscale = img_npz[idx]
        img_grayscale = np.divide( img_grayscale ,img_grayscale.max(), casting='unsafe')
        mask = mask_npz[idx]
        mask = np.divide(mask,mask.max(), casting='unsafe')

        #stack same image thrice to make 3 channle input
        img = np.stack((img_grayscale,)*3, axis=0)

        # instances are encoded as different colors
        #In our case 0 - background and 1 - aorta/artery
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a setof binary masks
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):

        image_stack = []
        for img_path in self.imgs_array_path:
            npz_file = np.load(img_path)
            npz_file=npz_file.astype(int)
            image_stack.append(npz_file.shape[0])
        return min(image_stack)

# if __name__ == "__main__":
#     dataset = AutoCTScanDataset()
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
#     # For Training
#     images,targets = next(iter(data_loader))
#     images = list(image for image in images)
#     # targets = [{k: v for k, v in t.items()} for t in targets]
