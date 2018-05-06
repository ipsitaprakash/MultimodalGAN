import os
import io
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pdb
from PIL import Image
import torch
from torch.autograd import Variable
import pdb
import torch.nn.functional as F
import h5py


class Sound2ImageDataset(Dataset):

    def __init__(self, datasetFile):
        self.datasetFile = datasetFile
        self.dataset = h5py.File(self.datasetFile, mode='r')
        self.dataset_keys = [str(k) for k in self.dataset.keys()]
        #self. = lambda x: int(np.array(x))

    def __len__(self):
        return len(self.dataset_keys)

    def __getitem__(self, idx):
        # pdb.set_trace()
        example_name = self.dataset_keys[idx]
        example = self.dataset[example_name]

        right_image = bytes(self.dataset['img'][idx])#bytes(np.array(example['img']))
        right_embed = self.dataset['embeddings'][idx]#np.array(example['embeddings'], dtype=float)
        wrong_image = bytes(self.dataset['class'][idx])#bytes(np.array(self.find_wrong_image(example['class'])))

        right_image = Image.open(io.BytesIO(right_image)).resize((64, 64))
        wrong_image = Image.open(io.BytesIO(wrong_image)).resize((64, 64))

        right_image = self.validate_image(right_image)
        wrong_image = self.validate_image(wrong_image)

        sample = {
                'right_images': torch.FloatTensor(right_image),
                'right_embed': torch.FloatTensor(right_embed),
                'wrong_images': torch.FloatTensor(wrong_image)
                 }

        sample['right_images'] = sample['right_images']     #TODO : .sub_(127.5).div_(127.5)
        sample['wrong_images'] = sample['wrong_images']      #TODO : .sub_(127.5).div_(127.5)

        return sample

    def find_wrong_image(self, category):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[example_name]
        _category = example['class']

        if _category != category:
            return example['img']

        return self.find_wrong_image(category)


    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)