import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
import re
from PIL import ImageFile
from os import path
import numpy as np
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

def tensor2quaternion(image, a = 0.33):
    red, green, blue = torch.split(image, 3, dim=0)[0] # red, green, blue
    r = a*(red + green + blue) # red, green, blue
    i = (1-a)*red # get an i component as a percentage of the red
    j = (1-a)*green # get an j componen as a percentage of the green
    k = (1-a)*blue # k an blue
    quat = torch.stack([r, i, j, k], dim=0) # quaternion | stack four things: r,i,j,k
    return quat


# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir, train_filename):
        super().__init__()
        train_list = train_data_dir + train_filename    # train_list -> train_data_dir + train_filename
        with open(train_list) as f:                     # open(train_list)
            contents = f.readlines()                    # contents = f.readlines()
            input_names = [i.strip().replace('./', '') for i in contents] # input_names
            gt_names = [i.strip().replace('input', 'gt').replace('_clean.png', '_rain.png') for i in input_names] #

        self.input_names = input_names  # input names
        self.gt_names = gt_names        # gt names
        self.crop_size = crop_size      # crop size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size    # crop_width, crop_height
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        img_id = re.split('/', input_name)[-1][:-4]

        input_img = Image.open(self.train_data_dir + input_name)

        try:
            gt_img = Image.open(self.train_data_dir + gt_name)
        except:
            gt_img = Image.open(self.train_data_dir + gt_name).convert('RGB')

        width, height = input_img.size

        if width < crop_width and height < crop_height:
            input_img = input_img.resize((crop_width, crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width:
            input_img = input_img.resize((crop_width, height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, height), Image.ANTIALIAS)
        elif height < crop_height:
            input_img = input_img.resize((width, crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size

        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        # here I need to rearange from 4 channel to 3 channel

        # --- Check the channel is 3 or not --- #
        if list(input_im.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return tensor2quaternion(input_im), tensor2quaternion(gt), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)



class ValData(data.Dataset):
    def __init__(self, val_data_dir,val_filename):
        super().__init__()
        val_list = val_data_dir + val_filename
        with open(val_list) as f:
            contents = f.readlines()
            input_names = [i.strip().replace('./', '') for i in contents]
            gt_names = [i.strip().replace('input','gt').replace('_rain.png', '_clean.png') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        input_img = Image.open(self.val_data_dir + input_name)
        gt_img = Image.open(self.val_data_dir + gt_name)

        # Resizing image in the multiple of 16"
        wd_new,ht_new = input_img.size
        if ht_new>wd_new and ht_new>1024:
            wd_new = int(np.ceil(wd_new*1024/ht_new))
            ht_new = 1024
        elif ht_new<=wd_new and wd_new>1024:
            ht_new = int(np.ceil(ht_new*1024/wd_new))
            wd_new = 1024
        wd_new = int(16*np.ceil(wd_new/16.0))
        ht_new = int(16*np.ceil(ht_new/16.0))
        input_img = input_img.resize((wd_new,ht_new), Image.ANTIALIAS)
        gt_img = gt_img.resize((wd_new, ht_new), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)

        return tensor2quaternion(input_im), tensor2quaternion(gt), input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
