import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage import exposure
from skimage import img_as_float64
import matplotlib.pyplot as plt

import argparse
import os
import datetime
import argparse
import numpy as np

import cv2
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy import signal
from torchvision.utils import make_grid
import numpy.random as random

from PIL import Image

from core_qnn.quaternion_layers import QuaternionTransposeConv
from core_qnn.quaternion_layers import QuaternionConv
from core_qnn.quaternion_ops import check_input
from core_qnn.quaternion_ops import get_r, get_i, get_j, get_k

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def edge_compute(x):
    x_diffx = torch.abs(x[:,:,1:] - x[:,:,:-1])
    x_diffy = torch.abs(x[:,1:,:] - x[:,:-1,:])

    y = x.new(x.size())
    y.fill_(0)
    y[:,:,1:] += x_diffx
    y[:,:,:-1] += x_diffx
    y[:,1:,:] += x_diffy
    y[:,:-1,:] += x_diffy
    y = torch.sum(y,0,keepdim=True)/3
    y /= 4
    return y

def batch_edge_compute(x):
    x_diffx = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
    x_diffy = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])

    y = x.new(x.size())
    y.fill_(0)
    y[:,:,:,1:] += x_diffx
    y[:,:,:,:-1] += x_diffx
    y[:,:,1:,:] += x_diffy
    y[:,:,:-1,:] += x_diffy
    y = torch.sum(y,1,keepdim=True)/3
    y /= 4
    return y

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    return image_numpy.astype(imtype)

def tensor2imgrid(input_image):
    im_grid = make_grid(input_image[:4, ...], nrow=2, normalize=True, range=(-128, 128))
    return im_grid
    # ndarr = im_grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    # im = Image.fromarray(ndarr)
    # return im

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def filter2(x, kernel, mode='same'):
    return signal.convolve2d(x, np.rot90(kernel, 2), mode=mode)

def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = filter2(img1, window, mode='valid')
    mu2 = filter2(img2, window, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(img1 * img1, window, mode='valid') - mu1_sq
    sigma2_sq = filter2(img2 * img2, window, mode='valid') - mu2_sq
    sigma12 = filter2(img1 * img2, window, mode='valid') - mu1_mu2
    if cs_map:
        return np.mean(np.mean((((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))))
    else:
        return np.mean(np.mean(((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))))

class MovingAvg(object):
    def __init__(self, pool_size=100):
        from queue import Queue
        self.pool = Queue(maxsize=pool_size)
        self.sum = 0
        self.curr_pool_size = 0
        self.pool_size = pool_size

    def set_curr_val(self, val):
        if not self.pool.full():
            self.curr_pool_size += 1
            self.pool.put_nowait(val)
        else:
            last_first_val = self.pool.get_nowait()
            self.pool.put_nowait(val)
            self.sum -= last_first_val

        self.sum += val
        return self.sum / self.curr_pool_size

    def reset(self):
        from queue import Queue
        self.pool = Queue(maxsize=self.pool_size)
        self.sum = 0
        self.curr_pool_size = 0

class FolderLoader(object):
    def __init__(self, fold_path):
        super(FolderLoader, self).__init__()
        self.fold_path = fold_path
        self.img_paths = make_dataset(self.fold_path)
        self.img_names = [os.path.basename(x) for x in self.img_paths]

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])#.convert('RGB')
        return self.img_names[index], img

    def __len__(self):
        return len(self.img_names)

def pil_loader(img_path):
    return Image.open(img_path).convert("RGB")


class ImagePairPrefixFolder(Dataset):
    def __init__(self, input_folder, gt_folder, max_img_size=0, size_unit=1, force_rgb=False):
        super(ImagePairPrefixFolder, self).__init__()

        self.gt_loader = FolderLoader(gt_folder)
        # build the map from image name to index
        self.gt_map = dict()
        for idx, img_name in enumerate(self.gt_loader.img_names):
            self.gt_map[os.path.splitext(img_name)[0].split('_')[0]] = idx

        self.input_loader = FolderLoader(input_folder)
        assert all([os.path.splitext(x)[0].split('_')[0] in self.gt_map for x in self.input_loader.img_names]), \
            'cannot find corresponding gt names'

        self.input_folder = input_folder
        self.gt_folder = gt_folder
        self.max_img_size = max_img_size
        self.size_unit = size_unit
        self.force_rgb = force_rgb

    def __getitem__(self, index):
        input_name, input_img = self.input_loader[index]
        input_basename = os.path.splitext(input_name)[0].split('_')[0]
        gt_idx = self.gt_map[input_basename]

        gt_name, gt_img = self.gt_loader[gt_idx]
        if self.force_rgb:
            input_img = input_img.convert('RGB')
            gt_img = gt_img.convert('RGB')
        im_w, im_h = input_img.size
        gt_w, gt_h = gt_img.size

        if (im_w != gt_w) or (im_h != gt_h):
            print(input_name)
            print(gt_name)

        assert im_w == gt_w and im_h == gt_h, 'input image and gt image size not match'

        im_w, im_h = input_img.size
        if 0 < self.max_img_size < max(im_w, im_h):
            if im_w < im_h:
                out_h = int(self.max_img_size) // self.size_unit * self.size_unit
                out_w = int(im_w / im_h * out_h) // self.size_unit * self.size_unit
            else:
                out_w = int(self.max_img_size) // self.size_unit * self.size_unit
                out_h = int(im_h / im_w * out_w) // self.size_unit * self.size_unit
        else:
            out_w = im_w // self.size_unit * self.size_unit
            out_h = im_h // self.size_unit * self.size_unit

        if im_w != out_w or im_h != out_h:
            input_img = input_img.resize((out_w, out_h), Image.BILINEAR)
            gt_img = gt_img.resize((out_w, out_h), Image.BILINEAR)

        im_w, im_h = input_img.size

        # input_img = np.array(input_img).astype('float')
        # input_img = np.array(input_img)

        input_img = np.array(input_img)
        gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
        # gray = cv2.imread(os.path.join('ds/train_general/trans/', input_name.replace('.jpg', '.png')), 0)
        # gray = cv2.resize(gray, (out_w, out_h))
        input_img = np.dstack([gray[:, :, np.newaxis], input_img])
        input_img = input_img.astype('float')

        gt_img = np.array(gt_img)
        gray = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)
        gt_img = np.dstack([gray[:, :, np.newaxis], gt_img])
        gt_img = gt_img.astype('float')
        if len(input_img.shape) == 2:
            input_img = input_img[:, :, np.newaxis]
        if len(gt_img.shape) == 2:
            gt_img = gt_img[:, :, np.newaxis]

        input_img = input_img
        gt_img = gt_img
        return {'input_img': input_img, 'gt_img': gt_img, 'input_h': im_h, "input_w": im_w}

    def get_input_info(self, index):
        image_name = os.path.splitext(self.input_loader.img_names[index])[0]
        return self.input_loader, image_name

    def __len__(self):
        return len(self.input_loader)


class ImagePairPrefixFolder(Dataset):
    def __init__(self, input_folder, gt_folder, max_img_size=0, size_unit=1, force_rgb=False):
        super(ImagePairPrefixFolder, self).__init__()

        self.gt_loader = FolderLoader(gt_folder)
        # build the map from image name to index
        self.gt_map = dict()
        for idx, img_name in enumerate(self.gt_loader.img_names):
            self.gt_map[os.path.splitext(img_name)[0].split('_')[0]] = idx

        self.input_loader = FolderLoader(input_folder)
        assert all([os.path.splitext(x)[0].split('_')[0] in self.gt_map for x in self.input_loader.img_names]), \
            'cannot find corresponding gt names'

        self.input_folder = input_folder
        self.gt_folder = gt_folder
        self.max_img_size = max_img_size
        self.size_unit = size_unit
        self.force_rgb = force_rgb

    def __getitem__(self, index):
        input_name, input_img = self.input_loader[index]
        input_basename = os.path.splitext(input_name)[0].split('_')[0]
        gt_idx = self.gt_map[input_basename]

        gt_name, gt_img = self.gt_loader[gt_idx]
        if self.force_rgb:
            input_img = input_img.convert('RGB')
            gt_img = gt_img.convert('RGB')
        im_w, im_h = input_img.size
        gt_w, gt_h = gt_img.size

        if (im_w != gt_w) or (im_h != gt_h):
            print(input_name)
            print(gt_name)

        assert im_w == gt_w and im_h == gt_h, 'input image and gt image size not match'

        im_w, im_h = input_img.size
        if 0 < self.max_img_size < max(im_w, im_h):
            if im_w < im_h:
                out_h = int(self.max_img_size) // self.size_unit * self.size_unit
                out_w = int(im_w / im_h * out_h) // self.size_unit * self.size_unit
            else:
                out_w = int(self.max_img_size) // self.size_unit * self.size_unit
                out_h = int(im_h / im_w * out_w) // self.size_unit * self.size_unit
        else:
            out_w = im_w // self.size_unit * self.size_unit
            out_h = im_h // self.size_unit * self.size_unit

        if im_w != out_w or im_h != out_h:
            input_img = input_img.resize((out_w, out_h), Image.BILINEAR)
            gt_img = gt_img.resize((out_w, out_h), Image.BILINEAR)

        im_w, im_h = input_img.size

        # input_img = np.array(input_img).astype('float')
        # input_img = np.array(input_img)

        input_img = np.array(input_img)
        gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
        # gray = cv2.imread(os.path.join('ds/train_general/trans/', input_name.replace('.jpg', '.png')), 0)
        # gray = cv2.resize(gray, (out_w, out_h))
        input_img = np.dstack([gray[:, :, np.newaxis], input_img])
        input_img = input_img.astype('float')

        gt_img = np.array(gt_img)
        gray = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)
        gt_img = np.dstack([gray[:, :, np.newaxis], gt_img])
        gt_img = gt_img.astype('float')
        if len(input_img.shape) == 2:
            input_img = input_img[:, :, np.newaxis]
        if len(gt_img.shape) == 2:
            gt_img = gt_img[:, :, np.newaxis]

        input_img = input_img
        gt_img = gt_img
        return {'input_img': input_img, 'gt_img': gt_img, 'input_h': im_h, "input_w": im_w}

    def get_input_info(self, index):
        image_name = os.path.splitext(self.input_loader.img_names[index])[0]
        return self.input_loader, image_name

    def __len__(self):
        return len(self.input_loader)

def var_custom_collate(batch):
    min_h, min_w = 10000, 10000
    for item in batch:
        min_h = min(min_h, item['input_h'])
        min_w = min(min_w, item['input_w'])
    inc = 1 if len(batch[0]['input_img'].shape)==2 else batch[0]['input_img'].shape[2]
    batch_input_images = torch.Tensor(len(batch), 4, min_h, min_w)
    batch_gt_images = torch.Tensor(len(batch), 4, min_h, min_w)

    for idx, item in enumerate(batch):
        off_y = 0 if item['input_h']==min_h else random.randint(0, item['input_h'] - min_h)
        off_x = 0 if item['input_w']==min_w else random.randint(0, item['input_w'] - min_w)
        crop_input_img = item['input_img'][off_y:off_y + min_h, off_x:off_x + min_w, :]
        crop_gt_img = item['gt_img'][off_y:off_y + min_h, off_x:off_x + min_w, :]
        batch_input_images[idx] = torch.from_numpy(crop_input_img.transpose((2, 0, 1))) - 128
        batch_gt_images[idx] = torch.from_numpy(crop_gt_img.transpose((2, 0, 1)))


    batch_input_edges = batch_edge_compute(batch_input_images) - 128
    return batch_input_images, batch_input_edges,  batch_gt_images

class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)

class QuaternionInstanceNorm2d(nn.Module):
    def __init__(self, num_features, gamma_init=1., beta_param=True, training=True):
        super(QuaternionInstanceNorm2d, self).__init__()
        self.num_features = num_features // 4
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.gamma = nn.Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)
        self.training = training
        self.eps = torch.tensor(1e-5)

    def reset_parameters(self):
        self.gamma = nn.Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)

    def forward(self, input):
        quat_components = torch.chunk(input, 4, dim=1)
        r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]
        delta_r, delta_i, delta_j, delta_k = r - torch.mean(r, axis=[1, 2, 3], keepdim=True), i - torch.mean(i, axis=[1, 2, 3], keepdim=True), j - torch.mean(j, axis=[1, 2, 3], keepdim=True), k - torch.mean(k, axis=[1, 2, 3], keepdim=True)
        quat_variance = torch.mean((delta_r**2 + delta_i**2 + delta_j**2 + delta_k**2))
        denominator = torch.sqrt(quat_variance + self.eps)

        # Normalize
        r_normalized = delta_r / denominator
        i_normalized = delta_i / denominator
        j_normalized = delta_j / denominator
        k_normalized = delta_k / denominator

        beta_components = torch.chunk(self.beta, 4, dim=1)

        # Multiply gamma (stretch scale) and add beta (shift scale)
        new_r = (self.gamma * r_normalized) + beta_components[0]
        new_i = (self.gamma * i_normalized) + beta_components[1]
        new_j = (self.gamma * j_normalized) + beta_components[2]
        new_k = (self.gamma * k_normalized) + beta_components[3]

        new_input = torch.cat((new_r, new_i, new_j, new_k), dim=1)

        return new_input

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma) \
               + ', beta=' + str(self.beta) \
               + ', eps=' + str(self.eps) + ')'

class SmoothDilatedResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(SmoothDilatedResidualBlock, self).__init__()
        self.pre_conv1 = ShareSepConv(dilation*2-1)
        self.conv1 = QuaternionConv(channel_num, channel_num, 3, 1, padding=dilation, dilatation=dilation, groups=group, bias=False)
        self.norm1 = QuaternionInstanceNorm2d(channel_num)
        self.pre_conv2 = ShareSepConv(dilation*2-1)
        self.conv2 = QuaternionConv(channel_num, channel_num, 3, 1, padding=dilation, dilatation=dilation, groups=group, bias=False)
        self.norm2 = QuaternionInstanceNorm2d(channel_num)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))
        y = self.norm2(self.conv2(self.pre_conv2(y)))
        return F.relu(x+y)

class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = QuaternionConv(channel_num, channel_num, 3, 1, padding=dilation, dilatation=dilation, groups=group, bias=False)
        self.norm1 = QuaternionInstanceNorm2d(channel_num)
        self.conv2 = QuaternionConv(channel_num, channel_num, 3, 1, padding=dilation, dilatation=dilation, groups=group, bias=False)
        self.norm2 = QuaternionInstanceNorm2d(channel_num)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x+y)

class GCANet(nn.Module):
    def __init__(self, in_c=4, out_c=3, only_residual=True):
        super(GCANet, self).__init__()
        self.conv1 = QuaternionConv(in_c, 64, 3, 1, 1, 1, bias=False)
        self.norm1 = QuaternionInstanceNorm2d(64)
        self.conv2 = QuaternionConv(64, 64, 3, 1, 1, 1, bias=False)
        self.norm2 = QuaternionInstanceNorm2d(64)
        self.conv3 = QuaternionConv(64, 64, 3, 2, 1,1, bias=False)
        self.norm3 = QuaternionInstanceNorm2d(64)

        self.res1 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res2 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res3 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res4 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res5 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res6 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res7 = ResidualBlock(64, dilation=1)

        self.gate = QuaternionConv(64 * 3, 64, 1, 1, 1, bias=True)

        self.deconv3 = QuaternionTransposeConv(64, 64, 4, 2, 1, 1)
        self.norm4 = QuaternionInstanceNorm2d(64)
        self.deconv2 = QuaternionConv(2 * 64, 64, 3, 1, 1, 1)
        self.norm5 = QuaternionInstanceNorm2d(64)
        self.deconv1 = QuaternionConv(64, out_c, 1, 1, bias=True)
        self.only_residual = only_residual

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y0 = F.relu(self.norm2(self.conv2(y)))
        y1 = F.relu(self.norm3(self.conv3(y0)))

        y = self.res1(y1)
        y = self.res2(y)
        y = self.res3(y)
        y2 = self.res4(y)
        y = self.res5(y2)
        y = self.res6(y)
        y3 = self.res7(y)


        r = torch.cat((get_r(y1), get_r(y2), get_r(y3)), dim=1)
        i = torch.cat((get_i(y1), get_i(y2), get_i(y3)), dim=1)
        j = torch.cat((get_j(y1), get_j(y2), get_j(y3)), dim=1)
        k = torch.cat((get_k(y1), get_k(y2), get_k(y3)), dim=1)

        gated_y = self.gate(torch.cat((r, i, j, k), dim=1))

        y = F.relu(self.norm4(self.deconv3(gated_y)))
        r = torch.cat((get_r(y0), get_r(y)), dim=1)
        i = torch.cat((get_i(y0), get_i(y)), dim=1)
        j = torch.cat((get_j(y0), get_j(y)), dim=1)
        k = torch.cat((get_k(y0), get_k(y)), dim=1)
        y = torch.cat((r, i, j, k), dim=1)
        y = F.relu(self.norm5(self.deconv2(y)))
        if self.only_residual:
            y = self.deconv1(y)
        else:
            y = F.relu(self.deconv1(y))

        return y

def restore_image(net, fname):
  img = Image.open(fname).convert('RGB')
  im_w, im_h = img.size
  if im_w % 4 != 0 or im_h % 4 != 0:
      img = img.resize((int(im_w // 4 * 4), int(im_h // 4 * 4)))

  img = np.array(img)
  h, w, c = img.shape

  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img = np.dstack([gray[:, :, np.newaxis], img])
  img = img.astype('float')

  img_data = torch.from_numpy(img.transpose((2, 0, 1))).float()
  c, w, h = img_data.size()
  in_data = img_data.reshape(1, 4, w, h) - 128
  in_data = in_data.cuda()
  with torch.no_grad():
      pred = net(Variable(in_data))

  out_img_data = (pred.data[0].cpu().float()).round().clamp(0, 255)
  out_img = out_img_data.numpy().astype(np.uint8).transpose(1, 2, 0)
  return out_img[:, :, 1:]


if __name__ == '__main__':
    # ./out/rain12/enhanced
    # ./out/rain12/dehazed_twice

    # ./out/rain200H/enhanced
    # ./out/rain200H/dehazed_twice

    # ./out/rain200L/enhanced
    # ./out/rain200L/dehazed_twice

    # ./out/rain12600/enhanced
    # ./out/rain12600/dehazed_twice

    # ./out/real-world-images/enhanced
    # ./out/real-world-images/dehazed_twice


    # -------------------------------------------------------------------------------------


    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./out/real-world-images/enhanced')
    parser.add_argument('--output_dir', type=str, default='./out/real-world-images/dehazed_twice')
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        os.system('rm -rf {}'.format(args.output_dir))

    os.mkdir(args.output_dir)

    net = GCANet(in_c=4, out_c=4, only_residual=False).cuda()
    data = torch.load('/home/franz/derain/net_epoch_15.pth')
    net.load_state_dict(data)

    for filepath in tqdm(glob(os.path.join(args.input_dir, '*.png'))):
        dehazed_image = restore_image(net, filepath)  # RGB
        dehazed_image = cv2.cvtColor(dehazed_image, cv2.COLOR_RGB2BGR)  # BGR
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(filepath)), dehazed_image)
