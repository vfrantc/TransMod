import time # python library for time, it is a standard
import torch # torch is library for tensor and deeplearning functions
import argparse # parsing of standard input
import torch.nn as nn # neural networks
from torch.utils.data import DataLoader # standard class to implement
from val_data_functions import ValData # ValData me
from utils import validation # some function
import os
import numpy as np
import random
# the actual model
from transweather_model import Transweather

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
# batch_size for validation
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
# something experimental, !!! what is this?
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
args = parser.parse_args()

val_batch_size = args.val_batch_size
exp_name = args.exp_name

#set seed
seed = args.seed
if seed is not None:
    # stand
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

# --- Set category-specific hyper-parameters  --- #
val_data_dir = './data/test/' # take the data

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Validation data loader --- #

val_filename = 'raindroptest1a.txt' ## This text file should contain all the names of the images and must be placed in ./data/test/ directory
val_data_loader = DataLoader(ValData(val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=8)

# --- Define the network --- #

net = Transweather()
net = nn.DataParallel(net, device_ids=device_ids)

# --- Load the network weight --- #
net.load_state_dict(torch.load('./{}/best'.format(exp_name))

# --- Use the evaluation model in testing --- #
net.eval()
if os.path.exists('./{}_results/{}/'.format(category,exp_name))==False: 	
	os.mkdir('./{}_results/{}/'.format(category,exp_name))	
	os.mkdir('./{}_results/{}/rain/'.format(category,exp_name))
print('--- Testing starts! ---')
start_time = time.time()
# so it computes both psnr and ssim on the validation,
val_psnr, val_ssim = validation(net, val_data_loader, device, category, exp_name, save_tag=True)
end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(end_time))
