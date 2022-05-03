import os
from glob import glob
import cv2
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize images')
    parser.add_argument('--input_dir_lle', type=str, default='.', help='Directory to summarize')
    parser.add_argument('--input_dir_dehazed', type=str, default='.', help='Directory to summarize')
    parser.add_argument('--output_dir', type=str, default='summary.png', help='Output file name')
    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for file_lle in tqdm(glob(os.path.join(args.input_dir_lle, '*.png'))):
        file_dehazed = file_lle.replace(args.input_dir_lle, args.input_dir_dehazed)
        if not os.path.exists(file_dehazed):
            continue
        img_lle = cv2.imread(file_lle)
        img_dehazed = cv2.imread(file_dehazed)

        img_dehazed = cv2.resize(img_dehazed, (img_lle.shape[1], img_lle.shape[0]))
        sume = cv2.addWeighted(img_lle, 0.3, img_dehazed, 0.7, 0)
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(file_lle)), sume)
