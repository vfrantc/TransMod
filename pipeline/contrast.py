import os
from glob import glob
from tqdm import tqdm
import argparse

from skimage import exposure
from skimage.io import imread, imsave

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./out/rain12/dehazed')
    parser.add_argument('--output_dir', type=str, default='./out/rain12/equalized')
    parser.add_argument('--method', type=str, default='equalize', choices=['CLAHE', 'equalize', 'stretch'])
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        os.system('rm -rf {}'.format(args.output_dir))
    os.mkdir(args.output_dir)

    print(f"Equalizing... from {args.input_dir} to {args.output_dir}")
    for filepath in tqdm(glob(os.path.join(args.input_dir, '*.png'))):
        image = imread(filepath)

        if args.method.lower() == 'clahe':
            image = exposure.equalize_adapthist(image, clip_limit=0.03)
        elif args.method.lower() == 'equalize':
            image = exposure.equalize_hist(image)
        elif args.method.lower() == 'stretch':
            image = exposure.rescale_intensity(image, out_range=(0, 255))

        imsave(os.path.join(args.output_dir, os.path.basename(filepath)), image)