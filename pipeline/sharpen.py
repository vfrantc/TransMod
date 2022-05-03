import os
from glob import glob
from tqdm import tqdm
import argparse
import numpy as np
import cv2

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    # https://en.wikipedia.org/wiki/Unsharp_masking

    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./out/rain12/stretched')
    parser.add_argument('--output_dir', type=str, default='./out/rain12/deblurred')
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        os.system('rm -rf {}'.format(args.output_dir))
    os.mkdir(args.output_dir)

    print(f"Equalizing... from {args.input_dir} to {args.output_dir}")
    for filepath in tqdm(glob(os.path.join(args.input_dir, '*.png'))):
        image = cv2.imread(filepath)
        image = unsharp_mask(image)
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(filepath)), image)