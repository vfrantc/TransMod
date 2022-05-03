import os
from glob import glob
from tqdm import tqdm
import argparse
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./out/rain12/dehazed')
    parser.add_argument('--output_dir', type=str, default='./out/rain12/equalized')
    parser.add_argument('--radius', type=int, default=32)
    parser.add_argument('--eps', type=float, default=0.1)
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        os.system('rm -rf {}'.format(args.output_dir))
    os.mkdir(args.output_dir)

    print(f"Filter... from {args.input_dir} to {args.output_dir}")
    for filepath in tqdm(glob(os.path.join(args.input_dir, '*.png'))):
        image = cv2.imread(filepath)
        image = cv2.ximgproc.guidedFilter(image, image, args.radius, args.eps)
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(filepath)), image)