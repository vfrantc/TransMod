import os
from glob import glob
from tqdm import tqdm
import argparse

import cv2
import numpy as np

def gamma_correction(image, w=0.5):
    def extract_value_channel(color_image):
        color_image = color_image.astype(np.float32) / 255.
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        return np.uint8(v * 255)

    def get_pdf(gray_image):
        height, width = gray_image.shape
        pixel_count = height * width
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        return hist / pixel_count

    def set_value_channel(color_image, value_channel):
        value_channel = value_channel.astype(np.float32) / 255
        color_image = color_image.astype(np.float32) / 255.
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        color_image[:, :, 2] = value_channel
        color_image = np.array(cv2.cvtColor(color_image, cv2.COLOR_HSV2BGR) * 255, dtype=np.uint8)
        return color_image

    is_colorful = len(image.shape) >= 3
    img = extract_value_channel(image) if is_colorful else image
    img_pdf = get_pdf(img)
    max_intensity = np.max(img_pdf)
    min_intensity = np.min(img_pdf)
    w_img_pdf = max_intensity * (((img_pdf - min_intensity) / (max_intensity - min_intensity)) ** w)
    w_img_cdf = np.cumsum(w_img_pdf) / np.sum(w_img_pdf)
    l_intensity = np.arange(0, 256)
    l_intensity = np.array([255 * (e / 255) ** (1 - w_img_cdf[e]) for e in l_intensity], dtype=np.uint8)
    enhanced_image = np.copy(img)
    height, width = img.shape
    for i in range(0, height):
        for j in range(0, width):
            intensity = enhanced_image[i, j]
            enhanced_image[i, j] = l_intensity[intensity]
    enhanced_image = set_value_channel(image, enhanced_image) if is_colorful else enhanced_image
    return enhanced_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./out/rain12/input/')
    parser.add_argument('--output_dir', type=str, default='./out/rain12/gamma_corrected/')
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        os.system('rm -rf {}'.format(args.output_dir))
    os.mkdir(args.output_dir)

    print(f"Adaptive gamma correction... from {args.input_dir} to {args.output_dir}")
    for filepath in tqdm(glob(os.path.join(args.input_dir, '*.png'))):
        image = cv2.imread(filepath, 1)
        image = gamma_correction(image)
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(filepath)), image)