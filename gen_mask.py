import numpy as np
import imageio
import argparse
import os
import glob
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='/mnt/sdb/xyl/baidu/dehw_train_dataset/images/')
parser.add_argument('--gt_path', type=str, default='/mnt/sdb/xyl/baidu/dehw_train_dataset/gts/')
parser.add_argument('--mask_path', type=str, default='/home/xyl/EraseNet-paddle-master/mask/')
parser.add_argument('--threshold', type=int, default=25)
args = parser.parse_args()

def mask(input_image, gt_image, threshold = 25):
    diff_image = np.abs(input_image.astype(np.float32) - gt_image.astype(np.float32))
    mean_image = np.mean(diff_image, axis=-1)
    mask = np.greater(mean_image, threshold).astype(np.uint8)
    #mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=0)
    mask = mask * 255
    return mask

def main():
    if not os.path.isdir(args.mask_path):
        os.makedirs(args.mask_path)
    input_images_list = glob.glob(args.input_path + '*.jpg')
    for input_image_path in input_images_list:
        input_image = imageio.imread(input_image_path)
        image_name = input_image_path.split('/')[-1].split('.')[0]
        gt_image = imageio.imread(input_image_path.replace('images', 'gts').replace('.jpg', '.png'))[:, :, 0:3]
        mask_image = mask(input_image, gt_image)
        imageio.imwrite(args.mask_path + image_name + '.png', mask_image)
        print(input_image_path)
    
main()