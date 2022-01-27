import argparse
import glob
import os.path
from dataset import Dataset_test
from transforms import Normalize, Resize
import paddle
import paddle.nn as nn
import cv2
import numpy as np
from models.sa_gan import STRNet
import time
import os
from utils import load_pretrained_model
from PIL import Image
from model_parallel import SAGAN

def parse_args():
    parser = argparse.ArgumentParser(description='Model testing')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default='/mnt/sdb/xyl/baidu/dehw_testA_dataset/images/')
#/home/xyl/EraseNet-paddle-master/EraseNet-paddle-master/work/val/images/
#/mnt/sdb/xyl/baidu/dehw_testA_dataset/images/

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default='checkpoint/modelG.pdparams')

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=1
    )

    parser.add_argument(
        '--save_path',
        dest='save_path',
        help='save_path',
        type=str,
        default='test_result'
    )


    return parser.parse_args()

def single_forward(inp, model):
    """PyTorch model forward (single test), it is just a simple warpper
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model
    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    with paddle.no_grad():
        model_output, mm = crop_forward_1024(inp, model)
        # if isinstance(model_output, list) or isinstance(model_output, tuple):
        #     output = model_output[0]
        # else:
        #     output = model_output
    # output = model_output.float().cpu()
    return model_output, mm

def flipx4_forward(inp, model):
    """Flip testing with X4 self ensemble, i.e., normal, flip H, flip W, flip H and W
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model
    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    # normal
    output_f, mm_f = single_forward(inp, model)

    # flip W
    output, mm = single_forward(paddle.flip(inp, (-1, )), model)
    output_f = output_f + paddle.flip(output, (-1, ))
    mm_f = mm_f + paddle.flip(mm, (-1, ))
    # # flip H
    # output, mm = single_forward(paddle.flip(inp, (-2, )), model)
    # output_f = output_f + paddle.flip(output, (-2, ))
    # mm_f = mm_f + paddle.flip(mm, (-2,))
    # # flip both H and W
    # output, mm = single_forward(paddle.flip(inp, (-2, -1)), model)
    # output_f = output_f + paddle.flip(output, (-2, -1))
    # mm_f = mm_f + paddle.flip(mm, (-2,-1))

    return output_f / 2, mm_f / 2

def crop_forward_1024(x, model):

    h, w = x.shape[-2], x.shape[-1]
    if h < 1024 or w < 1024:
        mul = 512

    # elif h>4096 or w>4096:
    #     mul = 2048

    else:
        mul = 1024 #cut 1024*1024

    inputlist = []
    outputlist = []
    mmlist = []
    range_y = np.arange(0, h-mul, mul) #[0,1024,...,]
    range_x = np.arange(0, w-mul, mul)
    if range_y[-1] != h - mul:
        range_y = np.append(range_y, h-mul)
    if range_x[-1] != w - mul:
        range_x = np.append(range_x, w-mul)
    sz = len(range_y) * len(range_x)

    for yi in range_y:
        for xi in range_x:
            patch = x[:, :, yi:(yi + mul), xi:(xi + mul)]
            inputlist.append(patch)


    with paddle.no_grad():
        # for i in range(sz):
        #     _, _, _, output_batch, mm = model(inputlist[i])
        #     outputlist.append(output_batch)
        #     mmlist.append(mm)

        input_batch = paddle.concat(inputlist, axis=0)
        _, _, _, output_batch, mm = model.generator(input_batch)

        outputlist.extend(output_batch.chunk(sz, axis=0))
        mmlist.extend(mm.chunk(sz, axis=0))

        output = paddle.zeros_like(x)
        mm_out = paddle.zeros_like(x)
        weight = paddle.zeros_like(x)

        i = 0
        for yi in range_y:
            for xi in range_x:
                output[:, :, yi:(yi + mul), xi:(xi + mul)] = output[:, :, yi:(yi + mul), xi:(xi + mul)] + outputlist[i]
                mm_out[:, :, yi:(yi + mul), xi:(xi + mul)] = mm_out[:, :, yi:(yi + mul), xi:(xi + mul)] + mmlist[i]
                weight[:, :, yi:(yi + mul), xi:(xi + mul)] = weight[:, :, yi:(yi + mul), xi:(xi + mul)] + 1
                i = i + 1

    return output, mm_out



def main(args):
    # model = STRNet(3)
    model = SAGAN()

    # model.eval()

    # paddle.set_printoptions(threshold=500000000)

    # if args.pretrained is not None:
    #     load_pretrained_model(model, args.pretrained)

    model.generator.set_state_dict(paddle.load(args.pretrained)['netG'])

    path = args.dataset_root
    path_imgs = os.listdir(path)
    path_imgs.sort()
    all_time = []


    for i in path_imgs:
        img = cv2.imread(os.path.join(path, i)).astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_raw = img

        if (img.shape[0]) >= 4096 or img.shape[1] >= 4096:  #img.shape[0]:h  img.shape[1]:w
            if img.shape[1] >= 3072:
                img_h = img.shape[0] - img.shape[0] % 1024 - 1024
                img_w = img.shape[1] - img.shape[1] % 1024 - 1024


            elif 2048 <= img.shape[1] < 3072:
                img_h = img.shape[0] - img.shape[0] % 1024
                img_w = img.shape[1] - img.shape[1] % 1024
            else:
                img_h = img.shape[0]
                img_w = img.shape[1]
            img = cv2.resize(img, dsize=(img_w, img_h), interpolation=cv2.INTER_CUBIC)

        if (img.shape[0]) >= 3072 or img.shape[1] >= 3072:
            if img.shape[1] < 2048 or img.shape[0] < 2048:
                img_h = img.shape[0]
                img_w = img.shape[1]
            else:
                img_h = img.shape[0] - img.shape[0] % 1024
                img_w = img.shape[1] - img.shape[1] % 1024

            img = cv2.resize(img, dsize=(img_w, img_h), interpolation=cv2.INTER_CUBIC)


        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, ...]  #b,c,h,w
        img = paddle.to_tensor(img, dtype='float32')
        paddle.device.cuda.empty_cache()

        start = time.time()
        img_out, mm = flipx4_forward(img, model)
        end = time.time()

        paddle.device.cuda.empty_cache()
        time_one = end - start
        all_time.append(time_one)

        print('The running time of an image is : {:2f} s'.format(time_one))

        img_out = img_out.squeeze(0)
        mm = mm.squeeze(0)

        img_out = paddle.clip(img_out * 255.0, 0, 255)
        img_out = paddle.transpose(img_out, [1, 2, 0])  #h,w,c
        mm = paddle.transpose(mm, [1, 2, 0]) #h,w,c

        img_out = img_out.numpy()
        mm = mm.numpy()

        img_out = cv2.resize(img_out, dsize=(img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_CUBIC)
        mm = cv2.resize(mm, dsize=(img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_CUBIC)


        mm[mm>=0] = 1
        mm[mm<0] = 0
        mm = cv2.dilate(mm, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=2)

        img_out = img_out * mm + (1 - mm) * img_raw

        save_path = args.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        Image.fromarray(np.uint8(img_out)).convert("RGB").save(os.path.join(args.save_path, i.replace('jpg', 'png')))
        # Image.fromarray(np.uint8(mm * 255)).convert("L").save('pic_test_mm/mm{}.png'.format(i))
    print("alltime is {}".format(np.mean(all_time)))



if __name__=='__main__':
    args = parse_args()
    main(args)