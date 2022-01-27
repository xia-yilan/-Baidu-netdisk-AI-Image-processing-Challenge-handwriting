import argparse
import os.path
import os
from posixpath import join
import random
import time
import datetime
import sys
import paddle.nn.functional as F
import numpy as np
import warnings

from utils import load_pretrained_model
import paddle
import paddle.distributed as dist
from transforms import sample_images, Resize, Normalize, RandomHorizontalFlip
from dataset import Dataset
# from utils import load_pretrained_model

from model_parallel import SAGAN


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default='/mnt/sdb/xyl/baidu/dehw_train_dataset/')

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=12
    )

    parser.add_argument(
        '--max_epochs',
        dest='max_epochs',
        help='max_epochs',
        type=int,
        default=3000
    )

    parser.add_argument(
        '--save_path',
        dest='save_path',
        help='save_path',
        type=str,
        default='train_result_cropandresizecrop3_l1'
    )

    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='log_iters',
        type=int,
        default=100
    )

    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='save_interval',
        type=int,
        default=10
    )

    parser.add_argument(
        '--sample_interval',
        dest='sample_interval',
        help='sample_interval',
        type=int,
        default=100
    )

    parser.add_argument(
        '--seed',
        dest='seed',
        help='random seed',
        type=int,
        default=1234
    )

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='load pretrained model',
        type=str,
        default='/home/xyl/EraseNet-paddle-master/EraseNet-paddle-master/work/train_result_cropandresizecrop3_l1/model/epoch_770/'
    )

    return parser.parse_args()


def main(args):
    # paddle.FLAGS_fraction_of_gpu_memory_to_use = 0.92
    # paddle.FLAGS_eager_delete_tensor_gb = 0.0
    # warnings.filterwarnings('ignore')
    dist.init_parallel_env()
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    transforms = [
        Resize(target_size=(512, 512)),
        Normalize(),
        RandomHorizontalFlip()
    ]
    dataset = Dataset(dataset_root=args.dataset_root, transforms=transforms)

    dataloader = paddle.io.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      num_workers=0,
                                      shuffle=True,
                                      return_list=True)

    model = SAGAN()

    if args.pretrained != '':
        load_pretrained_model(model.generator, args.pretrained)


    # model.set_state_dict(paddle.load(os.path.join(args.pretrained, 'modelG.pdparams'))['netG'])
    # model.set_state_dict(paddle.load(os.path.join(args.pretrained, 'modelD.pdparams'))['netD'])
    # model.generator.state_dict(paddle.load(os.path.join(args.pretrained, 'modelG.pdparams'))['netG'])
    # model.loss_fn.discriminator.state_dict(paddle.load((os.path.join(args.pretrained, 'modelD.pdparams'))['netD']))


    dpmodel = paddle.DataParallel(model, find_unused_parameters=True)  # find_unused_parameters=True


    # optimizer
    G_optimizer = paddle.optimizer.Adam(learning_rate=0.0001,
                                        parameters=dpmodel._layers.generator.parameters(),
                                        beta1=0.5,
                                        beta2=0.9,
                                        weight_decay=0.01)

    D_optimizer = paddle.optimizer.Adam(learning_rate=0.00001,
                                        parameters=dpmodel._layers.loss_fn.discriminator.parameters(),
                                        beta1=0,
                                        beta2=0.9,
                                        weight_decay=0.01)
    # dpmodel._layers.generator.set_state_dict(paddle.load(os.path.join(args.pretrained, 'modelG.pdparams'))['netG'])
    # dpmodel._layers.loss_fn.discriminator.set_state_dict(paddle.load(os.path.join(args.pretrained, 'modelD.pdparams'))['netD'])
    # G_optimizer.set_state_dict(paddle.load(os.path.join(args.pretrained, 'model_opG.pdopt'))['optimG'])
    # D_optimizer.set_state_dict(paddle.load(os.path.join(args.pretrained, 'model_opD.pdopt'))['optimD'])
    #
    # epoch_now = paddle.load(os.path.join(args.pretrained, 'model_opG.pdopt'))['epoch']



    count = 1
    paddle.device.cuda.empty_cache()
    prev_time = time.time()
    for epoch in range(1, args.max_epochs + 1):  #for epoch in range(1, args.max_epochs + 1):
        # paddle.device.cuda.empty_cache()
        for i, data_batch in enumerate(dataloader):
            input_img = data_batch[0]
            gt_img = data_batch[1]
            mask_img = data_batch[2]

            # model inference

            G_loss, holeLoss, validAreaLoss, mask_loss, D_loss, fake_images = \
                dpmodel(input_img, mask_img, gt_img, count, epoch, D_optimizer)

            # D_optimizer.clear_grad()
            # D_loss.backward(retain_graph=True)
            # D_optimizer.step()

            G_optimizer.clear_grad()
            G_loss.backward()
            G_optimizer.step()

            # determine approximate time left
            batches_done = epoch * len(dataloader) + 1
            batches_left = args.max_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            if i % args.log_iters == 0:  # args.log_iters=100 args.max_epochs=100
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f] ETA: %s" %
                                 (epoch, args.max_epochs,
                                  i, len(dataloader),
                                  G_loss.item(),
                                  time_left))
            # if i % args.sample_interval == 0:
            #     sample_images(epoch, i, fake_images, gt_img, input_img, args)

        print("G_loss={}, holeLoss={}, validAreaLoss={}, mask_loss={}, D_loss={}".format
              (G_loss.item(), holeLoss.item(), validAreaLoss.item(), mask_loss.item(), D_loss.item()))
        # if epoch % args.sample_interval == 0:
        if epoch % 10 == 0:
            current_save_dir = os.path.join(args.save_path,"model", f"epoch_{epoch}")
            # if not os.path.exists(current_save_dir):
            os.makedirs(current_save_dir, exist_ok=True)
            gen_dir = os.path.join(current_save_dir, "modelG.pdparams")
            dis_dir = os.path.join(current_save_dir, "modelD.pdparams")


            paddle.save({'netG': dpmodel._layers.generator.state_dict()}, gen_dir)
            paddle.save({'netD': dpmodel._layers.loss_fn.discriminator.state_dict()}, dis_dir)
            paddle.save({'epoch': epoch,
                         'optimG': G_optimizer.state_dict()}, os.path.join(current_save_dir,"model_opG.pdopt"))
            paddle.save({'epoch': epoch,
                         'optimD': D_optimizer.state_dict()}, os.path.join(current_save_dir, "model_opD.pdopt"))

            # paddle.save(model.generator.state_dict(),
            #             os.path.join(current_save_dir, "model.pdparams"))
            # paddle.save(model.loss_fn.discriminator.state_dict(),
            #             os.path.join(current_save_dir, "modelD.pdparams"))
            # paddle.save({'netG': model.generator.state_dict()}, os.path.join(current_save_dir, "model.pdparams")
            #             G_optimizer.state_dict(),
            #             D_optimizer.state_dict(),
            #             os.path.join(current_save_dir, "model_op.pdopt"))



if __name__ == "__main__":
    args = parse_args()
    # dist.spawn(main(args),nprocs=2, gpus='0,1')
    # dist.spawn(main(args,), nprocs=2, gpus='0,1', join=True)
    main(args)