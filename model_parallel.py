from models.sa_gan import STRNet
from losses import pre_network
from models.discriminator import Discriminator_STE
import paddle.nn as nn
import paddle
import paddle.nn.functional as F


def gram_matrix(feat):
    (b, c, h, w) = feat.shape
    feat = feat.reshape([b, c, h * w])
    feat_t = feat.transpose((0, 2, 1))
    gram = paddle.bmm(feat, feat_t) / (c * h * w)
    return gram


def dice_loss(input, target):
    input = F.sigmoid(input)

    input = input.reshape([input.shape[0], -1])  # .contiguous()
    target = target.reshape([target.shape[0], -1])

    a = paddle.sum(input * target, 1)
    b = paddle.sum(input * input, 1) + 0.001  # b = paddle.sum(input * target, 1) + 0.001
    c = paddle.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = paddle.mean(d)
    return 1 - dice_loss


class My_LossWithGAN_STE(nn.Layer):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.discriminator = Discriminator_STE(3)
        self.extractor = pre_network(pretrained='./vgg.pdparams')

    def forward(self, input, mask, x_o1, x_o2, x_o3, output, mm, gt, count, epoch, optim_D):

        D_real = self.discriminator(gt, mask)
        D_real = D_real.mean().sum() * -1
        # D_fake = self.discriminator(output, mask)
        D_fake = self.discriminator(output.detach(), mask)
        D_fake = D_fake.mean().sum() * 1
        D_loss = paddle.mean(F.relu(1. + D_real)) + paddle.mean(F.relu(1. + D_fake))

        # D_fake = -paddle.mean(D_fake)
        optim_D.clear_grad()
        # D_loss.backward(retain_graph=True)
        D_loss.backward()
        optim_D.step()

        D_fake = -paddle.mean(self.discriminator(output, mask))
        output_comp = (1-mask) * input + mask * output  # 0black 1white gaidong
        holeLoss = 1.2*100*10 * self.l1(mask * output, mask * gt)
        validAreaLoss = 2*2 * self.l1((1-mask) * output,(1-mask) * gt)
        mask_loss = 1.2*2*2*dice_loss(mm, mask)
        # masks_a = F.interpolate(mask, scale_factor=0.25)
        # masks_b = F.interpolate(mask, scale_factor=0.5)
        # imgs1 = F.interpolate(gt, scale_factor=0.25)
        # imgs2 = F.interpolate(gt, scale_factor=0.5)
        # msrloss = 8 * self.l1(mask * x_o3, mask * gt) + 0.8 * self.l1((1-mask) * x_o3, (1-mask) * gt) + \
        #           6 * self.l1((masks_b) * x_o2, (masks_b) * imgs2) + 1 * self.l1((1-masks_b) * x_o2,
        #                                                                                  (1-masks_b) * imgs2) + \
        #           5 * self.l1((masks_a) * x_o1, (masks_a) * imgs1) + 0.8 * self.l1((1 - masks_a) * x_o1,
        #                                                                                    (1 - masks_a) * imgs1)

        # feat_output_comp = self.extractor(output_comp)
        # feat_output = self.extractor(output)
        # feat_gt = self.extractor(gt)

        # prcLoss = 0.0

        # maps = ['relu1', 'relu2', 'relu3']
        # for i in range(3):
        #     prcLoss += 0.01 * self.l1(feat_output[maps[i]], feat_gt[maps[i]])
        #     prcLoss += 0.01 * self.l1(feat_output_comp[maps[i]], feat_gt[maps[i]])
        #
        # styleLoss = 0.0
        # for i in range(3):
        #     styleLoss += 120 * self.l1(gram_matrix(feat_output[maps[i]]), gram_matrix(feat_gt[maps[i]]))
        #     styleLoss += 120 * self.l1(gram_matrix(feat_output_comp[maps[i]]), gram_matrix(feat_gt[maps[i]]))
        #
        # GLoss = msrloss + holeLoss + validAreaLoss + prcLoss + styleLoss + 0.1 * D_fake + mask_loss
        GLoss =   holeLoss + validAreaLoss + 0.1 * D_fake + mask_loss

        return GLoss.sum(), holeLoss.sum(), validAreaLoss.sum(), mask_loss.sum(), D_loss


class SAGAN(nn.Layer):
    def __init__(self):
        super().__init__()
        self.generator = STRNet(3)

        self.loss_fn = My_LossWithGAN_STE()

    def forward(self, input_img, mask_img, gt_img, count, epoch, optim_D):
        x_o1, x_o2, x_o3, fake_images, mm = self.generator(input_img)
        G_loss, holeLoss, validAreaLoss, mask_loss, D_loss = \
            self.loss_fn(input_img, mask_img, x_o1, x_o2, x_o3, fake_images, mm, gt_img, count, epoch, optim_D)
        return G_loss, holeLoss, validAreaLoss, mask_loss, D_loss, fake_images
