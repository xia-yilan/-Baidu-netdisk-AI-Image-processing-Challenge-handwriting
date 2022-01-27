from numpy.lib.arraypad import pad
import paddle
import paddle.nn as nn
from paddle import autograd
import paddle.nn.functional as F
from models.discriminator import Discriminator_STE
from PIL import Image
import numpy as np

def gram_matrix(feat):
    (b, c,h, w) = feat.shape
    feat = feat.reshape([b, c, h*w])
    feat_t = feat.transpose((0,2,1))
    gram = paddle.bmm(feat, feat_t) / (c * h * w)
    return gram

def viaual(image):
    im = image.transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()


def dice_loss(input, target):
    input = F.sigmoid(input)

    input = input.reshape([input.shape[0], -1]) # .contiguous()
    target = target.reshape([target.shape[0], -1])
    
    a = paddle.sum(input * target, 1)
    b = paddle.sum(input * input, 1) + 0.001
    c = paddle.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = paddle.mean(d)
    return 1-dice_loss

class LossWithGAN_STE(nn.Layer):
    def __init__(self, extractor, Lamda, lr, betasInit=(0.5, 0.9)):
        super(LossWithGAN_STE, self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.discriminator = Discriminator_STE(3) #local global sn patch gan
        self.D_optimizer = paddle.optimizer.Adam(learning_rate=lr, 
                                                    parameters = self.discriminator.parameters(), 
                                                    beta1 = betasInit[0],
                                                    beta2 = betasInit[1],
                                                    weight_decay=0.01)
        self.cudaAvailable = paddle.device.is_compiled_with_cuda()
        self.numOfGPUs = paddle.device.cuda.device_count()
        self.lamda = Lamda
    

    def forward(self, input, mask, x_o1, x_o2, x_o3, output, mm, gt, count, epoch):

        D_real = self.discriminator(gt,mask)
        D_real = D_real.mean().sum() * -1
        D_fake = self.discriminator(output, mask)
        D_fake = D_fake.mean().sum() * 1
        D_loss = paddle.mean(F.relu(1. + D_real)) + paddle.mean(F.relu(1. + D_fake))

        D_fake = -paddle.mean(D_fake)
        self.D_optimizer.clear_grad()
        D_loss.backward(retain_graph=True)

        self.D_optimizer.step()

        output_comp = mask * input + (1 - mask) * output
        holeLoss = 10*self.l1((1 - mask) * output, (1 - mask) * gt)
        validAreaLoss = 2*self.l1(mask * output, mask * gt)
        mask_loss = dice_loss(mm, 1-mask)
        masks_a = F.interpolate(mask, scale_factor=0.25)
        masks_b = F.interpolate(mask, scale_factor=0.5)
        imgs1 = F.interpolate(gt, scale_factor=0.25)
        imgs2 = F.interpolate(gt, scale_factor=0.5)
        msrloss = 8 * self.l1((1-mask)*x_o3,(1-mask)*gt) + 0.8*self.l1(mask*x_o3, mask*gt)+\
                    6 * self.l1((1-masks_b)*x_o2,(1-masks_b)*imgs2)+1*self.l1(masks_b*x_o2,masks_b*imgs2)+\
                    5 * self.l1((1-masks_a)*x_o1,(1-masks_a)*imgs1)+0.8*self.l1(masks_a*x_o1,masks_a*imgs1)
        feat_output_comp = self.extractor(output_comp)
        feat_output = self.extractor(output)
        feat_gt = self.extractor(gt)

        prcLoss = 0.0

        # '3':'relu1',
        # '8':'relu2',
        # '13':'relu3',
        maps = ['relu1','relu2','relu3']
        for i in range(3):
            prcLoss += 0.01 * self.l1(feat_output[maps[i]], feat_gt[maps[i]])
            prcLoss += 0.01 * self.l1(feat_output_comp[maps[i]], feat_gt[maps[i]])

        styleLoss = 0.0
        for i in range(3):
            styleLoss += 120 * self.l1(gram_matrix(feat_output[maps[i]]), gram_matrix(feat_gt[maps[i]]))
            styleLoss += 120 * self.l1(gram_matrix(feat_output_comp[maps[i]]), gram_matrix(feat_gt[maps[i]]))
        
        GLoss = msrloss + holeLoss + validAreaLoss + prcLoss + styleLoss + 0.1 * D_fake + mask_loss
        # GLoss = holeLoss + validAreaLoss + mask_loss

        return GLoss.sum(), holeLoss.sum(), validAreaLoss.sum(), mask_loss.sum()
        # return GLoss