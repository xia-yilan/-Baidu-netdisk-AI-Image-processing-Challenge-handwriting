# 百度网盘AI大赛——图像处理挑战赛：手写文字擦除第十六名方案
   
比赛连接：[百度网盘AI大赛：手写文字擦除(赛题二)](https://aistudio.baidu.com/aistudio/competition/detail/129/0/introduction)

## 一、比赛介绍
百度网盘AI大赛——图像处理挑战赛是百度网盘开放平台发起的计算机视觉领域挑战赛，鼓励选手产出基于飞桨（paddlepaddle）框架的开源模型方案。手写文字擦除，旨在能区分出打印文字和手写文字，并擦除手写文字。

## 二、数据处理
官方提供训练集1081对，A榜测试集200张，B榜测试集200张。训练需要的黑白mask需要自己生成，具体操作是由训练集的images和gt图像做差，并设置差异的阙值，见代码gen_mask.py

![](https://ai-studio-static-online.cdn.bcebos.com/cd3e3041e72843f9b702c3c2314067d6fdb0fec237654086be1f3ca79f031b14)

## 三、模型介绍
采用的baseline为论文**EraseNet: End-to-End Text Removal in the Wild**提出的模型。选取其作为baseline的原因有两个：一是考虑到模型需要转换成paddlepaddle框架，而大赛提供了该模型的paddlepaddle框架代码，能节约时间；二是由于测试图片的分辨率很大，最大可到3300×4628，并且补全任务为擦除黑白试卷中的手写字体和其他颜色涂抹，任务简单，固可以采用更深度的网络，将图片下采样32倍，不仅可以节约显存也能解码出好的效果。

![](https://tva1.sinaimg.cn/large/008i3skNgy1gxgogc9id0j31m80u0tds.jpg)
1、模型由两阶段生成器网络和局部全局鉴别器构成，此外，还连接了一个分割头准确指示手写文本区域。
2、生成器设计了残差快，空洞卷积层和跳跃连接等结构。

## 四、实践调优
**训练策略**
考虑图片训练时的大小和分辨率之间的关系，训练时模型输入图片大小为512×512，对于高分辨率图片采用简单的裁剪致使文字太大，mask生成效果不好，固采用先resize再crop。具体详情见代码。训练脚本：train_parallel.py

**测试策略**
测试时采用的测试策略需要符合训练时的图片样式。输入图片尺寸需要为32的倍数，对小分辨率的裁成512×512大小，对于大分辨率的先resize成小一点的图片，再裁成1024×1024大小的图像块。具体详情见代码。测试脚本：test_crop.py

**参数策略**
训练主要选用l1损失，mask生成使用dice损失。

## 五、代码运行
1、下载数据集。获取地址：![](https://aistudio.baidu.com/aistudio/competition/detail/129/0/datasets)

2、生成训练黑白mask
```
python gen_mask.py
```
数据集目录如下所示：
```
| dehw_testA_dataset/
        -images
| dehw_train_dataset/
        -gts
        -images
        -mask
```

3、运行训练代码
```
python train_parallel.py
```

4、运行测试代码

下载预训练模型存放到./checkpoint(百度网盘链接：https://pan.baidu.com/s/1_ytGCIAlpWfDnpG2paGPuw  提取码：baid)
```
python test_crop.py
```
## 六、致谢及参考文献
感谢百度组织大赛，给选手们提供机会和参赛平台。

本次参赛的模型代码主要参考官方提供的baseline:[https://aistudio.baidu.com/aistudio/projectdetail/3257671](http://)

框架插图来源于论文**EraseNet: End-to-End Text Removal in the Wild**


