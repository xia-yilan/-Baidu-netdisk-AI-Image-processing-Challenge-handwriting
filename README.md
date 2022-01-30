# 百度网盘AI大赛——图像处理挑战赛：手写文字擦除第十六名方案
   
比赛连接：[百度网盘AI大赛：手写文字擦除(赛题二)](https://aistudio.baidu.com/aistudio/competition/detail/129/0/introduction)

## 一、比赛介绍
百度网盘AI大赛——图像处理挑战赛是百度网盘开放平台发起的计算机视觉领域挑战赛，鼓励选手产出基于飞桨（paddlepaddle）框架的开源模型方案。手写文字擦除，旨在能区分出打印文字和手写文字，并擦除手写文字。

## 二、数据处理
官方提供训练集1081对，A榜测试集200张，B榜测试集200张。训练需要的黑白mask需要自己生成，具体操作是由训练集的images和gt图像做差，并设置差异的阙值，见代码gen_mask.py
![](https://aistudio.baidu.com/aistudio/projectdetail/3279736)

