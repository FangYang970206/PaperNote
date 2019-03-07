# GAN笔记——理论与实现

标签（空格分隔）： Anime GAN DCGAN WGAN ConditionalGAN pytorch

---
`GAN`这一概念是由`Ian Goodfellow`于2014年提出，并迅速成为了非常火热的研究话题，GAN的变种更是有上千种，深度学习先驱之一的`Yann LeCun`就曾说,"`GAN及其变种是数十年来机器学习领域最有趣的idea`"。那么什么是GAN呢？GAN的应用有哪些呢？GAN的原理是什么呢？怎样去实现一个GAN呢？本文将一一阐述。具体大纲如下：

<ul>
    <li> <a href="#A">1.什么是GAN？</a></li>
    <ul> 
    <li><a href="#A1">1.1 对抗思想——啵啵鸟与枯叶蝶</a></li>
    <li><a href="#A2">1.2 GAN思想——画画的演变</a></li>
    <li><a href="#A3">1.3 零和博弈（zero-sum game）</a></li>
    <li><a href="#A4">1.4 小结</a></li>
    </ul>
    <li> <a href="#B">2. GAN的应用</a> </li>
    <li> <a href="#C">3. GAN的原理</a> </li>
    <ul>
    <li> <a href="#C1">3.1 生成器是否可以自我训练？</a></li>
    <li> <a href="#C2">3.2 鉴别器是否可以自我训练？</a></li>
    <li> <a href="#C3">3.3 生成器、鉴别器和GAN的优缺点</a></li>
    <li> <a href="#C4">3.4 GAN背后的理论</a></li>
    </ul>
    <li> <a href="#D">4.实现DCGAN</a> [<a href="https://github.com/FangYang970206/Anime_GAN">Github链接</a>]</li>
    <ul>
    <li> <a href="#D1">4.1 model.py</a></li>
    <li> <a href="#D2">4.2 AnimeDataset.py</a></li>
    <li> <a href="#D3">4.3 utils.py</a></li>
    <li> <a href="#D4">4.4 main.py</a></li>
    </ul>
    <li> <a href="#E">5. 实现WGAN[<a href="https://github.com/FangYang970206/Anime_GAN/">Github链接</a>]</li>
        <li> <a href="#F">6.实现ConditionalGAN</a> [<a href="https://github.com/FangYang970206/Anime_GAN">Github链接</a>]</li>
    <ul>
    <li> <a href="#F1">6.1 CGAN.py</a></li>
    <li> <a href="#F2">6.2 utils.py</a></li>
    <li> <a href="#F3">6.3 main.py</a></li>
    <li> <a href="#F4">6.4 GUI.py</a></li>
    </ul>
    <li> <a href="#G">7. GAN小技巧</a></li>
    <li> <a href="#H">8. 参考</a></li>
    
</ul>

<!-- > * 1.什么是GAN？
> * 2.GAN的应用
> * 3.GAN的原理
> * 4.实现DCGAN[ \[Github链接\]][1]
> * 5.GAN小技巧
> * 6.参考
> * 7.未完待续（后期还会加一些其他的GAN） -->

<h2>
<a id="A">
 1. 什么是GAN？
</a>
</h2>

<a href="https://arxiv.org/abs/1406.2661">GAN</a>的英文全称是`Generative Adversarial Network`，中文名是生成对抗网络，它由两个部分组成，一个是生成器（generative），还有一个是鉴别器，与生成器是敌对（Adversarial）关系。对GAN有了初步了解，知道它有两个模块组成，下面通过事例来理解这两个模块的产生思想？

<h3>
<a id="A1">
1.1 对抗思想——啵啵鸟与枯叶蝶
</a>
</h3>

![image_1cjq51f9kp4b136b77218i61pg5m.png-336.9kB][2]
在生物进化的过程中，被捕食者会慢慢演化自己的特征，从而达到欺骗捕食者的目的，而捕食者也会根据情况调整自己对被捕食者的识别，共同进化，上图中的啵啵鸟和枯叶蝶就是这样的一种关系。生成器代表的是枯叶蝶，鉴别器代表的是啵啵鸟。它们的对抗思想与GAN类似，但GAN却有所不同。

<h3>
<a id="A2">
1.2 GAN思想——画画的演变
</a>
</h3>

GAN之所以有所不同，这里的原因是**GAN所作的工作与自然界的生物进化不同，它是已经知道最终鉴别的目标是什么样子，不知道假目标是什么样子，它会对生成器所产生的假目标做惩罚和对真目标进行奖励，这样鉴别器就知道什么目标是不好的假目标，什么目标是好的真目标，而生成器则是希望通过进化，产生比上一次更好的假目标，使鉴别器对自己的惩罚更小。以上是一个轮回，下一个轮回，鉴别器通过学习上一个轮回进化的假目标和真目标，再次进化对假目标的惩罚，而生成器不屈不挠，再次进化，直到以假乱真，与真目标一致，至此进化结束。**
![1.jpg-1766.9kB][3]
以上图为例，我们最开始画人物头像只知道有一个头的大致形状，有眼睛有鼻子等等，但画得不精致，后来通过找老师学习，画得更好了，有模有样，直到，我们画得与专门画头像的老师一样好。这里的`我们`就像是`生成器`，一步步进化（对应生成器不同的等级），这里的`老师`就像是`鉴别器`（*这里只是比喻说明，现实世界的老师已经是一个成熟的鉴别器，不需要通过假样本进行学习，这里有那个意思就行*）

<h3>
<a id="A3">
1.3 零和博弈（zero-sum game）
</a>
</h3>

玩过纸牌的人知道，赢家的快乐是建立在输家的痛苦之上，收益和损失的总和始终为0。生成器和鉴别器也是这样一对博弈关系：鉴别器惩罚生成器，鉴别器收益，生成器损失；生成器进化，使鉴别器对自己惩罚小，生成器收益，鉴别器损失。

<h3>
<a id="A4">
1.4 小结
</a>
</h3>

什么是GAN？GAN是由生成器和鉴别器两个部分组成，生成器的目的是生成假的目标，企图彻底骗过鉴别器的识别。而鉴别器通过学习真目标和假目标，提高自己的鉴别能力，不让假目标骗过自己。两者相互进化，相互博弈，一方进化，另一方损失，最后直到假目标与真目标很相似则停止进化。

<h2>
<a id="B">
 2. GAN的应用
</a>
</h2>

首先，我们要知道`结构化学习`（**Structured Learning**），GAN也是结构化学习的一种。与分类和回归类似，结构化学习也是需要找到一个X$\rightarrow$Y的映射，但结构化学习的输入和输出多种多样，可以是序列（sequence）到序列，序列到矩阵（matrix），矩阵到图（graph），图到树（tree）等等。这样，GAN的应用就十分广泛了。例如，**机器翻译**（machine translation）可以用GAN去做，如下图所示
![2.jpg-24.7kB][4]
还有**语音识别**（speech recognition）以及**聊天机器人**（chat-bot）
![4.jpg-25.9kB][5]
在图像方面，我们可以做**图像转图像**(image-to-image)，**彩色化**（colorization），还有**文本转图像**（text-to-image）
![5.jpg-58.5kB][6]
当然，GAN的应用远不止这么些，有非常有趣的变脸，图像自动打马赛克，自动生成多表情图像，年轻转年老等等，更多cool又`skr`的应用静待各位挖掘！ 

<h2>
<a id="C">
 3. GAN原理 
</a>
</h2>

GAN的最终目的是为了生成能够产生以假乱真的目标的生成器。那么，是不是一定要用GAN呢？生成器可不可以自己训练得到目标？鉴别器可不可以自己训练得到目标？我们先来看这两个问题，然后再深入讨论GAN。

<h3>
<a id="C1">
3.1 生成器是否可以自我训练？
</a>
</h3>

答案是肯定的，我们所熟知的`自编码器`（**Auto-Encoder**)以及`变分自编码器`（**Variational Auto-Encoder**)都是典型的生成器。输入通过Encoder编码成code，然后code通过Decoder重建原图，其中自编码器中的Decoder就是生成器，code可随机取值，产生不同的输出。
自编码器的结构如下：
![6.jpg-655.3kB][7]
变分自编码器的结构如下
![7.jpg-1203.8kB][8]
然后自编码器存在着问题，我们来看看下面这张图
![8.jpg-1793kB][9]
**生成器的问题**:*由于自编码器的目标是让重建误差越来越小，但从上图中，我们可以看出，其中1个pixel的error，自编码器是觉得ok的，我们是觉得不行，另外6个pixel的误差我们觉得能接受的，自编码器不能接受，误差所在的位置很重要，而生成器并不知道这一点，自编码器缺少理解像素点之间的空间相关性的能力。还有一点，就是自编码器所产生的图像是模糊的，不能够产生十分清晰的图像*，如下图所示
![image_1cjqr5jm9uor1b90hcujc51g2c7m.png-65kB][10]
所以说目前单凭生成器是很难生成非常高质量的图像的。
 
<h3>
<a id="C2">
3.2 鉴别器是否可以自我训练？
</a>
</h3>

答案也是肯定的。鉴别器是给定一个输入，输出一个[0,1]的置信度，越接近1则置信越高，越接近0则置信度越低，如图所示：
![9.jpg-1000.8kB][11]
鉴别器的优势在于它可以很轻易地捕捉到元素之间的相关性，例如自编码器中出现的像素问题就不会在鉴别器中出现，如图所示，用一个滤波器就解决了。
![10.jpg-1426.8kB][12]
现在来说说鉴别器要怎么样产生样本，参考下图：
![11.jpg-1619.7kB][13]
首先也需要随机生成负样本，然后与真实样本一起送入鉴别器进行训练，在循环迭代中，通过最大概率选出最好的负样本，再与真样本一起送入鉴别器进行训练，然而，看起来和GAN训练差不多一致，没啥问题，其实这里面还有存在着问题的。我们来看下面这张图：
![12.jpg-63.5kB][14]
**鉴别器的问题**：*鉴别器的训练是对真样本进行奖励，对负样本进行压低，也就是图中的绿色抬高，蓝色压低，这就造成了问题，我们要训练出好的鉴别器，训练过程需要随机采样出除绿色图像外所有的假样本，这样鉴别器就只会对真实样本的分布取高分，对其他分布取低分，这样才能训练的好，然后再高维空间中，这样的负样本采样过程其实是很难进行的，而且还有一个问题，生成样本的过程要枚举大量样本，才有可能出现一个与真样本分布相符的样本，通过求那个最大化概率问题求出最好的样本，这实在是过于繁琐*。

<h3>
<a id="C3">
3.3 生成器、鉴别器和GAN的优缺点
</a>
</h3>

通过上面的阐述，我们初步知道了它们的优缺点，下面这张ppt直观地给出了每个的优缺点，如图所示：
![13.jpg-48.1kB][15]
可以看出生成器和鉴别器的优缺点是可以互补的，这也就是GAN的优势。（**生成器+鉴别器**），下图介绍了GAN的优点，从两个角度出发。

> * 从鉴别器的角度出发，利用生成器去生成样本，去求解最大化问题
> * 从生成器角度出发，生成的样本依旧是逐个元素，但通过鉴别器可以得到全局性。

![14.jpg-52.7kB][16]
当然，GAN也是又缺点的，它是一种隐变量模型，可解释没有生成器和鉴别器强，另外GAN是不好进行训练。我在训练DAGAN的时候就成功造成了鉴别器的误差为0，无法进行反向传播更新梯度。

<h3>
<a id="C4">
3.4 GAN背后的理论
</a>
</h3>

对于生成器而言，它的目标是希望能够学习到真实样本的分布，这样就可以随机生成以假乱真的样本。如下图所示
![18.jpg-24.5kB][17]
如何去学习真实样本分布呢，这就需要用到`极大似然估计`(**Maximum Likelihood Estimation**)，先来看看下面这张图
![16.jpg-57kB][18]
我们需要随机采样真实分布中的数据，通过学习$P(x;\theta)$中的$\theta$，希望$P(x;\theta)$越接近$P_{data}(x)$，其中每一个$x$对应的$P_{data}(x)$的概率是很大的，为了使$P(x;\theta)$越接近$P_{data}(x)$，原问题等价于最大化每一个$P(x_i;\theta)$，合起来就是最大化$\prod_{i=1}^mP_{G}(x^i;\theta)$。而实际上极大似然估计是等价于最小化$KL-divergence$，具体推导看下图，先取$log$（$log$是单调递增，不会改变原问题）将相乘化为相加，最后变成了$P_{data}$下$logP_{G}(x;\theta)$的期望，然后转化成积分的形式，后面加了一项$\intop_xP_{data}(x)logP_{data}(x)dx$，这一项是一个常数，没有变量$\theta$，加了也不会影响原问题的解，加了这一项之后原问题就等于最小化$P_{data}和P_{G}$的$KL-divergence$。
![17.jpg-43kB][19]
我们已经知道生成器要做的是$arg\space \underset{G}{min}\space Div(P_{data},P_{G})$，这里$P_{G}$是我们要去最优化的，虽然我们有真实样本，但$P_G$的分布我们还是不知道，而且如何去定量计算$P_{data}$和$P_G$的$divergence$，也就是$Div(P_{data},P_G)$，我们也是不知道的。所以接下来就需要引入鉴别器了。
虽然我们不知道$P_G$和$P_{data}$的分布，但我们可以随机采样它们分布的样本，如下图所示：
![19.jpg-36.7kB][20]
而我们知道鉴别器的目标是给真样本奖励，假样本惩罚，如下图所示，最后得到要鉴别器要优化的目标函数，鉴别器希望能够最大化这个目标函数，也就是$arg \space \underset{D}{max}\space V(D,G)$.注意，这里是是将$G$是$fixed$，是不变的。
![20.jpg-29.2kB][21]
我们再来解这个问题，解出最优$D^*$，接下来的步骤就比较数学了，给一个目标函数，求出极大值解。具体如图下
![21.jpg-39.1kB][22]
![22.jpg-42.2kB][23]
![23.jpg-43.1kB][24]
这个求解过程还是蛮详细的，最后我们竟然得到最大化$V(D,G)$竟然等于一个常数加上$P_G$和$P_{data}$的$JS-divergence$（$JS-divergence$与$KL-divergence$类似，不会改变解），这正是我们在生成器一直想求，可不会求得东西，鉴别器帮我们做到了。
于是，原始生成器的最优化问题$arg\space\underset{G}{min}Div(P_G,P_{data})$就可以转化成$arg\space\underset{G}{min}\space \underset{D}{max}V(G,D)$。那如何来求解$arg\space\underset{G}{min}\space \underset{D}{max}V(G,D)$这个最小最大问题呢？其实上面图上已经给出答案了，通过固定其中一个，求另一个，然后固定另一个，求之前固定住的这个。具体做法如图下：
![24.jpg-20kB][25]
更加详细的实践过程（也就是GAN的训练过程）如下所示，相信看了上面的一系列解释，会对GAN如此训练有了比较深的理解了吧。
![25.jpg-84.4kB][26]
GAN的理论就到此结束。

<h2>
<a id="D">
 4. 实现DCGAN
</a>
</h2>

这里使用数据集是Anime——台大李宏毅老师的GAN课程的数据集，点击[链接][27]下载，首先我们来看一下DCGAN的框架，如图所示
![26.jpg-40.5kB][28]
这个是生成器的结构图，鉴别器的结构与生成器大致相反，DCGAN与普通的GAN有一些区别，具体分为下面几点
> - DCGAN的网络都是全卷积的
> - 生成器除最后一层外都加batchnorm，鉴别器则是第一层没加bacthnorm
> - 鉴别器中的激活函数使用的是leaky_relu，负斜率是0.2
> - 生成器中的激活函数使用relu，输出层采用tanh
> - 采用Adam优化算法，学习率是0.0002，beta1=0.5

<h3>
<a id="D1">
4.1 model.py
</a>
</h3>

```python
import torch 
import torch.nn as nn 
import torch.functional as F

class Generate(nn.Module):
    def __init__(self, input_dim=100):
        super(Generate, self).__init__()
        channel = [512, 256, 128, 64, 3]
        kernel_size = 4
        stride = 2
        padding = 1
        self.convtrans1_block = self.__convtrans_bolck(input_dim, channel[0], 6, padding=0, stride=stride)
        self.convtrans2_block = self.__convtrans_bolck(channel[0], channel[1], kernel_size, padding, stride)
        self.convtrans3_block = self.__convtrans_bolck(channel[1], channel[2], kernel_size, padding, stride)
        self.convtrans4_block = self.__convtrans_bolck(channel[2], channel[3], kernel_size, padding, stride)
        self.convtrans5_block = self.__convtrans_bolck(channel[3], channel[4], kernel_size, padding, stride, layer="last_layer")
    
    def __convtrans_bolck(self, in_channel, out_channel, kernel_size, padding, stride, layer=None):
        if layer == "last_layer":
            convtrans = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
            tanh = nn.Tanh()
            return nn.Sequential(convtrans, tanh)
        else:
            convtrans = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
            batch_norm = nn.BatchNorm2d(out_channel)
            relu = nn.ReLU(True)
            return nn.Sequential(convtrans, batch_norm, relu)

    def forward(self, inp):
        x = self.convtrans1_block(inp)
        x = self.convtrans2_block(x)
        x = self.convtrans3_block(x)
        x = self.convtrans4_block(x)
        x = self.convtrans5_block(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        channels = [3, 64, 128, 256, 512]
        kernel_size = 4
        stride = 2
        padding = 1
        self.conv_bolck1 = self.__conv_block(channels[0], channels[1], kernel_size, stride, padding, "first_layer")
        self.conv_bolok2 = self.__conv_block(channels[1], channels[2], kernel_size, stride, padding)
        self.conv_bolok3 = self.__conv_block(channels[2], channels[3], kernel_size, stride, padding)
        self.conv_bolok4 = self.__conv_block(channels[3], channels[4], kernel_size, stride, padding)
        self.conv_bolok5 = self.__conv_block(channels[4], 1, kernel_size+1, stride, 0, "last_layer") 

    def __conv_block(self, inchannel, outchannel, kernel_size, stride, padding, layer=None):
        if layer == "first_layer":
            conv = nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding, bias=False)
            leakrelu = nn.LeakyReLU(0.2, inplace=True)
            return nn.Sequential(conv, leakrelu)
        elif layer == "last_layer":
            conv = nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding, bias=False)
            sigmoid = nn.Sigmoid()
            return nn.Sequential(conv, sigmoid)
        else:
            conv = nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding, bias=False)
            batchnorm = nn.BatchNorm2d(outchannel)
            leakrelu = nn.LeakyReLU(0.2, inplace=True)
            return nn.Sequential(conv, batchnorm, leakrelu)

    def forward(self,inp):
        x = self.conv_bolck1(inp)
        x = self.conv_bolok2(x)
        x = self.conv_bolok3(x)
        x = self.conv_bolok4(x)
        x = self.conv_bolok5(x)
        return x 


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0,0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0,0.01)
        m.bias.data.fill_(0)



if __name__ == "__main__":
    model1 = Generate()
    x = torch.randn(10,100,1,1)
    y = model1.forward(x)
    print(y.size())
    model2 = Discriminator()
    a = torch.randn(10,3,96,96)
    b = model2.forward(a)
    print(b.size())


```

<h3>
<a id="D2">
4.2 AnimeDataset.py
</a>
</h3>

``` python
import torch,torch.utils.data
import numpy as np 
import scipy.misc, os

class AnimeDataset(torch.utils.data.Dataset):
    def __init__(self, directory, dataset, size_per_dataset):
        self.directory = directory
        self.dataset = dataset
        self.size_per_dataset = size_per_dataset
        self.data_files = []
        data_path = os.path.join(directory, dataset)
        for i in range(size_per_dataset):
            self.data_files.append(os.path.join(data_path,"{}.jpg".format(i)))
        
    def __getitem__(self, ind):
        path = self.data_files[ind]
        img = scipy.misc.imread(path)
        img = img.transpose(2,0,1)-127.5/127.5
        return img

    def __len__(self):
        return len(self.data_files)

if __name__ == "__main__":
    dataset = AnimeDataset(os.getcwd(),"anime",100)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True,num_workers=4)
    for i, inp in enumerate(loader):
        print(i,inp.size())
```
<h3>
<a id="D3">
4.3 utils.py
</a>
</h3>

```python
import os, imageio,scipy.misc
import matplotlib.pyplot as plt


def creat_gif(gif_name, img_path, duration=0.3):
    frames = []
    img_names = os.listdir(img_path)
    img_list = [os.path.join(img_path, img_name) for img_name in img_names]
    for img_name in img_list:
        frames.append(imageio.imread(img_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)

def visualize_loss(generate_txt_path, discriminator_txt_path):
    
    with open(generate_txt_path, 'r') as f:
        G_list_str = f.readlines()

    with open(discriminator_txt_path, 'r') as f:
        D_list_str = f.readlines()
    
    D_list_float, G_list_float = [], []

    for D_item, G_item in zip(D_list_str, G_list_str):
        D_list_float.append(float(D_item.strip().split(':')[-1]))
        G_list_float.append(float(G_item.strip().split(':')[-1]))
    
    list_epoch = list(range(len(D_list_float)))

    full_path = os.path.join(os.getcwd(), "saved/logging.png")
    plt.figure()
    plt.plot(list_epoch, G_list_float, label="generate", color='g')
    plt.plot(list_epoch, D_list_float, label="discriminator", color='b')
    plt.legend()
    plt.title("DCGAN_Anime")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(full_path)

```
<h3>
<a id="D4">
4.4 main.py
</a>
</h3>

```python
import torch 
import torch.nn as nn 
from torch.optim import Adam
from torchvision.utils import make_grid
from model import Generate,Discriminator,weight_init
from AnimeDataset import AnimeDataset 
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import os, argparse
from tqdm import tqdm
from utils import creat_gif, visualize_loss

def main():

    parse = argparse.ArgumentParser()

    parse.add_argument("--lr", type=float, default=0.0001, 
                        help="learning rate of generate and discriminator")
    parse.add_argument("--beta1", type=float, default=0.5,
                        help="adam optimizer parameter")
    parse.add_argument("--batch_size", type=int, default=64,
                        help="number of dataset in every train or test iteration")
    parse.add_argument("--dataset", type=str, default="anime",
                        help="base path for dataset")
    parse.add_argument("--epochs", type=int, default=500,
                        help="number of training epochs")
    parse.add_argument("--loaders", type=int, default=4,
                        help="number of parallel data loading processing")
    parse.add_argument("--size_per_dataset", type=int, default=30000,
                        help="number of training data")
    parse.add_argument("--pre_train", type=bool, default=False,
                        help="whether load pre_train model")

    args = parse.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if not os.path.exists("saved"):
        os.mkdir("saved")
    if not os.path.exists("saved/img"):
        os.mkdir("saved/img")

    if os.path.exists("faces"):
        pass
    else:
        print("Don't find the dataset directory, please copy the link in website ,download and extract faces.tar.gz .\n \
        https://drive.google.com/drive/folders/1mCsY5LEsgCnc0Txv0rpAUhKVPWVkbw5I \n ")
        exit()
    if args.pre_train:
        generate = torch.load("saved/generate.t7").to(device)
        discriminator = torch.load("saved/discriminator.t7").to(device)
    else:
        generate = Generate().to(device)
        discriminator = Discriminator().to(device)

    generate.apply(weight_init)
    discriminator.apply(weight_init)

    dataset = AnimeDataset(os.getcwd(), args.dataset, args.size_per_dataset)
    dataload = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    criterion = nn.BCELoss().to(device)

    optimizer_G = Adam(generate.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    fixed_noise = torch.randn(64, 100, 1, 1).to(device)

    for epoch in range(args.epochs):

        print("Main epoch{}:".format(epoch))
        progress = tqdm(total=len(dataload.dataset))
        loss_d, loss_g = 0, 0
        
        for i, inp in enumerate(dataload):
            # train discriminator   
            real_data = inp.float().to(device)
            real_label = torch.ones(inp.size()[0]).to(device)
            noise = torch.randn(inp.size()[0], 100, 1, 1).to(device)
            fake_data = generate(noise)
            fake_label = torch.zeros(fake_data.size()[0]).to(device)
            optimizer_D.zero_grad()
            real_output = discriminator(real_data)
            real_loss = criterion(real_output.squeeze(), real_label)
            real_loss.backward()
            fake_output = discriminator(fake_data)
            fake_loss = criterion(fake_output.squeeze(), fake_label)
            fake_loss.backward()
            loss_D = real_loss + fake_loss
            optimizer_D.step()

            #train generate
            optimizer_G.zero_grad()
            fake_data = generate(noise)
            fake_label = torch.ones(fake_data.size()[0]).to(device)
            fake_output = discriminator(fake_data)
            loss_G = criterion(fake_output.squeeze(), fake_label)
            loss_G.backward()
            optimizer_G.step()

            progress.update(dataload.batch_size)
            progress.set_description("D:{}, G:{}".format(loss_D.item(), loss_G.item()))

            loss_g += loss_G.item()
            loss_d += loss_D.item()
        
        loss_g /= (i+1)
        loss_d /= (i+1)

        with open("generate_loss.txt", 'a+') as f:
            f.write("loss_G:{} \n".format(loss_G.item()))

        with open("discriminator_loss.txt", 'a+') as f:
            f.write("loss_D:{} \n".format(loss_D.item()))

        if epoch % 20 == 0:

            torch.save(generate, os.path.join(os.getcwd(), "saved/generate.t7"))
            torch.save(discriminator, os.path.join(os.getcwd(), "saved/discriminator.t7"))

            img = generate(fixed_noise).to("cpu").detach().numpy()

            display_grid = np.zeros((8*96,8*96,3))
            
            for j in range(int(64/8)):
                for k in range(int(64/8)):
                    display_grid[j*96:(j+1)*96,k*96:(k+1)*96,:] = (img[k+8*j].transpose(1, 2, 0)+1)/2

            img_save_path = os.path.join(os.getcwd(),"saved/img/{}.png".format(epoch))
            scipy.misc.imsave(img_save_path, display_grid)

    creat_gif("evolution.gif", os.path.join(os.getcwd(),"saved/img"))

    visualize_loss("generate_loss.txt", "discriminator_loss.txt")

                


if __name__ == "__main__":
    main()
    
```

代码运行请参考github的[readme][29]，最后500个epoch的结果图如下
![500.png-1159kB][30]

<h2>
<a id="E">
 5. 实现WGAN
</a>
</h2>
WGAN pytorch版本一直都有bug，目前还没找到原因，实现了一个keras版本的，代码如下(运行前记得看readme)：
```python
import os,scipy.misc
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)

os.environ['KERAS_BACKEND']='tensorflow' 
os.environ['TENSORFLOW_FLAGS']='floatX=float32,device=cuda'

def DCGAN_D(isize, nc, ndf):
    inputs = Input(shape=(isize, isize, nc))
    x = ZeroPadding2D()(inputs)
    x = Conv2D(ndf, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    for _ in range(4):
        x = ZeroPadding2D()(x)
        x = Conv2D(ndf*2, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(x)
        x = BatchNormalization(epsilon=1.01e-5, gamma_init=gamma_init)(x, training=1)
        x = LeakyReLU(alpha=0.2)(x)
        ndf *= 2    
    x = Conv2D(1, kernel_size=3, strides=1, use_bias=False, kernel_initializer=conv_init)(x)
    outputs = Flatten()(x)
    return Model(inputs=inputs, outputs=outputs)

def DCGAN_G(isize, nz, ngf):
    inputs = Input(shape=(nz,))
    x = Reshape((1, 1, nz))(inputs)
    x = Conv2DTranspose(filters=ngf, kernel_size=3, strides=2, use_bias=False,
                           kernel_initializer = conv_init)(x)
    for _ in range(4):
        x = Conv2DTranspose(filters=int(ngf/2), kernel_size=4, strides=2, use_bias=False,
                        kernel_initializer = conv_init)(x)
        x = Cropping2D(cropping=1)(x)
        x = BatchNormalization(epsilon=1.01e-5, gamma_init=gamma_init)(x, training=1) 
        x = Activation("relu")(x)
        ngf = int(ngf/2)
    x = Conv2DTranspose(filters=3, kernel_size=4, strides=2, use_bias=False,
                        kernel_initializer = conv_init)(x)
    x = Cropping2D(cropping=1)(x)
    outputs = Activation("tanh")(x)

    return Model(inputs=inputs, outputs=outputs)

nc = 3
nz = 100
ngf = 1024
ndf = 64
imageSize = 96
batchSize = 64
lrD = 0.00005 
lrG = 0.00005
clamp_lower, clamp_upper = -0.01, 0.01   

netD = DCGAN_D(imageSize, nc, ndf)
netD.summary()

netG = DCGAN_G(imageSize, nz, ngf)
netG.summary()

clamp_updates = [K.update(v, K.clip(v, clamp_lower, clamp_upper))
                          for v in netD.trainable_weights]
netD_clamp = K.function([],[], clamp_updates)

netD_real_input = Input(shape=(imageSize, imageSize, nc))
noisev = Input(shape=(nz,))

loss_real = K.mean(netD(netD_real_input))
loss_fake = K.mean(netD(netG(noisev)))
loss = loss_fake - loss_real 
training_updates = RMSprop(lr=lrD).get_updates(netD.trainable_weights,[], loss)
netD_train = K.function([netD_real_input, noisev],
                        [loss_real, loss_fake],    
                        training_updates)

loss = -loss_fake 
training_updates = RMSprop(lr=lrG).get_updates(netG.trainable_weights,[], loss)
netG_train = K.function([noisev], [loss], training_updates)

fixed_noise = np.random.normal(size=(batchSize, nz)).astype('float32')

datagen = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    rotation_range=20,
    rescale=1./255
)

train_generate = datagen.flow_from_directory("faces/", target_size=(96,96), batch_size=64, 
                                                shuffle=True, class_mode=None, save_format='jpg')

step = 0
print(dir(train_generate))
for step in range(100000):   
    
    for _ in range(5):
        real_data = (np.array(train_generate.next())*2-1)
        noise = np.random.normal(size=(batchSize, nz))
        errD_real, errD_fake  = netD_train([real_data, noise])
        errD = errD_real - errD_fake
        netD_clamp([])
    
    noise = np.random.normal(size=(batchSize, nz))  
    errG, = netG_train([noise])    
    print('[%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f' % (step, errD, errG, errD_real, errD_fake))
            
    if step%1000==0:
        netD.save("discriminator.h5")
        netG.save("generate.h5")
        fake = netG.predict(fixed_noise)
        display_grid = np.zeros((8*96,8*96,3))
        
        for j in range(int(64/8)):
            for k in range(int(64/8)):
                display_grid[j*96:(j+1)*96,k*96:(k+1)*96,:] = fake[k+8*j]
        img_save_path = os.path.join(os.getcwd(),"saved/img/{}.png".format(step))
        scipy.misc.imsave(img_save_path, display_grid)
```
代码运行请参考github的[readme][31]，100000step的结果：
![wgan_keras_result.png-1175.5kB][32]

<h2>
<a id="F">
 6. 实现ConditionalGAN
</a>
</h2>
详细运行请看github中的[readme][33]。

<h3>
<a id="F1">
 6.1 CGAN.py
</a>
</h3>
```python
import torch,os,scipy.misc,random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from utils import load_Anime,test_Anime



class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(self.shape)

class Generate(nn.Module):
    def __init__(self, z_dim, y_dim, image_height, image_width):
        super(Generate, self).__init__()
        self.conv_trans = nn.Sequential(
                nn.Linear(z_dim+y_dim, (image_height//16)*(image_width//16)*384),
                nn.BatchNorm1d((image_height//16)*(image_width//16)*384, 
                                eps=1e-5, momentum=0.9, affine=True),
                Reshape(-1, 384, image_height//16, image_width//16),
                nn.ConvTranspose2d(384, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256, eps=1e-5, momentum=0.9, affine=True),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128, eps=1e-5, momentum=0.9, affine=True),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64, eps=1e-5, momentum=0.9, affine=True),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh()
        )    
            
    def forward(self, z, y):
        z = torch.cat((z,y), dim=-1)
        z = self.conv_trans(z)
        return z

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential( 
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 384, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(407, 384, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.linear = nn.Linear(4*4*384, 1)
            
    def forward(self, x, y):
        x = self.conv(x)
        y = torch.unsqueeze(y, 2)
        y = torch.unsqueeze(y, 3)
        y = y.expand(y.size()[0], y.size()[1], x.size()[2], x.size()[3])
        x = torch.cat((x,y), dim=1)
        x = self.conv1(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        x = x.squeeze()
        x = F.sigmoid(x)       
        return x

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0,0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0,0.01)
        m.bias.data.fill_(0)

class CGAN(object):

    def __init__(self, dataset_path, save_path, epochs, batchsize, z_dim, device, mode):
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.epochs = epochs
        self.batch_size = batchsize
        self.mode = mode
        self.image_height = 64
        self.image_width = 64
        self.learning_rate = 0.0001
        self.z_dim = z_dim
        self.y_dim = 23
        self.iters_d = 2
        self.iters_g = 1
        self.device = device
        self.criterion = nn.BCELoss().to(device)
        if mode == "train":
            self.X, self.Y = load_Anime(self.dataset_path)
            self.batch_nums = len(self.X)//self.batch_size
        
    def train(self):
        generate = Generate(self.z_dim, self.y_dim, self.image_height, self.image_width).to(self.device)
        discriminator = Discriminator().to(self.device)
        generate.apply(weight_init)
        discriminator.apply(weight_init)
        optimizer_G = Adam(generate.parameters(), lr=self.learning_rate)
        optimizer_D = Adam(discriminator.parameters(), lr=self.learning_rate)
        step = 0
        for epoch in range(self.epochs):
            print("Main epoch:{}".format(epoch))
            for i in range(self.batch_nums):
                step += 1
                batch_images = torch.from_numpy(np.asarray(self.X[i*self.batch_size:(i+1)*self.batch_size]).astype(np.float32)).to(self.device)
                batch_labels = torch.from_numpy(np.asarray(self.Y[i*self.batch_size:(i+1)*self.batch_size]).astype(np.float32)).to(self.device)
                batch_images_wrong = torch.from_numpy(np.asarray(self.X[random.sample(range(len(self.X)), len(batch_images))]).astype(np.float32)).to(self.device)
                batch_labels_wrong = torch.from_numpy(np.asarray(self.Y[random.sample(range(len(self.Y)), len(batch_images))]).astype(np.float32)).to(self.device)
                batch_z = torch.from_numpy(np.random.normal(0, np.exp(-1 / np.pi), [self.batch_size, self.z_dim]).astype(np.float32)).to(self.device)
                # discriminator twice, generate once
                for _ in range(self.iters_d):
                    optimizer_D.zero_grad()
                    d_loss_real = self.criterion(discriminator(batch_images, batch_labels), torch.ones(self.batch_size).to(self.device))
                    d_loss_fake = (self.criterion(discriminator(batch_images, batch_labels_wrong), torch.zeros(self.batch_size).to(self.device)) \
                                  + self.criterion(discriminator(batch_images_wrong, batch_labels), torch.zeros(self.batch_size).to(self.device)) \
                                  + self.criterion(discriminator(generate(batch_z, batch_labels), batch_labels), torch.zeros(self.batch_size).to(self.device)))/3
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    optimizer_D.step()
                
                for _ in range(self.iters_g):
                    optimizer_G.zero_grad()
                    g_loss = self.criterion(discriminator(generate(batch_z, batch_labels), batch_labels), torch.ones(self.batch_size).to(self.device))
                    g_loss.backward()
                    optimizer_G.step()
                
                print("epoch:{}, step:{}, d_loss:{}, g_loss:{}".format(epoch, step, d_loss.item(), g_loss.item()))
                #show result and save model 
                if (step)%5000 == 0:
                    z, y = test_Anime()
                    image = generate(torch.from_numpy(z).float().to(self.device),torch.from_numpy(y).float().to(self.device)).to("cpu").detach().numpy()
                    display_grid = np.zeros((5*64,5*64,3))
                    for j in range(5):
                        for k in range(5):
                            display_grid[j*64:(j+1)*64,k*64:(k+1)*64,:] = image[k+5*j].transpose(1, 2, 0)
                    img_save_path = os.path.join(self.save_path,"training_img/{}.png".format(step))
                    scipy.misc.imsave(img_save_path, display_grid)
                    torch.save(generate, os.path.join(self.save_path, "generate.t7"))
                    torch.save(discriminator, os.path.join(self.save_path, "discriminator.t7"))

    def infer(self):
        z, y = test_Anime()
        generate = torch.load(os.path.join(self.save_path, "generate.t7")).to(self.device)
        image = generate(torch.from_numpy(z).float().to(self.device),torch.from_numpy(y).float().to(self.device)).to("cpu").detach().numpy()
        display_grid = np.zeros((5*64,5*64,3))
        for j in range(5):
            for k in range(5):
                display_grid[j*64:(j+1)*64,k*64:(k+1)*64,:] = image[k+5*j].transpose(1, 2, 0)
        img_save_path = os.path.join(self.save_path,"testing_img/test.png")
        scipy.misc.imsave(img_save_path, display_grid)
        print("infer ended, look the result in the save/testing_img/")
```

<h3>
<a id="F2">
 6.2 utils.py
</a>
</h3>
```python
# most code from https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw3/hw3_2/
import numpy as np
import cv2
import os

def test_Anime():
    np.random.seed(999)
    z = np.random.normal(0, np.exp(-1 / np.pi), [25, 62])
    tag_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 
            'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair', 
            'gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes',
            'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
    tag_txt = open("test.txt", 'r').readlines()
    labels = []
    for line in tag_txt:
        label = np.zeros(len(tag_dict))

        for i in range(len(tag_dict)):
            if tag_dict[i] in line:
                label[i] = 1
        labels.append(label)

    for i in range(len(tag_txt)):
        for j in range(4):
            labels.insert(5*i+j, labels[5*i])
    
    return z, np.array(labels)


def load_Anime(dataset_filepath):
    tag_csv_filename = dataset_filepath.replace('images/', 'tags.csv')
    tag_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 
            'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair', 
            'gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes',
            'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

    tag_csv = open(tag_csv_filename, 'r').readlines()

    id_label = []
    for line in tag_csv:
        id, tags = line.split(',')
        label = np.zeros(len(tag_dict))
        
        for i in range(len(tag_dict)):
            if tag_dict[i] in tags:
                label[i] = 1
        
        # Keep images with hair or eyes.
        if np.sum(label) == 2 or np.sum(label) == 1:
            id_label.append((id, label))


    # Load file name of images.
    image_file_list = []
    for image_id, _ in id_label:
        image_file_list.append(image_id + '.jpg')

    # Resize image to 64x64.
    image_height = 64
    image_width = 64
    image_channel = 3

    # Allocate memory space of images and labels.
    images = np.zeros((len(image_file_list), image_channel, image_width, image_height))
    labels = np.zeros((len(image_file_list), len(tag_dict)))
    print ('images.shape: ', images.shape)
    print ('labels.shape: ', labels.shape)

    print ('Loading images to numpy array...')
    data_dir = dataset_filepath
    for index, filename in enumerate(image_file_list):
        images[index] = cv2.cvtColor(
            cv2.resize(
                cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_COLOR), 
                (image_width, image_height)), 
                cv2.COLOR_BGR2RGB).transpose(2,0,1)
        labels[index] = id_label[index][34]
    
    print ('Random shuffling images and labels...')
    np.random.seed(9487)
    indice = np.array(range(len(image_file_list)))
    np.random.shuffle(indice)
    images = images[indice]
    labels = labels[indice]

    print ('[Tip 1] Normalize the images between -1 and 1.')
    # Tip 1. Normalize the inputs
    #   Normalize the images between -1 and 1.
    #   Tanh as the last layer of the generator output.
    return (images / 127.5) - 1, labels

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
```

<h3>
<a id="F3">
 6.3 main.py
</a>
</h3>
```python
import argparse
from CGAN import CGAN
from utils import check_folder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=62, help='Dimension of noise vector')
    parser.add_argument('--dataset_path', type=str, default='./images/',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--save_path', type=str, default='./save/',
                        help='Directory name to save the generated images')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or infer')
    parser.add_argument('--device', type=str, default='cuda',
                        help='train on GPU or CPU')
    parser.add_argument('--save_training_img_path', type=str, default='./save/training_img/',
                        help='Directory name to save the training images')
    parser.add_argument('--save_testing_img_path', type=str, default='./save/testing_img/',
                        help='Directory name to save the training images')
    return parser.parse_args()


def main():
    args = parse_args()
    check_folder(args.dataset_path)
    check_folder(args.save_path)
    gan = CGAN(args.dataset_path,
               args.save_path,
               args.epochs,
               args.batch_size,
               args.z_dim,
               args.device,
               args.mode)
    if args.mode == "train":
        check_folder(args.save_training_img_path)
        gan.train()
    else:
        check_folder(args.save_testing_img_path)
        gan.infer()    

if __name__ == "__main__":
    main()
```

55000step的结果：
![image_1cmmain3r1tuv1knphss1rc0pp09.png-316kB][35]

<h3>
<a id="F4">
 6.4 GUI.py
</a>
</h3>
```python
import torch
import tkinter as tk
import os 
import numpy as np
from tkinter import ttk
import scipy.misc
from PIL import Image,ImageTk


win = tk.Tk()
win.title('Conditional-GAN-GUI')
win.geometry('200x200')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

generate = torch.load("save/generate.t7").to(device)
generate.eval()

def create():
    z = np.random.normal(0, np.exp(-1 / np.pi), [1, 62])
    line = comboxlist1.get() + ' ' + comboxlist2.get()
    tag_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 
            'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair', 
            'gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes',
            'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
    y = np.zeros((1, len(tag_dict)))

    for i in range(len(tag_dict)):
        if tag_dict[i] in line:
            y[0][i] = 1
    
    image = generate(torch.from_numpy(z).float().to(device), torch.from_numpy(y).float().to(device)).to("cpu").detach().numpy()
    image = np.squeeze(image)
    image = image.transpose(1, 2, 0)
    scipy.misc.imsave('anime.png', image)
    img_open = Image.open('anime.png')
    img = ImageTk.PhotoImage(img_open)
    label.configure(image=img)
    label.image=img
    

# def go(*args):   #处理事件，*args表示可变参数
#     print(comboxlist.get()) #打印选中的值
 
comvalue1=tk.StringVar()#窗体自带的文本，新建一个值
comboxlist1=ttk.Combobox(win,textvariable=comvalue1) #初始化
comboxlist1["values"]=('orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 
                      'purple hair', 'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair')
comboxlist1.current(0)  #选择第一个
# comboxlist.bind("<<ComboboxSelected>>",go)  #绑定事件,(下拉列表框被选中时，绑定go()函数)
comboxlist1.pack()
 
comvalue2=tk.StringVar()
comboxlist2=ttk.Combobox(win,textvariable=comvalue2)
comboxlist2["values"]=('gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes',
                      'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes')
comboxlist2.current(0)
# comboxlist.bind("<<ComboboxSelected>>",go)
comboxlist2.pack()

bm = tk.PhotoImage(file ='anime.png')
label = tk.Label(win, image = bm)
label.pack()

b = tk.Button(win,
    text='create',      # 显示在按钮上的文字
    width=15, height=2, 
    command=create)     # 点击按钮式执行的命令
b.pack()   # 按钮位置

win.mainloop()
```
![GUI.jpg-16.1kB][36]

<h2>
<a id="G">
 7. GAN小技巧
</a>
</h2>

> 1.对真实图片进行归一化，与生成图片分布一样，也就是[-1,1].
> 2.随机噪声使用高斯分布，不要使用均匀分布，也就是在代码中使用torch.randn，而不是torch.rand
> 3.初始化权重很有必要，详细见model.py中的weight_init函数
> 4.在训练时，在鉴别器中产生的noise，生成器也要用这个noise进行参数，这点很重要。我最开始的时候就是鉴别器随机产生noise，生成器也随机产生noise，训练得很不好。
> 5.在训练过程中，很有可能鉴别器的loss等于0（鉴别器太强了，起初我试过减小鉴别器的学习率，但还是会有这个情况，我猜想原因是在某一个batch中，鉴别器恰好将随机噪声产生的图片和真实图片完全区分开，loss为0），导致生成器崩溃（梯度弥散），所以最好按多少个epoch保存模型，然后在导入模型再训练。个人觉得数据增强和增大batchsize会减弱这种情况的可能性，这个还未实践。
 
<h2>
<a id="H">
 8. 参考
</a>
</h2>

1 [李宏毅GAN课程及PPT][37]
2 [DCGAN paper][38]
3 [chenyuntc][39]


  [1]: https://github.com/FangYang970206/Anime_GAN
  [2]: http://static.zybuluo.com/fangyang970206/k43ryxpx8eg7irnkvd2el9nj/image_1cjq51f9kp4b136b77218i61pg5m.png
  [3]: http://static.zybuluo.com/fangyang970206/aap25fmxt1s0ppk86acamz3b/1.jpg
  [4]: http://static.zybuluo.com/fangyang970206/q5pq6r7qgh2y1s0j1trn2o5u/2.jpg
  [5]: http://static.zybuluo.com/fangyang970206/q9l80e7bjt6zbe51u6egs17t/4.jpg
  [6]: http://static.zybuluo.com/fangyang970206/28v7xiijynq4yr8di8zy3nka/5.jpg
  [7]: http://static.zybuluo.com/fangyang970206/3fyrq8t2eerlq359pjmb3efn/6.jpg
  [8]: http://static.zybuluo.com/fangyang970206/j61t3qunmcr533lsduehtyzi/7.jpg
  [9]: http://static.zybuluo.com/fangyang970206/agschrpbycy82mvvdvfd60ct/8.jpg
  [10]: http://static.zybuluo.com/fangyang970206/3lnner6u79s5uh96l2miuize/image_1cjqr5jm9uor1b90hcujc51g2c7m.png
  [11]: http://static.zybuluo.com/fangyang970206/tba0ho1sydvkbwjc1g47pr1z/9.jpg
  [12]: http://static.zybuluo.com/fangyang970206/p2cz5z2bwo48uoqlwcbpck5s/10.jpg
  [13]: http://static.zybuluo.com/fangyang970206/dlsgteu5824oinbzbfq3p3i9/11.jpg
  [14]: http://static.zybuluo.com/fangyang970206/308e7s4mava5poud7ym3l2jk/12.jpg
  [15]: http://static.zybuluo.com/fangyang970206/esdw13mima8zojpioiqv3wj3/13.jpg
  [16]: http://static.zybuluo.com/fangyang970206/mism2zxp8emrw790053075md/14.jpg
  [17]: http://static.zybuluo.com/fangyang970206/0kma2enjd1nk08vscmixolpq/18.jpg
  [18]: http://static.zybuluo.com/fangyang970206/t9m7sg8mpbqxlcns9pbvdri1/16.jpg
  [19]: http://static.zybuluo.com/fangyang970206/h44bvwaz11eo47ltee7p9sv6/17.jpg
  [20]: http://static.zybuluo.com/fangyang970206/kmgit862n70riboqk30apriv/19.jpg
  [21]: http://static.zybuluo.com/fangyang970206/0bvbxs27ddvub0npwq1jbmzm/20.jpg
  [22]: http://static.zybuluo.com/fangyang970206/yoosk1xmqrk7hvrjxxwn7zza/21.jpg
  [23]: http://static.zybuluo.com/fangyang970206/unx7mump8ngb2gmpv0vx7kfd/22.jpg
  [24]: http://static.zybuluo.com/fangyang970206/tyunuzqfrq743s42ovx9s0tb/23.jpg
  [25]: http://static.zybuluo.com/fangyang970206/3y41a91u50ol287q0wdgku6q/24.jpg
  [26]: http://static.zybuluo.com/fangyang970206/mj4i029dj7x8miuv83pl8j6l/25.jpg
  [27]: https://1drv.ms/u/s!AgBYzHhocQD4g0_Fr-mC-DYfWahJ
  [28]: http://static.zybuluo.com/fangyang970206/5bo1mjjfpi6vboawd5y552hw/26.jpg
  [29]: https://github.com/FangYang970206/Anime_GAN
  [30]: http://static.zybuluo.com/fangyang970206/jte65paxsabwrmj2z2taaxcd/500.png
  [31]: https://github.com/FangYang970206/Anime_GAN
  [32]: http://static.zybuluo.com/fangyang970206/0u770dscinkc7s37tgmiyquw/wgan_keras_result.png
  [33]: https://github.com/FangYang970206/Anime_GAN
  [34]: https://github.com/FangYang970206/Anime_GAN
  [35]: http://static.zybuluo.com/fangyang970206/splf7k5p8i8krcgaudsjfynf/image_1cmmain3r1tuv1knphss1rc0pp09.png
  [36]: http://static.zybuluo.com/fangyang970206/y3wq2zs05g2wjo21z7dy9n77/GUI.jpg
  [37]: http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html
  [38]: http://arxiv.org/abs/1511.06434
  [39]: https://github.com/chenyuntc/pytorch-GAN