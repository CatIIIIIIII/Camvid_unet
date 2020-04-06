# Camvid_unet
implementation of u-net on camvid dataset using fastaiv1

@[TOC]([fasi.ai] unet实现CamVid数据集预测)
# fastai介绍
最近发现了一个究极棒的python包fastai，他们的官网说宗旨是`make neural net uncool again`，我觉得也确实做到了这点。fastai相较于pytorch，就像keras相较于tensorflow，是一个高级封装。其封装程度之高，用5行就可以完成mnist数据集的训练：

```python
from fastai.vision import *
path = untar_data(MNIST_PATH)
data = image_data_from_folder(path)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit(1)
```
当然这么高的封装度肯定是以牺牲灵活性为代价的，但是其效果有很好。所以打算以后用fastai快速搭建模型进行初步筛选，然后再用pytorch深入研究。我用的是fastaiv1版本，因为其 比较稳定而且demo多，安装具体请参照官方github和文档：

> [https://github.com/fastai/fastai/blob/master/README.md#installation](https://github.com/fastai/fastai/blob/master/README.md#installation)
> [https://docs.fast.ai/training.html](https://docs.fast.ai/training.html)

# unet介绍
在图像分割任务特别是医学图像分割中，U-Net无疑是最成功的方法之一，该方法在2015年MICCAI会议上提出，目前已达到四千多次引用。其采用的编码器（下采样）-解码器（上采样）结构和跳跃连接是一种非常经典的设计方法。[^1]
![unet结构](https://img-blog.csdnimg.cn/20200406000555118.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70)
# CamVid数据集介绍
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200406000728410.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70)
Cambridge-driving Labeled Video Database (CamVid)是第一个具有目标类别语义标签的视频集合。数据库提供32个ground truth语义标签，将每个像素与语义类别之一相关联。该数据库解决了对实验数据的需求，以定量评估新兴算法。数据是从驾驶汽车的角度拍摄的，驾驶场景增加了观察目标的数量和异质性。[1]

Cambridge-driving标签视频数据库(CamVid)是第一个包含对象类语义标签的视频集合，其中包含元数据。该数据库提供了ground truth标签，将每个像素与32个语义类中的一个关联起来。该数据库解决了对实验数据的需求，以定量评估新兴算法。虽然大多数视频都是用固定位置的cctv式摄像机拍摄的，但我们的数据是从驾驶汽车的角度拍摄的。驱动场景增加了观察对象的数量和异构性。

提供超过10分钟的高质量30Hz连续镜头，对应的语义标记图像为1Hz，部分为15Hz。CamVid数据库提供了与对象分析研究人员相关的四个贡献。首先手动指定700多幅图像的逐像素语义分割，然后由第二个人检查并确认其准确性。其次，数据库中高质量、高分辨率的彩色视频图像对那些对驾驶场景或自我运动感兴趣的人来说，是有价值的长时间数字化视频。第三，我们拍摄了相机颜色响应和内部物理的标定序列，并计算了序列中每一帧的三维相机姿态。最后，为了支持扩展这个或其他数据库，我们提供了定制的标签软件，帮助用户为其他图像和视频绘制精确的类标签。我们通过测量算法在三个不同领域的性能来评估数据库的相关性:多类目标识别、行人检测和标签传播。[^2]

# 程序实现
fastai设计的时候就是基于交互式的想法，所以其与jupyter的切合度十分之高，以下编程均在`jupyter notebook`上完成。首先我们导入相应的包：
```python
from fastai.vision import *
```
事实上有人指出fastai在这点上做的不是很好，因为全导入的方式不利于理解类的继承结构；但是姑且先这么用，以后再细究。

数据集我们可以自己下，但是事实上fastai内部已经集成了相当多的数据集，包括`CoCo、CIFAR10`等等。当然第一次使用的时候还是需要下载的，我在服务器上下载`CAMVID`大概花了2h。下载完成后可以看一下文件包含的内容，fastai的路径都是python内置的`pathlib`中的`Path`类型的子类。相较于`Path`类添加了许多其他的功能比如列出文件和文件夹`.ls()`等等，可以很方便的进行路径合并和查看：

```python
path = untar_data(URLs.CAMVID)
path.ls()
```
**OUT:**
```bash
[PosixPath('/home/bir2160400081/.fastai/data/camvid/images'),
 PosixPath('/home/bir2160400081/.fastai/data/camvid/codes.txt'),
 PosixPath('/home/bir2160400081/.fastai/data/camvid/valid.txt'),
 PosixPath('/home/bir2160400081/.fastai/data/camvid/labels')]
```
`images`文件夹里包含的是训练数据的x；`labels`文件夹里的相当于y。我们可以看下有什么内容：

```python
path_img = path / "images"
path_label = path / "labels"
```

```python
path_img.ls()[:5]
```
**OUT:**

```bash
[PosixPath('/home/bir2160400081/.fastai/data/camvid/images/Seq05VD_f04920.png'),
 PosixPath('/home/bir2160400081/.fastai/data/camvid/images/0016E5_08073.png'),
 PosixPath('/home/bir2160400081/.fastai/data/camvid/images/0016E5_02250.png'),
 PosixPath('/home/bir2160400081/.fastai/data/camvid/images/Seq05VD_f01470.png'),
 PosixPath('/home/bir2160400081/.fastai/data/camvid/images/0016E5_08081.png')]
```

```python
path_label.ls()[:5]
```
**OUT:**

```python
[PosixPath('/home/bir2160400081/.fastai/data/camvid/labels/0006R0_f03630_P.png'),
 PosixPath('/home/bir2160400081/.fastai/data/camvid/labels/0001TP_008280_P.png'),
 PosixPath('/home/bir2160400081/.fastai/data/camvid/labels/0001TP_007770_P.png'),
 PosixPath('/home/bir2160400081/.fastai/data/camvid/labels/0016E5_06930_P.png'),
 PosixPath('/home/bir2160400081/.fastai/data/camvid/labels/0016E5_08520_P.png')]
```
可以看出里面都是文件名称相对应的图片，只不过`label`中的文件名多了后缀的`_P`。。

为了找到训练数据和label的对应关系，我们先定义一个函数来找到映射：

```python
get_y_fn = lambda x: path_label/f'{x.stem}_P{x.suffix}'
```
然后查看一下某一个训练数据是啥样的，这里直接用fastai提供的内置`open_image`函数就可以了：

```python
img = open_image(path_img.ls()[0])
img.show(figsize=(5, 5))
```
**OUT：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200406103510456.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70 =300x200)
由于label的含义是每个像素对应一个代表种类的数字，所以此处需要使用`open_mask`方法：
```python
mask = open_mask(get_y_fn(path_img.ls()[0]))
mask.show(figsize=(5, 5))
```
**OUT：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020040610410819.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70 =300x200)
路径中的`codes.txt`表示的是`label`图片中不同数字的意思：

```python
codes = np.loadtxt(path/'codes.txt', dtype=str)
codes
```
**OUT**

```python
array(['Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car', 'CartLuggagePram', 'Child', 'Column_Pole',
       'Fence', 'LaneMkgsDriv', 'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter', 'OtherMoving', 'ParkingBlock',
       'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol', 'Sky', 'SUVPickupTruck', 'TrafficCone',
       'TrafficLight', 'Train', 'Tree', 'Truck_Bus', 'Tunnel', 'VegetationMisc', 'Void', 'Wall'], dtype='<U17')
```
`valid.txt`则是确定哪些图片用来做验证集。因为CamVid都是取自视频中的某些帧，如果随机选取验证机会导致训练集和验证集之间有过大的相关性。

```python
valid_frame = np.loadtxt(path/'valid.txt', dtype=str)
valid_frame
```
**OUT**

```python
array(['0016E5_07959.png', '0016E5_07961.png', '0016E5_07963.png', '0016E5_07965.png', ..., '0016E5_08153.png',
       '0016E5_08155.png', '0016E5_08157.png', '0016E5_08159.png'], dtype='<U16')
```
接下来到了我认为fastai中最需要掌握的地方，就是数据集的搭建。

在pytorch中，数据集的构建分为两步：`Dataset()`和`DataLoader()`。`Dataset()`的原型如下所示：

```python
class Dataset:
    def __init__(self):
        pass
        
    def __getitem__(self, index):

        pass
    def __len__(self):
```
也就是说，用户只需要重写`__getitem__`和`__len__`这两个内置函数就行了。`__getitem__`是说，这个对象可以通过`o[3]`这种形式取出第三个变量，通过`len(o)`这种形式获取变量个数。

但是`Dataset()`只是定义了一个类`list`变量，实际过程中一般都是利用`mini-batch`进行训练，同时还要做一些比如统一图片大小、进行数据增强等等的工作。在pytorch中就是使用`DataLoader()`：

```python
class DataLoader(dataset,
				 batch_size=1, 
				 shuffle=False, 
				 sampler=None, 
				 batch_sampler=None, 
				 num_workers=0, 
				 collate_fn=<function default_collate>,
				 pin_memory=False, 
				 drop_last=False)
```
与pytorch十分类似，fastai也采用了这样的思路，不过设置为链式调用的方法对数据集进行设置。他们把最后能直接用来训练的数据类型叫做`DataBunch`，里面含有训练集、验证集和测试集（可选）。具体可以看下文档：

> [https://docs.fast.ai/data_block.html](https://docs.fast.ai/data_block.html)

整个链式调用的过程被称为`data block api`，可以简单的理解成创建fastai学习期所需的数据准备过程就可以了。

```python
src = (SegmentationItemList.from_folder(path_img)
       # Load in x data from folder
       .split_by_fname_file(path/'valid.txt')
       # Split data into training and validation set 
       .label_from_func(get_y_fn, classes=codes)
       # Label data using the get_y_fn function
)
```
这里的`SegmentationItemList`就可以理解成专为图片segmentation任务设置的特殊的`dataset`，这一步就是通过某种规则（这里是上面定义的查找`label`文件的函数）来生成训练集和验证集的`dataset`。

之后我们再对`src`进行图片增强，批处理等等工作，相当于`DataLoader`。最后生成的对象就可以直接输入模型了：

```python
tfms = get_transforms()
bs = 16
size= 128
data = (src.transform(tfms, size=size, tfm_y=True)
        # Flip images horizontally 
        .databunch(bs=bs)
        # Create a databunch
        .normalize(imagenet_stats)
        # Normalize for resnet
)
```
看一下结果

```python
data.show_batch(figsize=(5, 5))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200406111021791.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70 =400x400)
终于到了搭建训练器这一步了，因为camvid给的label有很多缺失值，我们先自己定义一个metrics函数：

```python
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
```
意思很简单，就是去掉void值之后再评价。然后搭建一个训练器，我们使用`resnet50`提取图片特征，然后输入`unet`进行segmentaion

```python
learn = unet_learner(data, models.resnet50, metrics=acc_camvid, wd=1e-2)
```
深度学习中学习率的选择很重要，fastai提供了一个很酷的方法帮助我们：

```python
learn.lr_find()
learn.recorder.plot()
```
**OUT**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200406130323784.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70)
这幅图显示的是在不同的学习率下的`Loss`值，因为我们使用梯度下降算法，所以找到梯度最大的地方，大概是在`1e-5`。然后就可以开始训练：

```python
lr = 1e-5
learn.fit_one_cycle(10, slice(lr), pct_start=0.8)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200406130539770.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70)
`fit_one_cycle`的意思是说，学习率先变大在变小，比较符合训练的过程：

```python
learn.recorder.plot_lr()
```
**OUT**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200406130713360.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70 =300x180)
损失函数如图所示：

```python
learn.recorder.plot_losses()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200406130824868.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70 =300x180)
保存模型后看看结果：

```python
learn.save('camvid-stage-1') # save model
learn.show_results(rows=3, figsize=(8, 9)) # show results
```
**OUT**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200406130931903.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70)
如果不满意还可以调整学习率接着训练，满意的话可以用learn.predict()预测一下：

```python
mask_pred = learn.predict(data.train_ds[0][0])
Image.show(data.train_ds[0][0])
Image.show(mask_pred[0])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200406131133293.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020040613114117.png)
效果还是相当不错的。
完整的代码在github：

[^1]: [https://zhuanlan.zhihu.com/p/44958351](https://zhuanlan.zhihu.com/p/44958351)
[^2]:[https://blog.csdn.net/qq_41185868/article/details/100146896](https://blog.csdn.net/qq_41185868/article/details/100146896)
