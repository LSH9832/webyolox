# WEB-YOLOX: 打开网页，实时管理你的YOLOX训练过程！

<div align="center"><img src="src/details.png"></div>

# webyolox
2021年旷视研究院推出的YOLOX算法性能相当可以，给的源码条理也非常清晰，嵌入式部署也非常方便，但无论如何，训练的时候部署数据集还有相应环境等还是很费时间的，所以呢，这个网页端管理界面能够帮你节省很多时间。
## 快速开始
**不想看介绍想直接用**的同学（我就是这种人，建议**至少还是看一下注意**）copy下面代码到服务器上运行就行(前提是安装完了**cuda,cudnn,torch,torchvision**哈)
```shell
sudo apt install screen
git clone https://github.com/LSH9832/webyolox
cd webyolox
pip3 install -r requirements.txt
python3 main.py -p 8080 --debug
```
然后打开浏览器，输入
```shell
服务器的局域网IP:8080
```
即可，初始用户名和密码都是admin，登录后可修改。
## 注意
- ~~本项目只能在Linux系统中使用~~，目前已支持Windows, 但仍建议在Ubuntu下运行。我只在Ubuntu下用过这个项目，别的Linux系统能不能跑不知道（Debian应该是能跑的），并确保你的计算机是配有Nvidia的显卡的，而且显卡驱动也正常安装成功了的。
- 安装好CUDA和cudNN再使用本项目。
- 自己安装好numpy>=1.19.5, torch>=1.7, torchvision再使用本项目。
- 网页服务程序不会占用显卡，只有在你点击“开始训练”后，才会另起一个程序运行训练代码。
- 网页端很多地方没有做输入非法验证，用的时候尽量不要乱输入请求的地址，还有改掉默认的用户名和密码的时候也不要输入特殊字符，没有做屏蔽特殊字符的功能（懒得做了，真的，这个项目就我一个人做，做了一个多星期了，疲惫ing。。。），你要这样把程序搞崩了我也没办法。就正常用就好。
- 最好不要把这个web服务端口开放到公网上，我也不知道安全不安全。
- 数据集一定要是COCO格式的，也就是在数据集的主目录下，文件夹annotations（**注意文件夹的名字一定是annotations**）里放json标签文件，然后其他文件夹分别放训练集图片和验证集图片（文件夹名字不限，两个集合的图片所在文件夹可以相同），**训练集主目录下一定要放一个类别文件，并命名为classes.txt，每行写好类别的名称，中间和结尾不要有空行（格式见./yolox/coco_classes.txt）** 附上COCO2017数据集下载地址<br> [训练集图像（18G）](http://images.cocodataset.org/zips/train2017.zip) <br> [验证集图像（1G）](http://images.cocodataset.org/zips/val2017.zip) <br> [测试集图像（6G）](http://images.cocodataset.org/zips/test2017.zip) <br> [训练集/验证集标签（241M）](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- 想要使用预训练模型文件的同学请到 **[yolox原项目](https://github.com/Megvii-BaseDetection/YOLOX)** 里面下载，这里给出链接。<br> [yolox_s.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth)<br> [yolox_m.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth)<br> [yolox_l.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth)<br> [yolox_x.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth)<br> [yolox_tiny.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth)<br> [yolox_nano.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth)
- 如果使用的是非COCO数据集，类别不是80个，训练开始时只能导入backbone，则到本项目的release里下载backbone权重文件 **[点此进入](https://github.com/LSH9832/webyolox/releases/tag/0.2.0)** 。然后将下载好的权重文件放入文件夹weight下。


## 0. 前言
### 0.1 为啥会想做一个网页端的管理界面
实验室一共2台服务器，一共8块显卡，之前我都是用QT写一个简单的配置界面就完事儿了。但是课题组有大概10个人在用，图形界面最多就只能两个人同时使用，大多数人都是靠pycharm通过ssh远程连接,全命令行部署训练环境以及相应配置肯定是没有图形界面快的；同时远程连接服务器，使用图形界面的远程控制有的时候会非常卡，所以网页端是个不错的选择，配置完后一劳永逸啊有没有！<br><br>
### 0.2 暂时没有
## 1. 部署本项目
### 1.1 安装相应依赖
首先，你要保证在上述**注意**中提到的依赖都装好了,检查一下
```shell
nvidia-smi
nvcc -V
```
然后
```shell
sudo apt install screen
```
这个应该服务器都会装这个，好用得很，训练的程序也是在screen里运行的。
### 1.2 下载并配置本项目
```shell
git clone https://github.com/LSH9832/webyolox.git
cd webyolox
pip3 install -r requirements.txt
```
### 1.3 设置开机自启动（可选）
用我的另一个上传的项目addstart, 首先打开一个bash脚本
```shell
sudo nano /usr/bin/webyolox
```
然后写下
```shell
cd 本项目的主目录
python3 main.py -p 8080 --debug&
```
自己可以去改端口，不写debug，在发生错误时就不会在网页端显示报错详情。
保存并关闭，然后
```shell
sudo chmod +x /usr/bin/webyolox
```
然后root运行addstart项目下的addstart.py文件
```shell
sudo su
python3 addstart.py
```
按照提示输入即可，完成后下次开机就自启动了，如果不方便重启，就输入
```shell
sudo service webyolox start
```
即可，本项目就运行成功了。

### 1.4 设置解释器
相信很多小伙伴都在服务器上装了虚拟环境或者conda环境，pytorch并不安装在系统的python环境中，所以登录后点击网页左上角就可以改训练时用的python解释器了。

## 2. 使用中可能需要注意的问题（想到新的就更）
### 2.1 web运行环境与训练运行环境
web运行环境安装
- pyyaml
- flask

即可，剩下的都要安装到训练环境中。
为了方便，最好就是web运行环境与训练环境都用同一个环境。
### 2.2 训练配置
#### 2.2.1 batch_size
设置训练配置的时候，batch_size一定要是你所要使用的GPU个数的整数倍！！不然会报错！！
#### 2.2.2 使用预训练模型
若要使用预训练模型，只需填写“./weight/XXX_backbone.pth”即可。然后生成了完整的模型文件后，如果断点继续训练，则填写“./settings/训练名称/output/last.pth”

### 2.3 训练开始时
刚开始点了“开始训练”后，训练程序会过几秒后甚至十几秒后才会生成日志文件，所以请等一会儿再点击“查看日志”，在日志中显示开始第一轮训练后才会生成数据，此时再打开“详情”才会有相应数据显示。

### 2.4 使用自己训练的权重文件
如果已经训练了至少一个epoch，则会有权重文件生成，点进列表的“详情”中即可下载到自己的电脑中，如果要使用的话，在登录进去的列表页面，右上角有一个“下载测试代码”，可以运行detect.py文件查看效果。在运行前打开配置文件detect_settings.yaml进行修改，把模型大小改成自己模型文件对应的大小，模型文件名改成自己训练的权重文件的名字，然后source改成自己的摄像头或者视频名称即可。

```yaml
# detect params
confidence_thres: 0.2
nms_thres: 0.3
device: 'gpu'  # cpu
auto_choose_device: true
input_size: 640
fp16: true

# weight file
weight_size: 's'                    # m,l,x,tiny
model_path: './best.pth'            # 改成你自己权重文件的名字
classes_file: './coco_classes.txt'
is_trt_file: false                  # 是否使用的是用trt.py转换为tensorrt加速后的文件，转换方法见文件夹内txt文件
```

然后就可以运行
```python3
python3 detect.py
```

看看训练得怎么样了！

## 3.模型转换
在yolox文件夹下有转换为onnx的torch2onnx.py文件。<br>
在“下载测试代码”的主文件夹下，有转换为tensorrt模型的TensorRTconverter.py文件。<br>

上述两个文件修改相关参数后在该目录下运行即可。
