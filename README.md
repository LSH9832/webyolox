# WEB-YOLOX: 在网页端管理你的YOLOX训练过程！
2021年旷视研究院推出的YOLOX算法性能相当可以，给的源码条理也非常清晰，嵌入式部署也非常方便，但无论如何，训练的时候部署数据集还有相应环境等还是很费时间的，所以呢，这个网页端管理界面能够帮你节省很多时间。
## 快速开始
**不想看介绍想直接用**的同学（我就是这种人，建议至少还是看一下注意）copy下面代码到服务器上运行就行(前提是安装完了**cuda,cudnn,torch,torchvision**哈)
```bash
sudo apt install screen
git clone https://github.com/LSH9832/webyolox.git
cd webyolox
pip3 install -r requirements.txt
python3 main.py -p 8080 --debug
```
然后打开浏览器，输入
```bash
服务器的局域网IP:8080
```
即可，初始用户名和密码都是admin，登录后可修改。
## 注意
- 本项目只能在Linux系统中使用，而且我只在Ubuntu下用过这个项目，别的Linux系统能不能跑不知道（Debian应该是能跑的），并确保你的计算机是配有Nvidia的显卡的，而且显卡驱动也正常安装成功了的。
- 安装好CUDA和CUDnn再使用本项目
- 自己安装好numpy>=1.19.5, torch>=1.7, torchvision再使用本项目。
- 网页服务程序不会占用显卡，只有在你点击“开始训练”后，才会另起一个程序运行训练代码。
- 网页端很多地方没有做输入非法验证，用的时候尽量不要乱输入请求的地址，还有改掉默认的用户名和密码的时候也不要输入特殊字符，没有做屏蔽特殊字符的功能（懒得做了，真的，这个项目就我一个人做，做了一个多星期了，疲惫ing。。。），你要这样把程序搞崩了我也没办法。就正常用就好。
- 最好不要把这个web服务端口开放到公网上，我也不知道安全不安全。
- 数据集一定要是COCO格式的，也就是在数据集的主目录下，文件夹annotations（**注意文件夹的名字一定是annotations**）里放json标签文件，然后其他文件夹分别放训练集图片和验证集图片（文件夹名字不限，两个集合的图片所在文件夹可以相同）
- 目前还没有想到别的，想到了再加。

## 0. 前言
### 0.1 为啥会想做一个网页端的管理界面
实验室一共2台服务器，一共8块显卡，之前我都是用QT写一个简单的配置界面就完事儿了。但是课题组有大概10个人在用，图形界面最多就只能两个人同时使用，大多数人都是靠pycharm通过ssh远程连接,全命令行部署训练环境以及相应配置肯定是没有图形界面快的；同时远程连接服务器，使用图形界面的远程控制有的时候会非常卡，所以网页端是个不错的选择。<br><br>
本来我是想写一个Linux和Windows通用的，后来写着写着发现有一些功能只能在Linux下使用，后来想想应该不会有人在GPU服务器上使用Windows系统进行深度学习相关的模型训练吧（大概应该可能也许...不会吧?会我也没办法，怪我才疏学浅不会弄哈）
### 0.2 暂时没有
## 1. 部署本项目
### 1.1 安装相应依赖
首先，你要保证在上述**注意**中提到的依赖都装好了,检查一下
```bash
nvidia-smi
nvcc -V
```
然后
```bash
sudo apt install screen
```
这个应该服务器都会装这个，好用得很，训练的程序也是在screen里运行的。
### 1.2 下载并配置本项目
```bash
git clone https://github.com/LSH9832/webyolox.git
cd webyolox
pip3 install -r requirements.txt
```
### 1.3 设置开机自启动（可选）
用我的另一个上传的项目addstart, 首先打开一个bash脚本
```bash
sudo nano /usr/bin/webyolox
```
然后写下
```bash
cd 本项目的主目录
python3 main.py -p 8080 --debug&
```
自己可以去改端口，不写debug，在发生错误时就不会在网页端显示报错详情。
保存并关闭，然后
```bash
sudo chmod +x /usr/bin/webyolox
```
然后root运行addstart项目下的addstart.py文件
```bash
sudo su
python3 addstart.py
```
按照提示输入即可，完成后下次开机就自启动了，如果不方便重启，就输入
```bash
sudo service webyolox start
```
即可，本项目就运行成功了。

### 1.4 设置解释器
相信很多小伙伴都在服务器上装了虚拟环境或者conda环境，pytorch并不安装在系统的python环境中，所以登录后点击网页左上角就可以改训练时用的python解释器了。

## 2. 使用中可能需要注意的问题（想到新的就更）
### 2.1 训练配置
设置训练配置的时候，batch_size一定要是你所要使用的GPU个数的整数倍！！因为这个一直报错别怪我咯~
### 2.2 使用自己训练的权重文件
如果已经训练了至少一个epoch，则会有权重文件生成，点进列表的“详情”中即可下载到自己的电脑中，如果要使用的话，在登录进去的列表页面，右上角有一个“下载测试代码”，里面有4个demo可以运行，分别是
- predict.py: 单进程，原始权重文件
- predict_trt.py: 单进程，经过tensorRT转换后的权重文件（需要安装相关环境，这里不过多介绍，使用测试代码中的trt.py转化，转化方式见How To Generate and Use trt file.txt）
- multi_pro_predict.py: 多进程，原始文件
- multi_pro_predict_trt.py: 多进程，tensorRT文件
打开这些文件中的任意一个，把模型大小改成自己模型文件对应的大小，模型文件名改成自己训练的权重文件的名字，然后source改成自己的摄像头或者视频名称即可。
```python3
weight_type = 's'     # 找到这一行，改成自己的权重文件对应的大小（s,m,l,x,tiny）
detector.setModelPath('weights/yolox_s.pth') # 找到这一行，改权重文件
source = 0   # 找到这一行，改成自己的摄像头地址或视频地址
```
然后就可以运行看看训练得怎么样了！
