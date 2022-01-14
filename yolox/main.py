from qt.main_ui2py import Ui_MainWindow as MAIN

from sys             import argv, exit

from PyQt5.QtCore    import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui     import QIcon

# from PyQt5           import *
# import multiprocessing as mp
import os
from glob import glob
import datetime

def browse_file(title, start_dir, accept_type):
    this_string = ""
    for this_type in accept_type:
        if len(this_type):
            this_string += "%s Files (*.%s);;" % (this_type, this_type)
    this_string += "All Files (*)"

    return QFileDialog.getOpenFileName(None, title, start_dir, this_string)


def browse_dir(title, start_dir):
    return QFileDialog.getExistingDirectory(None,title, start_dir)

# 主窗口
#######################################################################################################################
class MainWindow(QMainWindow, MAIN):

    # 初始化函数
    ###################################################################################################################
    def __init__(self, parent=None):

        super(MainWindow, self).__init__(parent)
        super(MainWindow, self).setupUi(self)
        # loadUi('main.ui', self)

        # 固定窗口大小
        self.setFixedSize(self.size())

        # 加载图标
        self.loadIcon()

        self.timer = QTimer()  # 定时器，或者计时器， whatever
        self.timer.start(1)
        self.timer.timeout.connect(self.timeoutFun)   # 每2000毫秒自动运行一次的函数


        # 激活全部按钮、菜单选项、下拉列表用于测试，实际使用时注释掉
        self.data_root_browse.setEnabled(True)
        self.output_browse.setEnabled(True)

        self.anno_convert.setEnabled(True)
        self.dataset_devide.setEnabled(True)

        # self.train_dir.setEnabled(True)
        # self.val_dir.setEnabled(True)
        # self.train_anno.setEnabled(True)
        # self.val_anno.setEnabled(True)
        self.model_size.setEnabled(True)

        # 事件连接函数
        self.data_root_browse.clicked.connect(self.data_root_browse_ClickFun)
        self.output_browse.clicked.connect(self.output_browse_ClickFun)
        self.weight_browse.clicked.connect(self.weight_browse_ClickFun)
        self.check_settings.clicked.connect(self.check_settings_ClickFun)
        self.clear_msg.clicked.connect(self.clear_msg_ClickFun)

        self.use_pretrained_weight.clicked.connect(self.use_pretrained_weight_ClickFun)


        self.anno_convert.triggered.connect(self.anno_convert_ClickFun)
        self.dataset_devide.triggered.connect(self.dataset_devide_ClickFun)

        self.train_dir.currentIndexChanged.connect(self.train_dir_ClickFun)
        self.val_dir.currentIndexChanged.connect(self.val_dir_ClickFun)
        self.train_anno.currentIndexChanged.connect(self.train_anno_ClickFun)
        self.val_anno.currentIndexChanged.connect(self.val_anno_ClickFun)
        self.model_size.currentIndexChanged.connect(self.model_size_ClickFun)

        self.data_root.setPlaceholderText('/path/to/your/dataset')
        self.output_dir.setText("")
        self.output_dir.setPlaceholderText('recommend path:   /path/to/your/dataset/yolox_train_output')

        # self.loadSettingsFromFile()

        self.now_msg = []

        # get gpu number of your device
        gpu_list = os.popen("nvidia-smi -L").readlines()
        print(gpu_list)
        gpu_num = len(gpu_list)
        self.gpu_num.setMinimum(0)
        self.gpu_num.setMaximum(0)
        if gpu_num and gpu_list[0].startswith('GPU'):
            self.gpu_num.setMinimum(1)
            self.gpu_num.setMaximum(gpu_num)
        self.settingsLocked = self.generate_code.isEnabled()



    def update_combo_list(self, directory):
        # 除annotations以外所有其他子文件夹名称
        all_list = []
        for i, name in enumerate(glob("%s/*" % directory)):
            if os.path.isdir(name) and not name.endswith("/annotations"):
                name = name[len("%s/" % directory):]
                all_list.append(name)

        # 子目录annotations中所有json文件
        anno_list = glob("%s/annotations/*.json" % directory)

        self.train_dir.clear()
        self.val_dir.clear()
        self.train_anno.clear()
        self.val_anno.clear()

        if len(all_list):
            self.train_dir.setEnabled(True)
            self.val_dir.setEnabled(True)
            for name in all_list:
                self.train_dir.addItem(name, name)
                self.val_dir.addItem(name, name)
        else:
            self.train_dir.setEnabled(False)
            self.val_dir.setEnabled(False)
            self.train_dir.addItem("(无)", "")
            self.val_dir.addItem("(无)", "")

        if len(anno_list):
            self.train_anno.setEnabled(True)
            self.val_anno.setEnabled(True)
            for name in anno_list:
                if os.path.isfile(name):
                    name = name[len("%s/annotations/" % directory):]
                    self.train_anno.addItem(name, name)
                    self.val_anno.addItem(name, name)
        else:
            self.train_anno.setEnabled(False)
            self.val_anno.setEnabled(False)
            self.train_anno.addItem("(无)", "")
            self.val_anno.addItem("(无)", "")

    # 按钮
    def data_root_browse_ClickFun(self):
        print("你按了 " + self.data_root_browse.text() + " 这个按钮")
        home_dir = "/home/" + os.popen("echo $USER").read().rstrip("\n")
        directory = browse_dir("选择数据集的根文件夹", home_dir)
        if len(directory):
            # fresh
            self.update_combo_list(directory)

            # 如果是自动生成的保存地址则改变
            if self.output_dir.text() == '%s/yolox_train_output' % self.data_root.text() or not len(self.output_dir.text()):
                self.output_dir.setText('%s/yolox_train_output' % directory)
            self.data_root.setText(directory)


    def output_browse_ClickFun(self):
        print("你按了 " + self.output_browse.text() + " 这个按钮")
        directory = browse_dir("选择保存训练结果的根文件夹", "./")
        if len(directory):
            self.output_dir.setText(directory)

    def weight_browse_ClickFun(self):
        print("你按了 " + self.weight_browse.text() + " 这个按钮")
        file, _ = browse_file("选择权重文件", "./", ['pth'])
        print(file)
        if len(file):
            self.weight_file_name.setText(file)

    def check_settings_ClickFun(self):
        if self.generate_code.isEnabled():
            self.generate_code.setEnabled(False)

            self.data_root.setEnabled(True)
            self.data_root_browse.setEnabled(True)
            self.update_list.setEnabled(True)
            self.train_dir.setEnabled(True)
            self.val_dir.setEnabled(True)
            self.train_anno.setEnabled(True)
            self.val_anno.setEnabled(True)
            self.exp_name.setEnabled(True)
            self.model_size.setEnabled(True)
            self.epochs.setEnabled(True)
            self.batch_size.setEnabled(True)
            self.gpu_num.setEnabled(True)
            self.fp16.setEnabled(True)
            self.use_pretrained_weight.setEnabled(True)
            if self.use_pretrained_weight.isChecked():
                self.weight_browse.setEnabled(True)
                self.weight_file_name.setEnabled(True)
            self.output_dir.setEnabled(True)
            self.output_browse.setEnabled(True)


            self.check_settings.setText('配置检查并锁定')
        else:

            flag = True
            data_root = self.data_root.text()
            if data_root.endswith('/'):
                data_root = data_root[:-1]
            train_dir = data_root + '/' + self.train_dir.currentText()
            val_dir = data_root + '/' + self.val_dir.currentText()
            train_anno =  data_root + '/annotations/' + self.train_anno.currentText()
            val_anno = data_root + '/annotations/' + self.val_anno.currentText()
            batch_size = self.batch_size.value()
            gpu_num = self.gpu_num.value()
            use_pretrained_weight = self.use_pretrained_weight.isChecked()
            weight_file = self.weight_file_name.text()
            output_dir = self.output_dir.text()
            if not os.path.isdir(data_root):
                flag = False
                self.now_msg.append('【错误】数据集根文件夹不存在！')
            if not os.path.isdir(train_dir):
                flag = False
                self.now_msg.append('【错误】训练集文件夹不存在！')
            if not os.path.isdir(val_dir):
                flag = False
                self.now_msg.append('【错误】验证集文件夹不存在！')
            if not os.path.isfile(train_anno):
                flag = False
                self.now_msg.append('【错误】训练集标签文件不存在！')
            if not os.path.isfile(val_anno):
                flag = False
                self.now_msg.append('【错误】验证集标签文件不存在！')
            if gpu_num>0 and not batch_size % gpu_num == 0:
                flag = False
                self.now_msg.append('【错误】训练批量大小不是使用GPU个数的整数倍！')
            if gpu_num == 0:
                # flag = False
                self.now_msg.append('【警告】此设备没有GPU CUDA核心或没有安装相应驱动!')
            if use_pretrained_weight:
                if not os.path.isfile(weight_file):
                    flag = False
                    self.now_msg.append('【错误】预训练权重文件不存在！')
                elif not weight_file.endswith('.pth'):
                    flag = False
                    self.now_msg.append('【错误】预训练权重文件格式错误（必须是pth文件）！')
            if not os.path.isdir(output_dir):
                self.now_msg.append('【警告】训练结果保存文件夹不存在，训练开始时将被创建。')
            if train_dir == val_dir:
                self.now_msg.append('【警告】训练集目录和测试集目录相同！请确保训练集图片和测试集图片确实均在该目录下！')
            if train_anno == val_anno:
                flag = False
                self.now_msg.append('【错误】训练集标签和测试集标签不能相同！')

            if flag:
                self.generate_code.setEnabled(True)

                self.data_root.setEnabled(False)
                self.data_root_browse.setEnabled(False)
                self.update_list.setEnabled(False)
                self.train_dir.setEnabled(False)
                self.val_dir.setEnabled(False)
                self.train_anno.setEnabled(False)
                self.val_anno.setEnabled(False)
                self.exp_name.setEnabled(False)
                self.model_size.setEnabled(False)
                self.epochs.setEnabled(False)
                self.batch_size.setEnabled(False)
                self.gpu_num.setEnabled(False)
                self.fp16.setEnabled(False)
                self.use_pretrained_weight.setEnabled(False)
                self.weight_browse.setEnabled(False)
                self.weight_file_name.setEnabled(False)
                self.output_dir.setEnabled(False)
                self.output_browse.setEnabled(False)

                self.check_settings.setText('解除锁定并修改')

    def clear_msg_ClickFun(self):
        self.msg.setText('')

    # checkBox
    def use_pretrained_weight_ClickFun(self):
        self.weight_browse.setEnabled(self.use_pretrained_weight.isChecked())
        self.weight_file_name.setEnabled(self.use_pretrained_weight.isChecked())


    # 菜单选项
    def anno_convert_ClickFun(self):
        print("你按了 " + self.anno_convert.text() + " 这个菜单选项")

    def dataset_devide_ClickFun(self):
        print("你按了 " + self.dataset_devide.text() + " 这个菜单选项")


    # 下拉列表
    def train_dir_ClickFun(self):
        print("你将该下拉列表选项变成了 " + self.train_dir.currentText())

    def val_dir_ClickFun(self):
        print("你将该下拉列表选项变成了 " + self.val_dir.currentText())

    def train_anno_ClickFun(self):
        print("你将该下拉列表选项变成了 " + self.train_anno.currentText())

    def val_anno_ClickFun(self):
        print("你将该下拉列表选项变成了 " + self.val_anno.currentText())

    def model_size_ClickFun(self):
        print("你将该下拉列表选项变成了 " + self.model_size.currentText())

    # 自动运行的函数
    def timeoutFun(self):
        if len(self.now_msg):
            self.msg.setText(self.msg.toPlainText() +
                             ('\n\n' if len(self.msg.toPlainText()) else '') +
                             '[%s]' % str(datetime.datetime.now()).split('.')[0])
            while len(self.now_msg):
                self.msg.setText(self.msg.toPlainText() + '\n' + self.now_msg[0])
                del self.now_msg[0]
            self.msg.moveCursor(self.msg.textCursor().End)

    # 加载图标
    def loadIcon(self):
        icon_path = './qt/icon.png'
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))


def main():
    app = QApplication(argv)
    w = MainWindow()
    w.show()
    exit(app.exec())


if __name__ == '__main__':
    # mp.freeze_support()
    main()