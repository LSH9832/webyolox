import os
import cv2
import torch
import sys
import argparse
import yaml


this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_path) if not this_path in sys.path else None   #print('start from this project dir')

import yolox.exp
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import fuse_model, postprocess, vis


def make_parser():
    parser = argparse.ArgumentParser("yolox detect parser")
    parser.add_argument("-s", "--source", type=str or int, default=source, help="source")
    parser.add_argument("-t", "--source-type", type=str, default=source_type, help="source type: cam, vid, image, image_dir")
    parser.add_argument("-p", "--pause", default=bool(start_with_pause), action='store_true', help="start with pause")
    parser.add_argument("-m", "--multi", default=False, action='store_true', help="run with multiprocess")
    return parser


class Colors:
    def __init__(self):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness

    c1, c2 = (int(x[0]), int((x[1]))), (int(x[2]), int((x[3])))
    # print(c1,c2)
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], (c1[1] - t_size[1] - 3) if (c1[1] - t_size[1] - 3) > 0 else (c1[1] + t_size[1] + 3)
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2) if (c1[1] - t_size[1] - 3) > 0 else (c1[0], c2[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_bb(img, pred, names, type_limit=None, line_thickness=2):
    if type_limit is None:
        type_limit = names
    for *xyxy, conf0, conf1, cls in pred:
        conf = conf0 * conf1
        if names[int(cls)] in type_limit:
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img, label=label, color=colors(int(cls), True), line_thickness=line_thickness)


class DirCapture(object):
    """read image file from a dir containing images or a image file"""
    __cv2 = cv2
    __support_type = ['jpg', 'jpeg', 'png', 'bmp']
    __img_list = []

    def __init__(self, path: str = None):
        if path is not None:
            self.open(path)

    def open(self, path):
        self.__img_list = []
        if os.path.isdir(path):
            path = path[:-1] if path.endswith('/') else path
            assert os.path.isdir(path)
            from glob import glob

            for img_type in self.__support_type:
                self.__img_list += sorted(glob('%s/*.%s' % (path, img_type)))
        elif os.path.isfile(path) and '.' in path and path.split('.')[-1] in self.__support_type:
            self.__img_list = [path]
        else:
            print('wrong input')
            self.__img_list = []

    def isOpened(self):
        return bool(len(self.__img_list))

    def read(self):
        this_img_name = self.__img_list[0]
        del self.__img_list[0]
        img = self.__cv2.imread(this_img_name)
        success = img.size > 0
        return success, img

    def release(self):
        self.__img_list = []


class Predictor(object):

    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,  # 类型名称
            trt_file=None,  # tensorRT File
            decoder=None,  # tensorRT decoder
            device="cpu",
            fp16=False,  # 使用混合精度评价
            legacy=False,  # 与旧版本兼容
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():

            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            if outputs[0] is not None:
                outputs = outputs[0].cpu().numpy()
                outputs[:, 0:4] /= ratio
            else:
                outputs = []
        return outputs


class listString(str):

    def __init__(self, this_string: str = ''):
        super(listString, self).__init__()
        self.__this_string = this_string
        # self.__all__ = ['append', ]

    def __getitem__(self, index):
        return self.__this_string[index]

    def __repr__(self):
        return self.__this_string

    def __len__(self):
        return len(self.__this_string)

    def append(self, add_string):
        self.__this_string += add_string


class Detector(object):
    __model_size_all = ['s', 'm', 'l', 'x', 'tiny', 'nano', 'v3']
    __device = 'cpu'
    __model = None  # 模型
    __model_size = None  # 模型大小（s,m,l,x, tiny, nano, v3）
    __model_path = None  # 模型权重文件位置
    __class_names = COCO_CLASSES  # 类别名称
    __detector = None  # 检测器
    __exp = None
    __fp16 = False
    __fuse = False

    __useTRT = False  # 使用TensorRT
    __legacy = False  # To be compatible with older versions
    __auto_choose_device = True

    __tsize = 640
    __conf = 0.25
    __nms = 0.5

    __result = None
    __img_info = None

    def __init__(
            self,
            model_path: str or None = None,  # 模型权重文件位置
            model_size: str or None = None,  # 模型大小（s,m,l,x,tiny,）
            class_path: str or None = None,  # 类别文件位置
            conf: float or None = None,  # 置信度阈值
            nms: float or None = None,  # 非极大值抑制阈值
            autoChooseDevice: bool = True,  # 自动选择运行的设备（CPU，GPU）
    ):

        self.__model_path = model_path
        self.__model_size = model_size
        self.__auto_choose_device = autoChooseDevice
        self.reloadModel = self.loadModel

        if class_path is not None:
            self.__load_class(class_path)
        if conf is not None:
            self.__conf = conf
        if nms is not None:
            self.__nms = nms

        self.__check_input()

        # self.

    #######################################################################################################
    # private function
    def __load_class(self, path):
        if os.path.exists(path):
            data = open(path).readlines()
            classes = []

            [classes.append(this_class[:-1] if this_class.endswith('\n') else this_class)
             if len(this_class) else None
             for this_class in data]

            self.__class_names = classes

    def __check_input(self):
        if self.__model_path is not None:
            this_error = '[model path input error]: Type of model path should be "string"!'
            assert type(self.__model_path) == str, this_error
            # print(self.__model_path)
            assert self.__model_path.endswith('.pth'), '[model path type error]:not a weight file'

        if self.__model_size is not None:
            allSizeStr = listString('[model path input error]: Available model size: ')
            [allSizeStr.append('%s, ' % this_size) for this_size in self.__model_size_all]
            assert self.__model_size in self.__model_size_all, '%s' % allSizeStr

    def __cuda(self):
        assert torch.cuda.is_available()
        assert self.__model is not None
        self.__model.cuda()
        if self.__fp16:
            self.__model.half()
        self.__model.eval()

    def __cpu(self):
        assert self.__model is not None
        self.__model.cpu()
        self.__model.eval()

    #######################################################################################################
    # public function
    def get_all_classes(self):
        return self.__class_names

    ################################################################################
    """
    you can use the following setting functions only before loading model, or you should reload model
    """

    def setModelPath(self, model_path: str) -> None:
        self.__model_path = model_path
        self.__check_input()

    def setModelSize(self, model_size: str) -> None:
        self.__model_size = model_size
        self.__check_input()

    def setClassPath(self, class_path: str) -> None:
        self.__load_class(class_path)

    def setAutoChooseDevice(self, flag: bool) -> None:
        self.__auto_choose_device = flag

    def setFuse(self, flag: bool) -> None:
        self.__fuse = flag

    def setLegacy(self, flag: bool) -> None:
        self.__legacy = flag

    def setDevice(self, device: str) -> None:
        assert device in ['cpu', 'gpu'], '[Device name error]: No device named %s' % device
        self.__device = device

    def setTsize(self, size: int) -> None:
        self.__tsize = size

    def setUseTRT(self, flag:bool):
        self.__useTRT = flag

    def setFp16(self, flag:bool):
        self.__fp16 = flag
    ################################################################################
    """
    you can use the following setting functions after loading model
    """

    def setConf(self, conf: float) -> None:
        self.__conf = conf
        if self.__detector is not None:
            self.__detector.confthre = conf

    def setNms(self, nms: float) -> None:
        self.__nms = nms
        if self.__detector is not None:
            self.__detector.nmsthre = nms

    ################################################################################
    def loadModel(self) -> None:
        assert self.__model_size is not None, 'model size not declared'
        assert self.__model_path is not None, 'model path not declared'

        # 载入网络结构
        self.__exp = yolox.exp.build.get_exp_by_name('yolox-%s' % self.__model_size)
        self.__exp.test_conf = self.__conf
        self.__exp.nmsthre = self.__nms
        self.__exp.test_size = (self.__tsize, self.__tsize)

        self.__model = self.__exp.get_model()

        if self.__auto_choose_device:
            if torch.cuda.is_available():
                self.__cuda()
            else:
                self.__cpu()
        else:
            if self.__device == 'cpu':
                self.__cpu()
            elif torch.cuda.is_available():
                self.__cuda()
            else:
                print('cuda is not available, use cpu')
                self.__cpu()

        trt_file = None
        decoder = None
        if not self.__useTRT:
            # 载入权重
            pt = torch.load(self.__model_path, map_location="cpu")
            # for name in pt:
            #     print(name)
            pt['classes'] = self.__class_names

            self.__model.load_state_dict(pt["model"])

            if self.__fuse:
                self.__model = fuse_model(self.__model)

        else:
            trt_file = self.__model_path
            self.__model.head.decode_in_inference = False
            decoder = self.__model.head.decode_outputs

        # 预测器
        self.__detector = Predictor(
            self.__model,
            self.__exp,
            self.__class_names,
            trt_file,
            decoder,
            self.__device,
            self.__fp16,
            self.__legacy,
        )

    def predict(self, image):
        if self.__detector is None:
            self.loadModel()

        # 预测
        image_use = image.copy()
        self.__result = self.__detector.inference(image_use)
        return self.__result


file_settings = None
if os.path.isfile('./detect_settings.yaml'):
    file_settings = yaml.load(open('./detect_settings.yaml'), yaml.Loader)
confidence_thres = file_settings['confidence_thres'] if file_settings is not None else 0.4
nms_thres = file_settings['nms_thres'] if file_settings is not None else 0.5
device = file_settings['device'] if file_settings is not None else 'gpu'
input_size = file_settings['input_size'] if file_settings is not None else 640
auto_choose_device = file_settings['auto_choose_device'] if file_settings is not None else True
weight_size = file_settings['weight_size'] if file_settings is not None else 's'
model_path = file_settings['model_path'] if file_settings is not None else './best.pth'
is_trt_file = file_settings['is_trt_file'] if file_settings is not None else False
fp16 = file_settings['fp16'] if file_settings is not None else False
classes_file = file_settings['classes_file'] if file_settings is not None else 'coco_classes.txt'
source = file_settings['source'] if file_settings is not None else 0
source_type = file_settings['source_type'] if file_settings is not None else 'cam'
save_image = file_settings['save_image'] if file_settings is not None else False
show_image = file_settings['show_image'] if file_settings is not None else True
start_with_pause = int(file_settings['start_with_pause'] if file_settings is not None else False)


parse = make_parser().parse_args()
source = int(parse.source) if parse.source.isdigit() else parse.source
source_type = parse.source_type
start_with_pause = parse.pause
# print(parse)


def detect(my_dict):
    from time import time
    import numpy as np
    
    weight_type = weight_size
    detector = Detector(
        model_path=model_path,
        model_size=weight_type,
        class_path=classes_file,
        conf=confidence_thres,
        nms=nms_thres,
        autoChooseDevice=auto_choose_device
    )

    """
    Before running loadModel, you can change params by using the following functions.
    You can also create a Detector without any params like this:

    detector = Detector()

    and then input params by using these functions.
    """

    # detector.setConf(0.4)
    # detector.setNms(0.5)
    detector.setDevice(device)
    detector.setTsize(input_size)
    detector.setUseTRT(is_trt_file)
    detector.setFp16(fp16)
    # detector.setAutoChooseDevice(True)
    # detector.setModelPath('weights/yolox_s.pth')

    """
    Then load model, it will take some time, Never forget this step!!!
    """
    detector.loadModel()

    """
    Start Detection
    """
    my_dict['classes'] = detector.get_all_classes()
    detector.predict(np.zeros([640,640,3]))  # 推理
    my_dict['run'] = True
    while my_dict['run']:  # 如果程序仍需要运行
        if my_dict['updated']:  # 如果图像已经更新
            img = my_dict['img']  # 获取图像
            my_dict['updated'] = False  # 设置图像状态为未更新
            t0 = time()  # 开始计时
            result = detector.predict(img)  # 推理
            my_dict['pre_fps'] = 1. / (time() - t0)  # 结束计时并计算FPS
            my_dict['result'] = result  # 存储结果
            my_dict['update_result'] = True  # 设置结果状态为已更新


def show(my_dict):

    from time import time, strftime, localtime

    if source_type in ['cam', 'vid']:
        cam = cv2.VideoCapture(source)
    elif source_type in ['image_dir', 'image']:
        cam = DirCapture(str(source))
    else:
        print('wrong source type')
        cam = DirCapture()

    fpss = []
    result = []
    time_delay = 1 - start_with_pause

    print('wait for model-loading')
    while not my_dict['run']:
        pass
    t0 = time()
    while cam.isOpened():
        t1 = time()
        success, frame = cam.read()
        if success:

            # frame = cv2.resize(frame, (210 * 6, 90 * 6))
            my_dict['img'] = frame
            my_dict['updated'] = True

            fpss.append((1 / (time() - t0)))
            if len(fpss) > 10:
                fpss = fpss[1:]
            now_mean_fps = sum(fpss) / len(fpss)
            print('\r播放帧率=%.2fFPS, 推理帧率=%.2fFPS' % (now_mean_fps, my_dict['pre_fps']), end='')

            t0 = time()
            if my_dict['update_result']:
                result = my_dict['result']
                my_dict['update_result'] = False
            if len(result):
                draw_bb(frame, result, my_dict['classes'])

            if save_image:
                img_name = strftime("%Y_%m_%d_%H_%M_%S.jpg", localtime())
                cv2.imwrite(img_name, frame)

            cv2.imshow('yolox detect', frame) if show_image else None

            key = cv2.waitKey(time_delay)
            if key == 27:
                cv2.destroyAllWindows()
                break
            elif key == ord(' '):
                time_delay = 1 - time_delay
        else:
            break
        while not time() - t1 >= 0.03:
            pass

    print('')
    my_dict['run'] = False
    cv2.destroyAllWindows()


def single():
    import time
    weight_type = weight_size
    detector = Detector(
        model_path=model_path,
        model_size=weight_type,
        class_path=classes_file,
        conf=confidence_thres,
        nms=nms_thres,
        autoChooseDevice=auto_choose_device
    )

    detector.setDevice(device)
    detector.setTsize(input_size)
    detector.setUseTRT(is_trt_file)
    detector.setFp16(fp16)

    detector.loadModel()

    #########################################################################################################

    if source_type in ['cam', 'vid']:
        cap = cv2.VideoCapture(source)
    elif source_type in ['image_dir', 'image']:
        cap = DirCapture(str(source))
    else:
        print('wrong source type')
        cap = DirCapture()

    #########################################################################################################

    t0 = time.time()
    wait_time = 1 - start_with_pause

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            t1 = time.time()
            results = detector.predict(frame)
            draw_bb(frame, results, detector.get_all_classes())
            print('\r播放帧率=%.2fFPS, 推理帧率=%.2fFPS' % ((1 / (time.time() - t0)), (1 / (time.time() - t1))), end='')

            t0 = time.time()

            if show_image:
                cv2.imshow('results', frame)

            if save_image:
                img_name = time.strftime("%Y_%m_%d_%H_%M_%S.jpg", time.localtime())
                cv2.imwrite(img_name, frame)

            key = cv2.waitKey(wait_time)
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                break
            elif key == ord(' '):
                wait_time = 1 - wait_time
    print('')


def multy():
    print('starting multiprocess')
    from multiprocessing import Process, Manager, freeze_support
    freeze_support()
    d = Manager().dict()

    d['run'] = False
    d['updated'] = False
    d['img'] = None
    d['result'] = []
    d['pre_fps'] = 0
    d['classes'] = []
    d['update_result'] = False

    processes = [Process(target=show, args=(d,)),
                 Process(target=detect, args=(d,))]
    [process.start() for process in processes]
    [process.join() for process in processes]


if __name__ == '__main__':
    multy() if parse.multi else single()


