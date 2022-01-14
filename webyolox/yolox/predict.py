import os
import cv2
import torch

import yolox.exp
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import fuse_model, postprocess, vis

__all__ = ["Detector"]


class Predictor(object):

    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,     # 类型名称
        trt_file=None,              # tensorRT File
        decoder=None,               # tensorRT decoder
        device="cpu",
        fp16=False,                 # 使用混合精度评价
        legacy=False,               # 与旧版本兼容
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
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

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

        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img, []
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res, information = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        # print('info: ', information)
        return vis_res, information


class listString(str):

    def __init__(self, this_string:str = ''):
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
    __model = None                      # 模型
    __model_size = None                 # 模型大小（s,m,l,x, tiny, nano, v3）
    __model_path = None                 # 模型权重文件位置
    __class_names = COCO_CLASSES         # 类别名称
    __detector = None                   # 检测器
    __exp = None
    __fp16 = False
    __fuse = False

    __useTRT = False                    # 使用TensorRT
    __legacy = False                    # To be compatible with older versions
    __auto_choose_device = True

    __tsize = 640
    __conf = 0.25
    __nms = 0.5

    __result = None
    __img_info = None


    def __init__(
            self,
            model_path:str or None = None,      # 模型权重文件位置
            model_size:str or None = None,      # 模型大小（s,m,l,x,tiny,）
            class_path:str or None = None,      # 类别文件位置
            conf:float or None = None,          # 置信度阈值
            nms:float or None = None,           # 非极大值抑制阈值
            autoChooseDevice:bool = True,       # 自动选择运行的设备（CPU，GPU）
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

    ################################################################################
    #  you can use the following setting functions only before loading model, or you should reload model
    def setModelPath(self, model_path:str) -> None:
        self.__model_path = model_path
        self.__check_input()

    def setModelSize(self, model_size:str) -> None:
        self.__model_size = model_size
        self.__check_input()

    def setClassPath(self, class_path:str) -> None:
        self.__load_class(class_path)

    def setAutoChooseDevice(self, flag:bool) -> None:
        self.__auto_choose_device = flag

    def setFuse(self, flag:bool) -> None:
        self.__fuse = flag

    def setLegacy(self, flag:bool) -> None:
        self.__legacy = flag

    def setDevice(self, device:str) -> None:
        assert device in ['cpu', 'gpu'], '[Device name error]: No device named %s' % device
        self.__device = device

    def setTsize(self, size:int) -> None:
        self.__tsize = size

    ################################################################################
    def setConf(self, conf:float) -> None:
        self.__conf = conf
        if self.__detector is not None:
            self.__detector.confthre = conf

    def setNms(self, nms:float) -> None:
        self.__nms = nms
        if self.__detector is not None:
            self.__detector.nmsthre = nms

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
            for name in pt:
                print(name)
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
        self.__result, self.__img_info = self.__detector.inference(image_use)

        # 数据处理
        result_frame, result_info = self.__detector.visual(self.__result[0], self.__img_info, self.__detector.confthre)
        return result_frame, result_info


"""DOCS"""
"""how to use class Detector"""
if __name__ == '__main__':

    detector = Detector(
        model_path='weights/yolox_s.pth',
        model_size='s',
        class_path='coco_classes.txt',
        conf=0.5,
        nms=0.5,
        autoChooseDevice=True
    )
    #########################################################################################################
    """
    Before running loadModel, you can change params by using the following functions.
    You can also create a Detector without any params like this:
    
    detector = Detector()
    
    and then input params by using these functions.
    """
    detector.setFuse(False)
    detector.setLegacy(False)
    detector.setConf(0.4)
    detector.setNms(0.5)
    detector.setDevice('gpu')
    detector.setTsize(640)
    detector.setAutoChooseDevice(True)
    #detector.setModelPath('weights/yolox_s.pth')

    #########################################################################################################
    # load model, it will take some time
    detector.loadModel()

    #########################################################################################################
    import cv2

    # source = 0
    source = 'test.mkv'
    # source = '/path/to/your/image_dir'
    # source = 'assets/dog.jpg'
    # source_type = 'image'
    source_type = 'vid'
    # source_type = 'image_dir'
    # source_type = 'image'
    show_image = True
    save_image = False

    #########################################################################################################
    """read image file from a dir containing images or a image file"""
    class DirCapture(object):
        import cv2
        __cv2 = cv2
        __support_type = ['jpg', 'jpeg', 'png', 'bmp']
        __img_list = []
        def __init__(self, path:str = None):
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

    #########################################################################################################

    if source_type in ['cam', 'vid']:
        cap = cv2.VideoCapture(source)
    elif source_type in ['image_dir', 'image']:
        cap = DirCapture(str(source))
    else:
        print('wrong source type')
        cap = DirCapture()

    #########################################################################################################
    import time
    t0 = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_show, results = detector.predict(frame)
            # print(results)
            print('\r%.1fFPS' % (1/(time.time()-t0)), end='')
            t0 = time.time()
            if show_image:
                cv2.imshow('results', img_show)
                

            if save_image:
                img_name = time.strftime("%Y_%m_%d_%H_%M_%S.jpg", time.localtime())
                cv2.imwrite(img_name, img_show)

            if cv2.waitKey(0 if source_type == 'image' else 1) in [ord('q'), ord('Q'), 27]:
                cap.release()
                cv2.destroyAllWindows()
                break

