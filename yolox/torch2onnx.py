from loguru import logger

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


class Settings:
    output_name = "yolox.onnx"
    input = "images"
    output = "output"
    opset = 11
    batch_size = 1
    dynamic = True
    no_onnxsim = False
    exp_file = None
    experiment_name = None
    name = "yolox-s"
    ckpt = None
    opts = None

    def __init__(
            self,
            name=None,
            pth_file=None,
            output_file="yolox.onnx",
            batch_size=1,
            dynamic=True,
            opset=11,
            no_onnxsim=False

    ):
        self.name = name
        self.ckpt = pth_file
        self.output_name = output_file
        self.batch_size = batch_size
        self.opset = opset
        self.dynamic = dynamic
        self.no_onnxsim = no_onnxsim


class Convertor:

    def __init__(
            self,
            name,
            pth_file,
            output_file="yolox.onnx",
            batch_size=1,
            dynamic=True,
            opset=11,
            no_onnxsim=False
    ):
        self.settings = Settings(name, pth_file, output_file, batch_size, dynamic, opset, no_onnxsim)
        self.exp = get_exp(self.settings.exp_file, self.settings.name)
        self.exp.merge(self.settings.opts) if self.settings.opts is not None else None

        logger.info("loading checkpoint ...")

        self.model = self.exp.get_model()
        ckpt = torch.load(self.settings.ckpt, map_location="cpu")
        ckpt = ckpt["model"] if "model" in ckpt else ckpt

        self.model.eval()
        self.model.load_state_dict(ckpt)

        self.model = replace_module(self.model, nn.SiLU, SiLU)
        self.model.head.decode_in_inference = False

        logger.info("loading checkpoint done.")

    def convert(self):
        dummy_input = torch.randn(self.settings.batch_size, 3, self.exp.test_size[0], self.exp.test_size[1])

        torch.onnx._export(
            self.model,
            dummy_input,
            self.settings.output_name,
            input_names=[self.settings.input],
            output_names=[self.settings.output],
            dynamic_axes={self.settings.input: {0: 'batch'},
                          self.settings.output: {0: 'batch'}} if self.settings.dynamic else None,
            opset_version=self.settings.opset,
        )
        logger.info("generated onnx model named {}".format(self.settings.output_name))

        if not self.settings.no_onnxsim:
            import onnx

            from onnxsim import simplify

            input_shapes = {self.settings.input: list(dummy_input.shape)} if self.settings.dynamic else None

            # use onnxsimplify to reduce reduent model.
            onnx_model = onnx.load(self.settings.output_name)
            model_simp, check = simplify(onnx_model,
                                         dynamic_input_shape=self.settings.dynamic,
                                         input_shapes=input_shapes)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(model_simp, self.settings.output_name)
            logger.info("generated simplified onnx model named {}".format(self.settings.output_name))


if __name__ == '__main__':
    converter = Convertor(
        name="yolox-tiny",
        pth_file="weights/yolox_tiny.pth",
        output_file="yolox_tiny.onnx",
        dynamic=False,
        opset=10
    )
    converter.convert()
