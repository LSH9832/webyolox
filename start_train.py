import os
import sys

setting_file = os.path.abspath('settings.yaml')
hyp_file = os.path.abspath('hyp.yaml')
setting_dir = hyp_file[:-len('hyp.yaml')]
os.makedirs(setting_dir + 'output', exist_ok=True)
os.popen('echo %s > %s' % (str(os.getpid()), setting_dir + 'output/pid'))

this_file_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
running_path = os.path.join(this_file_dir, 'yolox')

os.chdir(running_path)
sys.path.append(running_path)

import os.path
import random
import warnings
import yaml
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

##############################################################################

from yolox.core import launch
from yolox.utils import configure_nccl, configure_omp, get_num_devices
from model import Exp as yoloExp
from train import Trainer


@logger.catch
def main(exp, json_data):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = Trainer(exp, json_data)
    trainer.train()


def get_yaml_data(yaml_file_name):
    assert os.path.exists(yaml_file_name)
    return yaml.load(open(yaml_file_name, encoding='utf8'), yaml.FullLoader)


if __name__ == "__main__":
    this_train_data = get_yaml_data(setting_file)

    num_gpu = this_train_data['gpu_num']
    data_dir = this_train_data['data_dir']
    train_dir = this_train_data['train_dataset_path']
    val_dir = this_train_data['val_dataset_path']
    train_ann = this_train_data['train_annotation_file']
    val_ann = this_train_data['val_annotation_file']

    assert num_gpu <= get_num_devices()

    exp = yoloExp(
        exp_name='yolox-%s' % this_train_data['model_size'],
        max_epoch=this_train_data['epochs'],
        output_dir=this_train_data['output_dir'],
        data_dir=data_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        train_ann=train_ann,
        val_ann=val_ann,
    )

    exp.load_yaml(hyp_file)       # 载入超参数
    # exp.merge(None)

    launch(
        main_func=main,
        num_gpus_per_machine=num_gpu,
        num_machines=1,
        machine_rank=0,
        backend="nccl",
        dist_url="auto",
        args=(exp, this_train_data),
    )
