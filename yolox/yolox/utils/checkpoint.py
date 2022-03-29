import os
import shutil
from loguru import logger

import torch


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best, save_dir, model_name="", epoch: int or None = None, is_devided=False):
    if is_devided:
        filename_backbone = os.path.join(save_dir, "backbone_%s.pth" % ("best" if is_best else "last"))
        filename_head = os.path.join(save_dir, "head_%s.pth" % ("best" if is_best else "last"))
        torch.save(state["backbone"], filename_backbone)
        torch.save(state["head"], filename_head)
    else:
        if is_best:
            filename = os.path.join(
                save_dir,
                "best%s.pth" % ('_epoch_%s' % epoch
                                if epoch is not None
                                else '')
            )
        else:
            if epoch is not None and isinstance(epoch, int):
                last_model_name = '%s_%04d.pth' % (model_name, epoch)
            else:
                last_model_name = 'last.pth'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            filename = os.path.join(save_dir, last_model_name)

        torch.save(state, filename)

    # if epoch is not None and isinstance(epoch, int):
    #     last_model_name = '%s_%04d.pth' % (model_name, epoch)
    # else:
    #     last_model_name = '%s_last.pth' % model_name
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # filename = os.path.join(save_dir, last_model_name)
    # torch.save(state, filename)
    # if is_best:
    #     best_filename = os.path.join(save_dir, model_name + "_best.pth")
    #     shutil.copyfile(filename, best_filename)
