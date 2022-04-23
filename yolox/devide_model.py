import torch
from yolox.exp import get_exp

print("loading ckpt")
model_size = "m"
ckpt = torch.load("./yolox_%s.pth" % model_size)

print("get model")
exp = get_exp(None, "yolox-%s" % model_size)
model = exp.get_model()

print("load state dict")
model.load_state_dict(ckpt["model"])

print("save state dict")
torch.save({"backbone": model.backbone.state_dict()}, "./%s_backbone.pth" % model_size)
torch.save({"head": model.head.state_dict()}, "./%s_head.pth" % model_size)

print("ok")
