from models.resnet_xai_sd import resnet18
import torch
net = resnet18(num_classes=196)
tensor = torch.randn(32, 3, 240, 360)

output = net(tensor)

