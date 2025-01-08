import sys
import os
import torch
import torch.nn as nn
current_path = os.path.abspath(__file__)
sys_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
sys.path.append(sys_path)
from lib.backbone.Res2Net_v1b import res2net50_v1b_26w_4s

model = res2net50_v1b_26w_4s(pretrained=True)

# for name, layer in model.named_children():
#     print(name, layer)

print(*list(model.children())[:6])

import ipdb; ipdb.set_trace()