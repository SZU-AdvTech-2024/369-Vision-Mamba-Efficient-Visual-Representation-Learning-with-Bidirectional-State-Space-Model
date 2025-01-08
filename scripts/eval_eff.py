import os
import time
import numpy as np
from ptflops import get_model_complexity_info
# from thop import profile
import re

import sys
current_path = os.path.abspath(__file__)
sys_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(sys_path)

import torch

# from lib.module.PNSPlusNetwork import PNSNet as Network
from lib.mamba.mambanet import MambaNet


def computeTime(model, inputs, device='cuda'):
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    time_spent = []
    for idx in range(100):
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs, mode='eval', sdpm_on=True, udfe_on=True)
            # _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if idx > 10:
            time_spent.append(time.time() - start_time)
    print('Avg execution time (ms): %.4f, FPS:%d'%(np.mean(time_spent),1*1//np.mean(time_spent)))
    return 1*1//np.mean(time_spent)


if __name__=="__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # device = torch.device('cuda:{}'.format(1) if torch.cuda.is_available() else 'cpu')

    torch.backends.cudnn.benchmark = True
    
    # To load model
    model = MambaNet(f_num=5, img_size=(352, 352), mlp_ratio=2.0).cuda()
    # model = Network().cuda()
    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (6, 3, 352, 352), as_strings=True,
                                                 print_per_layer_stat=False, verbose=True)

    inputs = torch.randn(1, 6, 3, 352, 352)

    print(str(params) + '\t' + str(macs))
    computeTime(model, inputs)

    # print(' ')
    # flops, params = profile(model, torch.randn(1, 1, 6, 3, 256, 512).cuda())
    # print('flops: ', str(flops / 10**9))