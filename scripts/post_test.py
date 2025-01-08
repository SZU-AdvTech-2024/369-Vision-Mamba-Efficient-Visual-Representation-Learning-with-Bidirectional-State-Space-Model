import os
import glob
import numpy as np
import logging
from tqdm import tqdm
from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Compose, Resize
import torch.nn.functional as F

import sys
current_path = os.path.abspath(__file__)
sys_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(sys_path)

from config import config
# from lib.module.PNSPlusNetwork import PNSNet as Network
from lib.mamba.mambanet import MambaNet
# from lib.dataloader.dataloader import Test_Dataset
from lib.dataloader.dataloader import get_video_dataset
from eval.evaluator import Evaluator
from eval.dice_score import dice_coeff, iou_mean, auto_data_convert
from sklearn import metrics

def safe_save(img, save_path, gt_path, to_resize=True):
    os.makedirs(save_path.replace(save_path.split('/')[-1], ""), exist_ok=True)
    if to_resize:
        mask = Image.open(gt_path).convert('L')
        img = img.resize(mask.size)
    img.save(save_path)

class HighlightResult:
    def __init__(self, data_root, test_dataset, process_root):
        self.data_root = data_root
        self.test_dataset = test_dataset
        self.process_root = process_root
        self.save_root = os.path.join(process_root, 'highlight')

    def run(self):
        for dst in self.test_dataset:
            current_root = os.path.join(self.process_root, dst)
            gt_root = os.path.join(self.data_root, dst) + '/GT'
            cases = os.listdir(current_root)
            for case in cases:
                current_case = os.path.join(current_root, case)
                gt_case = os.path.join(gt_root, case)
                files = glob.glob(current_case + '/*')
                for file in files:
                    file_name = str(file.split('/')[-1])
                    gt = os.path.join(gt_case, file_name)
                    current_mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                    gt_mask = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)

                    overlay_seg = np.repeat(np.expand_dims(current_mask, axis=2), 3, axis=2) * 0
                    overlay_gt = np.repeat(np.expand_dims(gt_mask, axis=2), 3, axis=2) * 0
                    overlay_seg[current_mask > 128] = (0, 255, 0)   # green
                    overlay_gt[gt_mask > 128] = (255, 0, 0)              # red
                    overlay = cv2.add(overlay_seg, overlay_gt)

                    save_path = file.replace(self.process_root, self.save_root)
                    safe_save(Image.fromarray(overlay.astype(np.uint8)), save_path, gt_path='', to_resize=False)
                    print("Save: ", save_path)
        
        return
        

if __name__ == "__main__":

    process_root = "/media/cgl/Mamba/experiments/2024-0722-163517-pvtb2-cvc"
    
    highlight_ = HighlightResult(config.dataset_root, 
        # ['TestEasyDataset/Seen', 'TestHardDataset/Seen', 'TestEasyDataset/Unseen', 'TestHardDataset/Unseen'],
        ['TestDataset'],
        process_root)

    highlight_.run()