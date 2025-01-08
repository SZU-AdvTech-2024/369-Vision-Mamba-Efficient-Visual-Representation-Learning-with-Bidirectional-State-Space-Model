import os
import logging
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn import metrics

import sys
current_path = os.path.abspath(__file__)
sys_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(sys_path)

from config import config
#from lib.module.PNSPlusNetwork import PNSNet
from lib.mamba.mambanet import MambaNet
# from lib.mamba.pure_mambanet import PureMambaNet

### Comparisons ###
# from lib.vmamba.vmambanet import VmambaNet
from lib.mamba.vimnet import VimNet
from lib.g_cascade.networks import PVT_GCASCADE

MODEL_TYPE = "MambaNet"


from lib.dataloader.dataloader import get_video_dataset, VideoDataset
from lib.utils.utils import clip_gradient, adjust_lr
from eval.evaluator import Evaluator
from eval.dice_score import dice_coeff, iou_mean, auto_data_convert

def cofficent_calculate(preds,gts,threshold=0.5):
    eps = 1e-5
    # preds = preds > threshold
    intersection = (preds * gts).sum()
    union =(preds + gts).sum()
    dice = 2 * intersection  / (union + eps)
    iou = intersection/(union - intersection + eps)
    return dice, iou

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    
    def dice_loss(self, input, target):
        input = torch.sigmoid(input)
        target = torch.sigmoid(target)
        N = target.size(0)
        smooth = 1
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) +smooth)
        loss = 1- loss.sum() / N

        return loss
    
    def structure_loss(self, pred, mask):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        # wbce = F.binary_cross_entropy(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(-2, -1)) / weit.sum(dim=(-2, -1))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(-2, -1))
        union = ((pred + mask)*weit).sum(dim=(-2, -1))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()

    def forward(self, *inputs):
        pred, target = tuple(inputs)
        #total_loss = F.binary_cross_entropy(pred.squeeze(), target.squeeze().float())
        total_loss = self.structure_loss(pred.squeeze(), target.squeeze().float()) + self.dice_loss(pred.squeeze(), target.squeeze().float())
        # total_loss = self.structure_loss(pred.squeeze(), target.squeeze().float())
        #total_loss = F.binary_cross_entropy(pred.squeeze(), target.squeeze().float()) + self.dice_loss(pred.squeeze(), target.squeeze().float())
        return total_loss

@torch.no_grad()
def start_eval(eval_loader, model):
    mean_dice = 0.
    ## Evaluation ##
    if config.eval_on:
        logging.info("Start Eval...")
        model = model.eval()
        evaluator = Evaluator(config.metric_list, config.batchsize*(config.video_time_clips+1))
        mean_case_score_list, max_case_score_list = [], []
        # my dice #
        tot_dice = 0.
        tot_dice2 = 0.
        tot_iou = 0.
        tot_mae = 0.
        nums = 0
        for i, (images, gts, _) in enumerate(eval_loader, start=1):
            images = images.cuda(device=device_ids[0])

            if MODEL_TYPE == "MambaNet":
                preds = model(images, mode='eval', sdpm_on=True, udfe_on=True)
                # preds = model(images)
                # preds = torch.sigmoid(preds)
            elif MODEL_TYPE == "PVT_GCASCADE":
                res1, res2, res3, res4 = model(images)
                preds = F.upsample(res1 + res2 + res3 + res4, size=(gts.shape[3], gts.shape[4]), mode='bilinear', align_corners=False)
                preds = torch.sigmoid(preds)
            else:
                preds = model(images)
                preds = torch.sigmoid(preds)
            gts = gts.reshape(-1, gts.shape[3], gts.shape[4])

            # my dice #
            if not config.tf_img_only:
                tot_dice += dice_coeff(preds.squeeze(), gts.to(preds.device), preds.device).item()
                temp_dice, temp_iou = cofficent_calculate(preds.squeeze(), gts.to(preds.device))
                tot_dice2 += temp_dice.item()
                tot_iou += temp_iou.item()
                m = auto_data_convert(preds.squeeze()).astype(int)
                t = auto_data_convert(gts).astype(int)
                tot_mae += metrics.mean_absolute_error(t, m)
            else:
                # TODO: 尺度需一致
                tot_dice += dice_coeff(preds.squeeze(), gts.float().to(preds.device), preds.device).item()
            nums += 1

            preds = preds.squeeze().cpu().detach().numpy()
            gts = gts.squeeze().cpu().numpy()
            ###### keep the first frame and last frame for evaluation ######
            evaluator.eval(preds, gts, config.tf_img_only)
            # eval_step += 1
        
        # my dice #
        # print("Length of eval:", nums)
        mean_dice = tot_dice / nums
        mean_dice2 = tot_dice2 / nums
        mean_iou = tot_iou / nums
        mean_mae = tot_mae / nums
        print("Dice coeff: ", mean_dice, " Dice2: ", mean_dice2)
        logging.info("Dice coeff: " + str(mean_dice) + " Dice2: " + str(mean_dice2))
        print("mIoU: ", mean_iou)
        logging.info("mIoU: " + str(mean_iou))
        print("MAE: ", mean_mae)
        logging.info("MAE: " + str(mean_mae))

        result = evaluator.get_result()
        mean_score_ind, max_score_ind = [], []
        mean_score_list, max_score_list = [], []
        for i, (name, value) in enumerate(result.items()):
            if 'max' in name or 'mean' in name:
                if 'max' in name:
                    max_score_list.append(value)
                    max_score_ind.append(i)
                else:
                    mean_score_list.append(value)
                    mean_score_ind.append(i)
            else:
                mean_score_list.append([value]*256)
                mean_score_ind.append(i)

        # calculate all the metrics at frame-level
        max_case_score_list.append(max_score_list)
        mean_case_score_list.append(mean_score_list)
        max_case_score_list = np.mean(np.array(max_case_score_list), axis=0)
        mean_case_score_list = np.mean(np.array(mean_case_score_list), axis=0)
        case_score_list = []
        for index in range(len(config.metric_list)):
            real_max_index = np.where(np.array(max_score_ind) == index)
            real_mean_index = np.where(np.array(mean_score_ind) == index)
            if len(real_max_index[0]) > 0:
                case_score_list.append(max_case_score_list[real_max_index[0]].max().round(5))
            else:
                case_score_list.append(mean_case_score_list[real_mean_index[0]].mean().round(5))
        final_score_list = ['{:.5f}'.format(case) for case in case_score_list]
        print([config.metric_list[i] + ": " + final_score_list[i] for i in range(len(final_score_list))])
        logging.info([config.metric_list[i] + ": " + final_score_list[i] for i in range(len(final_score_list))])
    
    return mean_dice

def train(train_loader, eval_loader, model, optimizer, epoch, save_path, loss_func, max_Dice):
    global step
    model.cuda(device=device_ids[0]).train()
    loss_all = 0
    epoch_step = 0
    # eval_step = 0

    try:
        # mean_dice = start_eval(eval_loader, model)
        ## Training ##
        for i, (images, gts, _) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
    
            images = images.cuda(device=device_ids[0])
            gts = gts.cuda(device=device_ids[0])
            
            preds = model(images, mask_on=True, sdpm_on=True, udfe_on=True, mode='train')
            # preds = model(images)

            if len(preds) == (gts.shape[0] * gts.shape[1]):
                loss = loss_func(preds.squeeze().contiguous(), gts.contiguous().view(-1, *(gts.shape[2:])))
            else:
                loss = 0.
                for pred in preds:
                    loss += loss_func(pred.squeeze().contiguous(), gts.contiguous().view(-1, *(gts.shape[2:])))
                # loss = loss_func(preds[0].squeeze().contiguous(), gts.contiguous().view(-1, *(gts.shape[2:]))) + \
                #        loss_func(preds[1].squeeze().contiguous(), gts.contiguous().view(-1, *(gts.shape[2:]))) + \
                #        loss_func(preds[2].squeeze().contiguous(), gts.contiguous().view(-1, *(gts.shape[2:])))
                
            loss.backward()

            # clip_gradient(optimizer, config.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 50 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                      format(datetime.now(), epoch, config.epoches, i, total_step, loss.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                    format(epoch, config.epoches, i, total_step, loss.data))
        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, config.epoches, loss_all))
        
        ## Evaluation ##
        if config.eval_on:
            mean_dice = start_eval(eval_loader, model)

            # dice_idx = config.metric_list.index('meanDice')
            # if float(final_score_list[dice_idx]) > max_Dice:
            if float(mean_dice) > max_Dice:
                # max_Dice = float(final_score_list[dice_idx])
                # logging.info("meanDice: " + final_score_list[dice_idx] + ", saving epoch: " + str(epoch))
                max_Dice = float(mean_dice)
                logging.info("meanDice: " + str(mean_dice) + ", saving epoch: " + str(epoch))

                #os.makedirs(os.path.join(save_path, "epoch_%d" % (epoch + 1)), exist_ok=True)
                #save_root = os.path.join(save_path, "epoch_%d" % (epoch + 1))
                torch.save(model.state_dict(), os.path.join(save_path, "ckpt_epoch_%d.pth"%(epoch)))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, "ckpt_epoch_%d.pth"%(epoch)))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + '/Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise
    
    return max_Dice

gpu_id = config.gpu_id
if ',' in gpu_id:
    device_ids = gpu_id.split(',')
    device_ids = [int(idx) for idx in device_ids]
else:
    device_ids = [int(gpu_id)]
device = torch.device('cuda:{}'.format(device_ids[0]) if torch.cuda.is_available() else 'cpu')
print('USE GPU: ', gpu_id)

if __name__ == '__main__':
    
    current_time = time.strftime('%Y-%m%d-%H%M%S', time.localtime())
    save_path = os.path.join(config.save_path, current_time)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # logging
    logging.basicConfig(filename=os.path.join(save_path,'log.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.getLogger(__name__)

    if MODEL_TYPE == "MambaNet":
        model = MambaNet(f_num=config.video_time_clips, img_size=config.size, mlp_ratio=2.0)
    elif MODEL_TYPE == "VmambaNet":
        model = VmambaNet()
    elif MODEL_TYPE == "VimNet":
        model = VimNet()
    elif MODEL_TYPE == "PVT_GCASCADE":
        model = PVT_GCASCADE(img_size=352, k=11, padding=5, conv='mr', gcb_act='gelu', skip_aggregation='additive')

    #model = nn.DataParallel(model, device_ids=device_ids)
    print("model success loaded!")

    cudnn.benchmark = True

    # base_params = [params for name, params in model.named_parameters() if ("temporal_high" in name)]
    # finetune_params = [params for name, params in model.named_parameters() if ("temporal_high" not in name)]
    # optimizer = torch.optim.Adam([
    #     {'params': base_params, 'lr': config.base_lr, 'weight_decay': 1e-4, 'name': "base_params"},
    #     {'params': finetune_params, 'lr': config.finetune_lr, 'weight_decay': 1e-4, 'name': 'finetune_params'}])

    # optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr, betas=config.betas, eps=1e-8, weight_decay=0, amsgrad=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.base_lr, betas=config.betas, eps=1e-8, weight_decay=config.weight_decay, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.finetune_lr,
            last_epoch=-1
        )

    loss_func = CrossEntropyLoss()

    # load data
    print('load data...')
    train_dataset = get_video_dataset(config.dataset, 'train')
    eval_dataset = get_video_dataset(config.evaldataset, 'eval')
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=config.batchsize,
                                   shuffle=True,
                                   num_workers=config.num_workers,
                                   pin_memory=False
                                   )
    eval_loader = data.DataLoader(dataset=eval_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=config.num_workers,
                                    pin_memory=False)

    logging.info('Train on {}'.format(config.dataset))
    logging.info('Eval on {}'.format(config.evaldataset))
    print('Train on {}'.format(config.dataset))
    print('Eval on {}'.format(config.evaldataset))

    total_step = len(train_loader)
    
    logging.info("Network-Train")
    print("Network-Train")
    
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; '
                 'save_path: {}; decay_epoch: {}'.format(config.epoches, config.base_lr, config.batchsize, config.size, config.clip,
                                                         config.decay_rate, config.save_path, config.decay_epoch))
    print('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; '
                 'save_path: {}; decay_epoch: {}'.format(config.epoches, config.base_lr, config.batchsize, config.size, config.clip,
                                                         config.decay_rate, config.save_path, config.decay_epoch))
    step = 0

    print("Start train...")
    max_Dice = 0.
    for epoch in range(config.epoches):
        if epoch == (config.epoches-1):
            import ipdb; ipdb.set_trace()
        # cur_lr = adjust_lr(optimizer, config.base_lr, epoch, config.decay_rate, config.decay_epoch)
        max_Dice = train(train_loader, eval_loader, model, optimizer, epoch, save_path, loss_func, max_Dice)
        scheduler.step()
        logging.info("Current LR:" + str(optimizer.state_dict()['param_groups'][0]['lr']) )
