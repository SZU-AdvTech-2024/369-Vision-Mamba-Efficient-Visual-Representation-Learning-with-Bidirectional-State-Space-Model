import argparse


parser = argparse.ArgumentParser()

# optimizer
parser.add_argument('--gpu_id', type=str, default='2', help='train use gpu')
parser.add_argument('--lr_mode', type=str, default="poly")
parser.add_argument('--base_lr', type=float, default=2e-4)
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--T_max', type=int, default=60, help='epoch nums before reaching min_lr')
parser.add_argument('--finetune_lr', type=float, default=5e-5)
parser.add_argument('--decay_rate', type=float, default=0.8, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')

# train schedule
parser.add_argument('--epoches', type=int, default=100)

# data
parser.add_argument('--data_statistics', type=str,
                    default="lib/dataloader/statistics.pth", help='The normalization statistics.')
parser.add_argument('--dataset', type=str,
                    default="TrainDataset2", help="TrainDataset / 2")
parser.add_argument('--evaldataset', type=str,
                    default="TestDataset", help="TestHardDataset/Unseen , TestDataset")
parser.add_argument('--dataset_root', type=str,
                    default="/media/cgl/ClinicDB/", help="/media/cgl/SUN-SEG/ , /media/cgl/ClinicDB/")
parser.add_argument('--size', type=tuple,
                    default=(352, 352))
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--video_time_clips', type=int, default=1)

parser.add_argument('--save_path', type=str, default='/media/cgl/Mamba/experiments/')

# eval
parser.add_argument('--eval_on', type=bool, default=True)
parser.add_argument('--tf_img_only', type=bool, default=False)
parser.add_argument(
    '--metric_list', type=list, help='set the evaluation metrics',
    default=['Smeasure', 'meanEm', 'wFmeasure', 'MAE'],
    choices=["Smeasure", "wFmeasure", "MAE", "adpEm", "meanEm", "maxEm", "adpFm", "meanFm", "maxFm",
                "meanSen", "maxSen", "meanSpe", "maxSpe", "meanDice", "maxDice", "meanIoU", "maxIoU"])

config = parser.parse_args()
