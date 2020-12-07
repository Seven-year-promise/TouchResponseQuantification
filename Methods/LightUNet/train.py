import os, sys
import sys
sys.path.append('../..')
from Methods.LightUNet.DataLoader import dataset_loader
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import cv2
import util
from Methods.LightUNet.UNet import UNet
import argparse
import torchvision.transforms as transforms
import time
from collections import defaultdict
import torch.nn.functional as F
from Methods.LightUNet.loss import dice_loss

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_path', type=str, default='dataset/train/',
                    help='enter the path for training')
parser.add_argument('--test_path', type=str, default='data//random_2816//samples_for_test.csv',
                    help='enter the path for testing')
parser.add_argument('--eval_path', type=str, default='data//random_2816//samples_for_evaluation.csv',
                    help='enter the path for evaluating')
parser.add_argument('--model_path', type=str, default='models//Liebherr10000checkpoint.pth',
                    help='enter the path for trained model')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='enter the path for training')
parser.add_argument('--batch_size', type=int, default=12,
                    help='enter the batch size for training')
parser.add_argument('--workers', type=int, default=6,
                    help='enter the number of workers for training')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='enter the weight_decay for training')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='enter the momentum for training')
parser.add_argument('--display', type=int, default=2,
                    help='enter the display for training')
parser.add_argument('--max_iter', type=int, default=160000,
                    help='enter the max iterations for training')
parser.add_argument('--test_interval', type=int, default=50,
                    help='enter the test_interval for training')
parser.add_argument('--topk', type=int, default=3,
                    help='enter the topk for training')
parser.add_argument('--start_iters', type=int, default=0,
                    help='enter the start_iters for training')
parser.add_argument('--best_model', type=float, default=12345678.9,
                    help='enter the best_model for training')
parser.add_argument('--lr_policy', type=str, default='multistep',
                    help='enter the lr_policy for training')
parser.add_argument('--policy_parameter', type=dict, default={"stepvalue":[50000, 100000, 120000], "gamma": 0.33},
                    help='enter the policy_parameter for training')
parser.add_argument('--epoch', type=int, default=400,
                    help='enter the path for training')
parser.add_argument('--lamda', type=float, default=0.0,
                    help='enter the path for training')
parser.add_argument('--save_path', type=str, default='models/',
                    help='enter the path for training')



def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_net(model, args):
    img_path = args.train_path + "Images/"
    ann_path = args.train_path + "annotation/"

    stride = 8
    cudnn.benchmark = True

    trans = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
    ])

    train_loader = torch.utils.data.DataLoader( dataset_loader(cropped_size = 240, trans=trans, img_path = img_path, ann_path = ann_path),
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.workers, pin_memory=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.base_lr) #SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.MSELoss().cuda()

    device = (torch.device("cuda:0") if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    print(model)
    iters = 0
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()
    losses = util.AverageMeter()
    losses_list = [util.AverageMeter() for i in range(12)]
    end = time.time()

    heat_weight = 48 * 48 * 25 / 2.0  # for convenient to compare with origin code
    # heat_weight = 1

    while iters < args.max_iter:
        metrics = defaultdict(float)
        epoch_samples = 0
        for i, (input_im, heatmap) in enumerate(train_loader):
            data_time.update(time.time() - end)
            input_var = torch.autograd.Variable(input_im).to(device)
            heatmap_var = torch.autograd.Variable(heatmap).to(device)

            heat = model(input_var)
            #loss = dice_loss(heat, heatmap_var)
            loss = calc_loss(heat, heatmap_var, metrics)
            """

            losses.update(loss.data[0], input.size(0))
            loss_list = [loss]
            for cnt, l in enumerate(loss_list):
                losses_list[cnt].update(l.data[0], input.size(0))
            """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            iters += 1
            if iters % args.display == 0: #
                print('Train Iteration: ', iters, 'Learning rate: ', args.base_lr, 'Loss = ', loss.cpu().data.numpy())

                batch_time.reset()
                data_time.reset()
                losses.reset()

            if iters % 1000 == 0:
                torch.save({ 'iter': iters, 'state_dict': model.state_dict(), },  str(iters) + '.pth.tar')

            if iters == args.max_iter:
                break

if __name__ == '__main__':
    args = parser.parse_args()
    model = UNet(n_class = 2)
    model.double()
    train_net(model, args)
