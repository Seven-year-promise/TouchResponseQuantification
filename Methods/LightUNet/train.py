import os, sys
from DataLoader import dataset_loader
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import cv2
import util
import UNet.UNet
import argparse
import torchvision.transforms as transforms
import time

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_path', type=str, default='"dataset/',
                    help='enter the path for training')
parser.add_argument('--test_path', type=str, default='data//random_2816//samples_for_test.csv',
                    help='enter the path for testing')
parser.add_argument('--eval_path', type=str, default='data//random_2816//samples_for_evaluation.csv',
                    help='enter the path for evaluating')
parser.add_argument('--model_path', type=str, default='models//Liebherr10000checkpoint.pth',
                    help='enter the path for trained model')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='enter the path for training')
parser.add_argument('--batchsize', type=int, default=32,
                    help='enter the batchsize for training')
parser.add_argument('--workers', type=int, default=6,
                    help='enter the number of workers for training')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='enter the weight_decay for training')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='enter the momentum for training')
parser.add_argument('--display', type=int, default=50,
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

def train_net(model, args):
    img_path = args.train_path + "Images/"
    anno_path = args.train_path + "annotation/"

    stride = 8
    cudnn.benchmark = True
    train_loader = torch.utils.data.DataLoader( dataset_loader(cropped_size = 220, img_path = img_path, anno_path = anno_path),
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.workers, pin_memory=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.MSELoss().cuda()

    model.train()
    iters = 0
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()
    losses = util.AverageMeter()
    losses_list = [util.AverageMeter() for i in range(12)]
    end = time.time()

    heat_weight = 48 * 48 * 25 / 2.0  # for convenient to compare with origin code
    # heat_weight = 1

    while iters < args.max_iter:
        for i, (input, heatmap) in enumerate(train_loader):
            learning_rate = util.adjust_learning_rate(optimizer, iters, args.base_lr, policy=args.lr_policy,\
								policy_parameter=args.policy_parameter, multiple=args.multiple)
            data_time.update(time.time() - end)

            input = input.cuda()
            heatmap = heatmap.cuda()
            input_var = torch.autograd.Variable(input)
            heatmap_var = torch.autograd.Variable(heatmap)

            heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var)
            loss1 = criterion(heat1, heatmap_var) * heat_weight
            loss2 = criterion(heat2, heatmap_var) * heat_weight
            loss3 = criterion(heat3, heatmap_var) * heat_weight
            loss4 = criterion(heat4, heatmap_var) * heat_weight
            loss5 = criterion(heat5, heatmap_var) * heat_weight
            loss6 = criterion(heat6, heatmap_var) * heat_weight
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            losses.update(loss.data[0], input.size(0))
            loss_list = [loss1 , loss2 , loss3 , loss4 , loss5 , loss6]
            for cnt, l in enumerate(loss_list):
                losses_list[cnt].update(l.data[0], input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()


            iters += 1
            if iters % args.display == 0:
                print('Train Iteration: {0}\t'
				      'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
				      'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
				      'Learning rate = {2}\n'
				      'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
					iters, args.display, learning_rate, batch_time=batch_time,
					data_time=data_time, loss=losses))
                for cnt in range(0, 6):
                    print('Loss{0}_1 = {loss1.val:.8f} (ave = {loss1.avg:.8f})'.format(cnt + 1,loss1=losses_list[cnt]))
                print(time.strftime(
					'%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n',
					time.localtime()))

                batch_time.reset()
                data_time.reset()
                losses.reset()
                for cnt in range(12):
                    losses_list[cnt].reset()

            if iters % 5000 == 0:
                torch.save({
					'iter': iters,
					'state_dict': model.state_dict(),
				},  str(iters) + '.pth.tar')

            if iters == args.max_iter:
                break

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    args = parser.parse_args()
    model = UNet(args)
    train_net(model, args)
