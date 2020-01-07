
###within dataset
#python -u main_resnet18.py --arch myresnet_basic --dataset data --test_id 0 --outdir results/resnet_preact/00 --batch_size 32 --base_lr 0.1 --momentum 0.9 --nesterov True --weight_decay 1e-4 --epochs 40 --milestones '[30, 35]' --lr_decay 0.1 --tensorboard

###cross dataset
#python -u main_resnet18.py --arch myresnet_basic --dataset data_UT_Multiview --testset data  --outdir results/resnet_preact/00 --batch_size 32 --base_lr 0.1 --momentum 0.9 --nesterov True --weight_decay 1e-4 --epochs 40 --milestones '[30, 35]' --lr_decay 0.1 --tensorboard


#!/usr/bin/env python
# coding: utf-8


import os
import time
import json
from collections import OrderedDict
import importlib
import logging
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

from dataloader import get_loader,get_loader_multiview_mpiigaze

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0


def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arch', type=str, required=True, choices=['lenet', 'resnet_preact','zhang','myresnet_basic'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--test_id', type=int)
    parser.add_argument('--testset', type=str)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=7)

    # optimizer
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--milestones', type=str, default='[20, 30]')
    parser.add_argument('--lr_decay', type=float, default=0.1)

    # TensorBoard
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', default=True)
    parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false')
    parser.add_argument('--tensorboard_images', action='store_true')
    parser.add_argument('--tensorboard_parameters', action='store_true')

    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = True
        args.tensorboard_images = True
        args.tensorboard_parameters = False

    assert os.path.exists(args.dataset)
    args.milestones = json.loads(args.milestones)

    return args


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def convert_to_unit_vector(angles):
    #print("arxika:",angles)
    x = -torch.cos(angles[:, 0]) * torch.sin(angles[:, 1])
    y = -torch.sin(angles[:, 0])
    z = -torch.cos(angles[:, 1]) * torch.cos(angles[:, 1])
    #print("meta:(x=",x,",y=",y,",z=",z,")")
    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    #print("telos:(x=",x,",y=",y,",z=",z,")")

    return x, y, z


def compute_angle_error(preds, labels):
    #pred_x, pred_y, pred_z = convert_to_unit_vector(preds)
    #label_x, label_y, label_z = convert_to_unit_vector(labels)
    #angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    #print("teliko error:",angles)
    #print("teliko error se moires:",torch.acos(angles) * 180 / np.pi)
    angles = abs(preds-labels)
    angles = angles[:,0]+angles[:,1]
    return angles *180/np.pi
    #return torch.acos(angles) * 180 / np.pi

def train(epoch, model, optimizer, criterion, train_loader, config, writer):
    global global_step
    
    logger.info('Train {}'.format(epoch))

    model.train()

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    for step, (images, poses, gazes) in enumerate(train_loader):
        #print("edw0")
        global_step += 1


        if config['tensorboard_images'] and step == 0:
            image = torchvision.utils.make_grid(
                images, normalize=True, scale_each=True)
            writer.add_image('Train/Image', image, epoch)#https://tensorboardx.readthedocs.io/en/latest/tensorboard.html#tensorboardX.SummaryWriter.add_image
        #print("edw05")
        #images = images.cuda()
        #poses = poses.cuda()
        #gazes = gazes.cuda()

        optimizer.zero_grad()
        #print("edw1")

        outputs = model(images, poses)

        #print("edw2")
        #print(outputs)

        loss = criterion(outputs, gazes)
        #print("edw3")
        loss.backward()
        #print("edw4")
        optimizer.step()

        #absolute errors(vert+hor)
        angle_error = compute_angle_error(outputs, gazes).mean()
        #print("edw5")
        num = images.size(0)
        #print("edw51")
        loss_meter.update(loss.item(), num)
        #print("edw52")
        angle_error_meter.update(angle_error.item(), num)
        #print("edw53")
        if config['tensorboard']:
            #print("errors:",loss_meter.val,",step:",global_step)
            writer.add_scalar('Train/RunningLoss', loss_meter.val, global_step)
            # errors: 35.89125061035156 ,step: 2
            # errors: 13759.7177734375 ,step: 3
            # errors: 443536992.0 ,step: 4
            # errors: 3.3213777035879766e+22 ,step: 5
            # errors: inf ,step: 6
            # Warning: NaN or Inf found in input tensor.

            #https://tensorboardx.readthedocs.io/en/latest/tensorboard.html#tensorboardX.SummaryWriter.add_scalar
        #print("edw6")
        if step % 100 == 0:
            #print("edw7")
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'AngleError {:.2f} ({:.2f})'.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss_meter.val,
                            loss_meter.avg,
                            angle_error_meter.val,
                            angle_error_meter.avg,
                        ))
            #print("edw8")
    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    #print("edw9")
    if config['tensorboard']:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/AngleError', angle_error_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)
        #https://tensorboardx.readthedocs.io/en/latest/tensorboard.html#tensorboardX.SummaryWriter.add_scalar
    #print("edw10")


def test(epoch, model, criterion, test_loader, config, writer):
    logger.info('Test {}'.format(epoch))

    model.eval()

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()

    std_errors=[]
    for step, (images, poses, gazes) in enumerate(test_loader):
        if config['tensorboard_images'] and epoch == 0 and step == 0:
            image = torchvision.utils.make_grid(
                images, normalize=True, scale_each=True)
            writer.add_image('Test/Image', image, epoch)

        #images = images.cuda()
        #poses = poses.cuda()
        #gazes = gazes.cuda()

        with torch.no_grad():
            outputs = model(images, poses)
            
        loss = criterion(outputs, gazes)

        mult_errors=compute_angle_error(outputs, gazes)
        angle_error = mult_errors.mean()
        std_errors=std_errors+mult_errors.tolist()

        #find new average:
        num = images.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

    std_errors=np.array(std_errors)
    #mean()
  
    #print("average:",angle_error_meter.avg)
    #print("my_avg:",abs(std_errors).mean())
    #print("my_stdev:",abs(std_errors).std())

   
    std=abs(std_errors).std()
    logger.info('Epoch {} Loss {:.4f} AngleError {:.2f} AngleError(std) {:.2f}'.format(
        epoch, loss_meter.avg, angle_error_meter.avg,std))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if config['tensorboard']:
        if epoch > 0:
            writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
            writer.add_scalar('Test/AngleError', angle_error_meter.avg, epoch),
            writer.add_scalar('Test/AngleError(std)', abs(std_errors).std(), epoch)
        writer.add_scalar('Test/Time', elapsed, epoch)

    if config['tensorboard_parameters']:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step)
            #https://tensorboardx.readthedocs.io/en/latest/tensorboard.html#tensorboardX.SummaryWriter.add_histogram
            

    return (angle_error_meter.avg,std)


def main():
    args = parse_args()
    logger.info(json.dumps(vars(args), indent=2))

    # TensorBoard SummaryWriter:https://tensorboardx.readthedocs.io/en/latest/tensorboard.html
    writer = SummaryWriter() if args.tensorboard else None

    # set random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create output directory
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = os.path.join(outdir, 'config.json')
    with open(outpath, 'w') as fout:
        json.dump(vars(args), fout, indent=2)

    # data loaders
    # train_loader, test_loader = get_loader(
    #     args.dataset, args.test_id, args.batch_size, args.num_workers, True)
    train_loader, test_loader = get_loader_multiview_mpiigaze(
        args.dataset, args.testset, args.batch_size, args.num_workers, True)


    # model
    module = importlib.import_module('models.{}'.format(args.arch))
    model = module.Model()
    #print(model)
    #from torchsummary import summary
    #img_n = (1,1,60,36)
    #pose_n = (1,2)
    #print(summary(model,(img_n,pose_n)))
    #model.cuda()

    criterion = nn.MSELoss(size_average=True)

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.lr_decay)

    config = {
        'tensorboard': args.tensorboard,
        'tensorboard_images': args.tensorboard_images,
        'tensorboard_parameters': args.tensorboard_parameters,
    }

    # run test before start trainings
    #test(0, model, criterion, test_loader, config, writer)

    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        train(epoch, model, optimizer, criterion, train_loader, config, writer)
        (angle_error,std) = test(epoch, model, criterion, test_loader, config,
                           writer)

        state = OrderedDict([
            ('args', vars(args)),
            ('state_dict', model.state_dict()),
            ('optimizer', optimizer.state_dict()),
            ('epoch', epoch),
            ('angle_error', angle_error),
            ('angle_error_std', std),
        ])
        model_path = os.path.join(outdir, 'model_state.pth')
        torch.save(state, model_path)

    if args.tensorboard:
        outpath = os.path.join(outdir, 'all_scalars.json')
        writer.export_scalars_to_json(outpath)
        #https://tensorboardx.readthedocs.io/en/latest/tensorboard.html#tensorboardX.SummaryWriter.export_scalars_to_json
    writer.close()

if __name__ == '__main__':
    main()
