'''
-*- encoding: UTF-8 -*-
Description      :train model
'''
import os
import torch
import argparse
import random
import numpy as np
from torch.optim import optimizer
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Model
from torch.utils.tensorboard import SummaryWriter
from augmentation.data import transfer, get_k_fold_data, smokeData
from augmentation.loadImg import run
from torch.utils.data.sampler import WeightedRandomSampler
from augmentation.data_rotation import rotation

# use-cuda:
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
use_cuda = torch.cuda.is_available()


def k_fold(k, args):
    run(args.root, args.txt, args.imgs)
    data, labels = transfer(args.imgs)
    # rotation(args.root, args.txt, args.imgs, args.rotationpath)
    # data, labels = transfer(args.rotationpath)

    print("start training:")
    print("k-fold cross validation:")
    print("total "+str(k*args.epoch)+" nums")

    for i in range(k):
        data_train, label_train, data_val, label_val = get_k_fold_data(
            k, i, data, labels)  # 获取k折交叉验证的训练和验证数据

        model = Model(args.model)
        if use_cuda:
            torch.backends.cudnn.benchmark = True
            model.backbone.cuda()
        # 使用多GPU
        # model.backbone = nn.DataParallel(model.backbone)

        writer = SummaryWriter(args.log+'/%d' % i)

        train(i, model, data_train, label_train, data_val, label_val,
              args.batch, args.val_batch, args.epoch, writer)
        print('\n')

        # model.save(args.savepath)


def train(k, model, data_train, label_train, data_val, label_val, batch, val_batch, epoch, writer):

    train_dataset = smokeData(data_train, label_train, True)

    # # 对数量较少的样本进行过采样    样本少的权重大  num_samples是采样总数，一般不超过训练样本总数
    weights = [1/label_train.count(0) if label == 0 else 1/label_train.count(
        1) for index, (data, label) in enumerate(train_dataset)]
    sampler = WeightedRandomSampler(
        weights, num_samples=len(label_train), replacement=True)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch, sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)

    val_dataset = smokeData(data_val, label_val, False)
    val_dataloader = DataLoader(
        val_dataset, batch_size=val_batch, shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.backbone.parameters(),
                           lr=3.5e-04, weight_decay=5e-4, betas=(0.9, 0.999))
    # optimizer = optim.Adam(model.backbone.parameters(),
    #                        lr=0.00001, weight_decay=0, betas=(0.9, 0.999),eps=1e-08)     
    # optimizer=optim.SGD(model.backbone.parameters(),lr=1.5e-04,momentum=0.9)                     
    # 动态调整学习率，当epoch为20时，lr*gamma,当epoch为40时再乘gamma
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 40], gamma=0.1)

    if use_cuda:
        criterion.cuda()

    # train
    for epoch in tqdm(range(epoch)):

        epoch_loss = []
        num_correct = 0
        num_total = 0

        for index, (data, label) in enumerate(train_dataloader):
            input = Variable(data)
            target = Variable(label)

            if use_cuda:
                input = input.cuda()
                target = target.cuda()

            # backwrd
            optimizer.zero_grad()
            score = model.backbone(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                epoch_loss.append(loss.item())
                # pre
                prediction = torch.argmax(score, dim=1)
                num_total += label.size(0)
                num_correct += torch.sum(prediction == target).item()

            del data, label, score, loss, prediction

        train_acc = num_correct / num_total
        val_acc = val(model, val_dataloader)

        scheduler.step()
        lr = scheduler.get_last_lr()

        print("k:{k},epoch:{epoch},train_acc:{train_acc},val_acc:{val_acc}".format(
            k=k,
            epoch=epoch,
            train_acc=train_acc,
            val_acc=val_acc)
        )

        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)
        writer.add_scalar('loss', np.mean(epoch_loss), epoch)
        writer.add_scalar('lr', lr[-1], epoch)

        # 每个epoch保存一次模型
        torch.save(model.backbone.state_dict(), os.path.join(
            'checkpoints/epoch', str(str(epoch)+'.pth')))


def val(model, val_dataloader):
    with torch.no_grad():
        model.backbone.eval()
        num_correct = 0
        num_total = 0

    # correct = 0
        for index, (data, label) in enumerate(val_dataloader):
            val_input = data
            val_label = label
            val_input = Variable(val_input)
            val_label = Variable(val_label)

            if use_cuda:
                val_input = val_input.cuda()
                val_label = val_label.cuda()
            # forward
            score = model.backbone(val_input)
            # pre
            prediction = torch.argmax(score, dim=1)
            num_total += label.size(0)
            num_correct += torch.sum(prediction == val_label).item()

    model.backbone.train()

    return num_correct / num_total


def parse_args():
    parser = argparse.ArgumentParser(description="training arguments")

    parser.add_argument('--env', default='default', help='train nums')

    parser.add_argument('--model', default='resnet50', choices=[
                        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help='model to train')
    parser.add_argument('--savepath', default='./checkpoints/',
                        help='path to save model')
    parser.add_argument('--log',  default='./log',
                        help='dir to save log for display')

    parser.add_argument('--root', default='/datasets/patches_new/')

    parser.add_argument(
        '--imgs', default='image/imgs', help='train dataset')

    parser.add_argument(
        '--txt', default='change/image_index.txt', help='images index')

    parser.add_argument(
        '--rotationpath', default='image/rotation', help='rotation images path')

    parser.add_argument('--batch', type=int, default=256,
                        help='train batchsize')
    parser.add_argument('--val_batch', type=int,
                        default=128, help='val batchsize')

    parser.add_argument('--epoch', type=int, default=50, help='train nums')

    return parser.parse_args()


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    # k折划分
    k_fold(4, parse_args())
