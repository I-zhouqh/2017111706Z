
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from vggnet import *
from utils import *
from torch.utils.data.sampler import SubsetRandomSampler
import logging
from logging import handlers
import pandas as pd
from sklearn.decomposition import PCA


args={'seed':4396,'verbose':False,'nmb_cluster':100,'batch_size':64,'num_workers':0,'epochs':35,
      'small_epoch':1,'wd':-4,'momentum':0.9,'pca':64,'n_init':5,
      'known_label_size':500,'eval_size':10000
    #lr':0.05,
    #'lr_classifier':0.05,
    #'model_dir':'./model_save0/parameter_epoch_40.pkl',
    # 'conv':2,
}

# grid search
grid_search_parameters = []

model_list = ['./model_save_Dec15_04_42/parameter_epoch_20.pkl',
              './model_save_Dec15_04_42/parameter_epoch_40.pkl',
              './model_save_Dec15_04_42/parameter_epoch_60.pkl',
              './model_save_Dec16_04_18/parameter_epoch_20.pkl',
              './model_save_Dec16_04_18/parameter_epoch_40.pkl',
              './model_save_Dec16_04_18/parameter_epoch_60.pkl',
              './model_save_Dec16_07_05/parameter_epoch_40.pkl',
              './model_save_Dec16_07_05/parameter_epoch_80.pkl',
              './model_save_Dec16_07_05/parameter_epoch_120.pkl']

conv_list = [3, 5, 7, 9, 11]
lr_list = [0.01]

for c in conv_list:
    for l in lr_list:
        for m in model_list:
            d = {}
            d['model_dir'] = m
            d['conv'] = c
            d['lr'] = l
            d['best_val_acc'] = 0
            d['best_val_epoch'] = 0
            grid_search_parameters.append(d)

grid_search_parameters.append({'model_dir': 'No pretrain',
                               'conv': 0,
                               'lr': 0.01,
                               'best_val_acc': 0,
                               'best_val_epoch': 0})  # 加上一个无模型的纯分类层




nowtime=time.strftime("%b%d_%H_%M", time.localtime())   # 'Dec12_16_13'
save_file_name='./linear_choose_'+nowtime

# Create output directory if needed
if not os.path.exists(save_file_name):
    os.makedirs(save_file_name)


DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def forward(x, model, conv):
    model.eval()
    #Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    Avgpool = nn.AvgPool2d(kernel_size=2, stride=2)


    count = 1
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)
        if isinstance(m, nn.ReLU):
            if count == conv:
                break
            count = count + 1

    if conv<11:  #如果层数太高了就不再去pool了，会导致x的size太小
        x=Avgpool(x)

    x=x.view(x.size(0),-1)

    return x

def Get_final_feature_dim(loader,model,conv):

    #Maxpool=nn.MaxPool2d(kernel_size=2, stride=2)
    Avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    model.eval()
    for data in loader:
        input,label=data

        input = torch.autograd.Variable(input.cuda())

        count = 1
        for m in model.features.modules():
            if not isinstance(m, nn.Sequential):
                input = m(input)
            if isinstance(m, nn.ReLU):
                if count == conv:
                    break
                count = count + 1

        if conv<11:
            input=Avgpool(input)
        input=input.view(input.size(0),-1)

        return input.size()[1]

def GetLabeledIndex(truth_label):  #有一些样本是know label的，得到他们的序号

    # 抽出的样本必须和之前用的是一样的，所以set一样的seed

    np.random.seed(seed=args['seed'])
    known_label_size = args['known_label_size']  # 有多少个已知标签
    assert known_label_size % 10 == 0, "known_label_size must be 10的倍数"

    chosen_index = []
    not_chosen_index=[]

    for classed_label in range(10):
        this_class_index = np.flatnonzero(truth_label == classed_label)
        tmp = np.random.choice(this_class_index, int(known_label_size / 10), replace=False)
        chosen_index.extend(tmp.tolist())
        #chosen_index[classed_label] = tmp

    for i in range(len(truth_label)):
        if i not in chosen_index:
            not_chosen_index.append(i)

    return chosen_index,not_chosen_index #是一个list

def My_read_model(dir):
    checkpoint = torch.load(dir)

    def rename_key(key):
        if not 'module' in key:
            return key
        return ''.join(key.split('.module'))

    checkpoint = {rename_key(key): val
                  for key, val
                  in checkpoint.items()}

    model = vgg16(out=args['nmb_cluster'])
    model.load_state_dict(checkpoint)
    model.cuda()
    return model

def init_logger(filename, when='D', backCount=3,
                fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    logger = logging.getLogger(filename)
    format_str = logging.Formatter(fmt)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)

    return logger


logger = init_logger(filename=save_file_name+"/grid_search_diary.log")


logger.info("All the information will be stored in %s"%save_file_name)
logger.info("The config are: %s"%str(args))
logger.info("The grid search parameters are:%s"%str(grid_search_parameters))



def main():

    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    np.random.seed(args['seed'])

    ### load data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
         ])

    dataset = datasets.CIFAR10(root='./cifar',
                               train=True,
                               transform=transform,
                               download=True)

    truth_label = np.array(dataset.targets)  # 真实标签
    known_label_index, not_known_label_index = GetLabeledIndex(truth_label)  # know label的样本的index

    train_sampler = SubsetRandomSampler(known_label_index)

    tmp=np.random.choice(not_known_label_index,args['eval_size'],replace=False)  #抽一波，快一点
    not_known_label_index=tmp.tolist()

    valid_sampler = SubsetRandomSampler(not_known_label_index)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'],
                                               sampler=train_sampler, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'],
                                             sampler=valid_sampler, pin_memory=True)

    logger.info("len of train_loader: %s; len of val_loader: %s   "%( str(len(train_loader)), str(len(val_loader))  ))



    for search_step,grid_search_param in enumerate(grid_search_parameters): #网格搜索

        logger.info('************* search step %d/%d ***************************\n' % (search_step+1, len(grid_search_parameters) ))
        logger.info('The Now grid search parameter is %s'%str(grid_search_param))


        ### load model


        if grid_search_param['model_dir']!='No pretrain':
            path=grid_search_param['model_dir']
            model = My_read_model(path)
            cudnn.benchmark = True
            for param in model.features.parameters():
                param.requires_grad = False

        else: #如果是no pretrain，也就是直接全连接，这时候借用一个model来读取层数，并不用这个model来传播。
            model=None




        # freeze the features layers


        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()

        #得到特征层最后一层的维度
        if model is not None:
            feature_dim=Get_final_feature_dim(train_loader,model,grid_search_param['conv'])
        else:
            feature_dim=3072   #如果没有模型，直接把图展开（3072维度）做全连接

        # 特征层后面只接一个分类网络
        reglog = RegLog(feature_dim, len(dataset.classes)).cuda()

        optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, reglog.parameters()),
            grid_search_param['lr'],
            momentum=args['momentum'],
            weight_decay=10 ** args['wd']
        )

        num_of_prec_100=0  #记录下训练集达到100准确率的次数，达到3次就break

        for epoch in range(1,args['epochs']+1):

            logger.info('******** EPOCH %d/%d ********\n' % (epoch, args['epochs']))

            #########Training###########

            logger.info("**Now Training**")

            if model is not None:
                model.eval()   # freeze all BN

            for i, data in enumerate(train_loader):

                if model is not None:  #
                    input_tensor, target = map(lambda x: x.to(DEVICE), data)
                    output = forward(input_tensor, model, grid_search_param['conv'])
                else:  #如果模型是没有预训练，直接pca提取到指定维度，然后就全连接
                    input,target=map(lambda x: x.to(DEVICE), data)
                    input=input.view(input.size()[0],-1)  # batch * 3072维度
                    output=input


                output = reglog(output)
                loss = criterion(output, target)
                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.info("prec1: %.2f,prec5: %.2f,loss %.2f" % (prec1.item(), prec5.item(), loss.item()))  #只打出最后一个batch的好了
            logger.info('\n')

            if prec1.item()>99.9:
                num_of_prec_100+=1
            if num_of_prec_100>4:
                break

            #########Val###########
            logger.info("***Now Val***")

            top1_array = []
            top3_array = []
            loss_array = []

            # switch to evaluate mode
            if model is not None:
                model.eval()
            #softmax = nn.Softmax(dim=1).cuda()
            for i, data in enumerate(val_loader):
                with torch.no_grad():

                    if grid_search_param['model_dir'] != 'No pretrain':
                        input_tensor, target = map(lambda x: x.to(DEVICE), data)
                        output = forward(input_tensor, model, grid_search_param['conv'])
                    else:  # 如果模型是没有预训练，直接pca提取到指定维度，然后就全连接
                        input, target = map(lambda x: x.to(DEVICE), data)
                        input = input.view(input.size()[0], -1)  # batch * 3072维度
                        output = input

                    output = reglog(output)
                    prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
                    loss = criterion(output, target)

                    top1_array.append(prec1.item())
                    top3_array.append(prec3.item())
                    loss_array.append(loss.item())

            # The mean loss across batch
            prec1, prec3, loss = np.mean(top1_array), np.mean(top3_array), np.mean(loss_array)

            if grid_search_param['best_val_acc']<prec1:
                grid_search_param['best_val_acc']=prec1
                grid_search_param['best_val_epoch']=epoch

            logger.info("The val result of this epoch : prec1: %.2f,prec5: %.2f,loss %.2f"%(prec1.item(),prec3.item(),loss.item()))
            logger.info('\n\n')

        logger.info("In this grid search parameter, the best val is %.3f in epoch %d "
                    %(grid_search_param['best_val_acc'],grid_search_param['best_val_epoch'])  )
        logger.info('\n\n')


    logger.info("Fully Complete!!")
    #logger.info("The final stats of grid search is :",str(grid_search_parameters))
    logger.info("Writing stats to excel Successfully!!")
    pd.DataFrame(grid_search_parameters).to_csv(save_file_name+'/grid_search_result.csv',index=False)




class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, feature_dim,num_labels):
        super(RegLog, self).__init__()
        self.linear = nn.Linear(feature_dim, num_labels)

    def forward(self, x):
        return self.linear(x)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()



#
# def train(train_loader, model, reglog, criterion, optimizer, epoch):
#
#     def learning_rate_decay(optimizer, t, lr_0):
#         for param_group in optimizer.param_groups:
#             lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
#             param_group['lr'] = lr
#
#     # freeze also batch norm layers
#     model.train()
#
#     for i, (input, target) in enumerate(train_loader):
#
#
#         # adjust learning rate
#         #learning_rate_decay(optimizer, len(train_loader) * epoch + i, args['lr'])
#
#         target = target.cuda(non_blocking=True)
#         input_var = torch.autograd.Variable(input.cuda())
#         target_var = torch.autograd.Variable(target)
#         # compute output
#
#         output = forward(input_var, model, args['conv'])
#
#         output = reglog(output)
#         loss = criterion(output, target_var)
#         # measure accuracy and record loss
#         prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         print("prec1: %.2f,prec5: %.2f,loss %.2f"%(prec1.item(),prec5.item(),loss.item()))
#
#
#
#
#
# def validate(val_loader, model, reglog, criterion):
#
#     top1_array=[]
#     top3_array=[]
#     loss_array=[]
#
#     # switch to evaluate mode
#     model.eval()
#     softmax = nn.Softmax(dim=1).cuda()
#     end = time.time()
#     for i, (input_tensor, target) in enumerate(val_loader):
#
#         with torch.no_grad():
#
#
#             target = target.cuda(non_blocking=True)
#             input_var = torch.autograd.Variable(input_tensor.cuda())
#             target_var = torch.autograd.Variable(target)
#
#             output = reglog(forward(input_var, model, args['conv']))
#
#
#             output_central = output
#
#             prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
#
#             loss = criterion(output_central, target_var)
#
#
#             top1_array.append(prec1.item())
#             top3_array.append(prec3.item())
#             loss_array.append(loss.item())
#
#
#
#         # if args.verbose and i % 100 == 0:
#         #     print('Validation: [{0}/{1}]\t'
#         #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#         #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#         #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#         #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
#         #           .format(i, len(val_loader), batch_time=batch_time,
#         #                   loss=losses, top1=top1, top5=top5))
#
#     return np.mean(top1_array), np.mean(top3_array), np.mean(loss_array)



