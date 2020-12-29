
import argparse
import os
import pickle
import time
import logging
from logging import handlers
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pandas as pd
from vgg import *
from torch.utils.data.sampler import  WeightedRandomSampler
from torch.utils.data.sampler import  SubsetRandomSampler


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


args={'seed':4396,
      'batch_size':64,
      'num_workers':0,
      'lr':0.05,'wd':-4,'momentum':0.9,'pca':64,'n_init':5,'lr_classifier':0.05,
      'known_label_size':500,
      'nmb_cluster':100,
      'batches_print':100,
      'frozen_model_path':'./model_save_Dec16_07_05/parameter_epoch_80.pkl',
      'frozen_model_conv':5,
      'epoch': 80,
      'max_alpha':0.3, #权重最大为
      'confidence_threshold':0.4,  #达到多少置信度才将这个样本加入计算loss


      'batch_print':200   #每过多少个batch print一次，一共是700个batch。
      }







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

nowtime=time.strftime("%b%d_%H_%M", time.localtime())   # 'Dec12_16_13'
save_file_name='./simple_run_vgg'+nowtime

# Create output directory if needed
if not os.path.exists(save_file_name):
    os.makedirs(save_file_name)

logger = init_logger(filename=save_file_name+"/diary.log")

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info("All the information will be stored in %s"%save_file_name)
logger.info("The config are: %s"%str(args))



# fix random seeds
torch.manual_seed(args['seed'])
torch.cuda.manual_seed_all(args['seed'])
np.random.seed(args['seed'])

# model



model=VGG()

model.to(DEVICE)


optimizer = torch.optim.SGD(
    filter(lambda x: x.requires_grad, model.parameters()),
    lr=args['lr'],
    momentum=args['momentum'],
    weight_decay=10**args['wd'],
)



# preprocessing of data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
     ])


#  dataset & dataloader
dataset = datasets.CIFAR10(root='./cifar',
                                             train=True,
                                             transform=transform,
                                             download=True)

val_dataset= datasets.CIFAR10(root='./cifar',
                                             train=False,
                                             transform=transform,
                                             download=True)

truth_label=np.array(dataset.targets)   #真实标签
chosen_index,not_chosen_index=GetLabeledIndex(truth_label)  #know label的样本的index

train_sampler = SubsetRandomSampler(chosen_index)


train_loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'],
                                           sampler=train_sampler, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'],
                                          pin_memory=True)


stats_array=[]

crossentropyloss = nn.CrossEntropyLoss().cuda()
#logsoftmax_func=nn.LogSoftmax(dim=1).cuda()
#softmax_func = nn.Softmax(dim=1).cuda()

# training convnet with DeepCluster
for big_epoch_iter in range(1,args['epoch']+1):



    logger.info('************* EPOCH %d/%d ***************************\n'%(big_epoch_iter,args['epoch']) )


    #这一步相当重要，get data loader




    model.train()

    train_loss_array=[]   #记录已知label部分的loss
    train_acc_array=[]

    for step, batch_data in enumerate(train_loader):

        data,label=map(lambda x: x.to(DEVICE), batch_data)
        model.zero_grad() #梯度清零

        outputs= model(data)
        loss= crossentropyloss(outputs, label)

        #acc=torch.mean( torch.eq(torch.argmax(outputs,dim=1),label))
        acc, _ = accuracy(outputs.data, label, topk=(1, 3))

        train_loss_array.append(loss.item())
        train_acc_array.append(acc.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #########Val###########
    logger.info("***Now Val***")

    val_loss_array=[]   #记录已知label部分的loss
    val_acc_array=[]

    model.eval()
    # softmax = nn.Softmax(dim=1).cuda()

    for i, batch_data in enumerate(val_loader):
        with torch.no_grad():
            data, label = map(lambda x: x.to(DEVICE), batch_data)
            outputs = model(data)
            loss = crossentropyloss(outputs, label)

            #acc = torch.mean(torch.eq(torch.argmax(outputs, dim=1), label))
            acc, _ = accuracy(outputs.data, label, topk=(1, 3))

            val_loss_array.append(loss.item())
            val_acc_array.append(acc.item())


    stats={'epoch':big_epoch_iter,
           'train_loss':np.mean(train_loss_array),
           'train_acc' :np.mean(train_acc_array),
           'val_loss'  :np.mean(val_loss_array),
           'val:acc'   :np.mean(val_acc_array)}
    print(stats)

    stats_array.append(stats)





logger.info("Fully Complete!!")

df_stats = pd.DataFrame(stats_array)
logger.info("Now writing stats to excel")
df_stats.to_csv(save_file_name+"/Train_Stats.csv",index=False)


save_model_dir = save_file_name + '/model_parameter.pkl'
torch.save(model.state_dict(), save_model_dir)
# print("Successfully Save model to%s" % save_model_dir)
logger.info("Successfully Save model to%s" % save_model_dir)
