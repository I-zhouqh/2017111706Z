
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
from model.vgg import *
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
      
      'nmb_cluster':100,
      'batches_print':100,
      'epoch': 80,
      'confidence_threshold':0.4,  #达到多少置信度才将这个样本加入计算loss
      'model_path':'./model/deepclustering_with_SSL.pkl',
      'net_type':'vgg',
      'batch_print':200   #每过多少个batch print一次，一共是700个batch。
      }

def My_read_model(path,net_type):
    
    assert net_type in ['vgg','deep_clustered_vgg'],"net_type should be included in either vgg or deep_clustered_vgg"
    
    if net_type=='vgg':
        path='./simple_run_vggDec16_14_45/model_parameter.pkl'

        model=VGG()

        model.load_state_dict(torch.load(path))

        model.to(DEVICE)
        
        return model
    
    elif net_type=='deep_clustered_vgg':
        
        
        # Maybe here we should read the frozen model first, and use the frozen model to create the SSL model.
        # that's to say, we need two parameter paths here. It is inconvenient, but I can not figure out a better way.

        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        frozen_model = VGG(100)
        model=SSL_Model(frozen_model,5,False)
        checkpoint=torch.load(path)
        
        checkpoint = {rename_key(key): val
              for key, val
              in checkpoint.items()}

        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        
        return model

        



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
save_file_name='./VALIDATION_for_all_model'+nowtime

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

# read model


model=My_read_model(args['model_path'],args['net_type'])








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

truth_label=np.array(dataset.targets)   #真实标签
chosen_index,not_chosen_index=GetLabeledIndex(truth_label)  #know label的样本的index

val_sampler = SubsetRandomSampler(not_chosen_index)

print(len(not_chosen_index))

val_loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'],sampler=val_sampler , pin_memory=True)


stats_array=[]

crossentropyloss = nn.CrossEntropyLoss().cuda()
#logsoftmax_func=nn.LogSoftmax(dim=1).cuda()
#softmax_func = nn.Softmax(dim=1).cuda()

    #########Val###########
logger.info("***Now Val***")

val_loss_array=[]   #记录已知label部分的loss
val_acc_array=[]

model.eval()

for i, batch_data in enumerate(val_loader):
    with torch.no_grad():
        data, label = map(lambda x: x.to(DEVICE), batch_data)
        outputs = model(data)
        loss = crossentropyloss(outputs, label)

        #acc = torch.mean(torch.eq(torch.argmax(outputs, dim=1), label))
        acc, _ = accuracy(outputs.data, label, topk=(1, 3))

        val_loss_array.append(loss.item())
        val_acc_array.append(acc.item())


stats={
       'val_loss'  :np.mean(val_loss_array),
       'val:acc'   :np.mean(val_acc_array)}
print(stats)

stats_array.append(stats)

print(stats)

logger.info("Fully Complete!!")

