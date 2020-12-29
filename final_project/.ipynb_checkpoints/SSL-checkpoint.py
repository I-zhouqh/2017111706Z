
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
from utils import *
from sklearn.cluster import KMeans
from model.vgg import *
import pandas as pd
from SSL_model_and_data import *
from torch.utils.data.sampler import  WeightedRandomSampler


args={'seed':4396,
      'batch':64,
      'num_workers':0,
      'lr':0.001,'wd':-4,'momentum':0.9,'pca':64,'n_init':5,'lr_classifier':0.05,
      'known_label_size':500,
      'nmb_cluster':100,
      'batches_print':100,
      'frozen_model_path':'./model_save_Dec16_07_05/parameter_epoch_80.pkl',
      'frozen_model_conv':5,
      'T1':10,  # 假标签部分何时开始有权重
      'T2':60, # 权重最大到此时为止
      'epoch': 100,
      'max_alpha':0.6, #权重最大为
      'confidence_threshold':0.4,  #达到多少置信度才将这个样本加入计算loss
      'use_additional_features':False,  #是否要在冻住的特征层之外，再加一个特征层。


      'batch_print':200   #每过多少个batch print一次，一共是700个batch。
      }





def GetLabeledIndex(truth_label):  #有一些样本是know label的，得到他们的序号

    # 抽出的样本必须和之前用的是一样的，所以set一样的seed

    np.random.seed(seed=args['seed'])
    known_label_size = args['known_label_size']  # 有多少个已知标签
    assert known_label_size % 10 == 0, "known_label_size must be 10的倍数"

    chosen_index = []

    for classed_label in range(10):
        this_class_index = np.flatnonzero(truth_label == classed_label)
        tmp = np.random.choice(this_class_index, int(known_label_size / 10), replace=False)
        chosen_index.extend(tmp.tolist())
        #chosen_index[classed_label] = tmp

    return chosen_index #是一个list

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

def get_unlabeled_weight(now_epoch):
    alpha = 0.0
    T1=args['T1']
    T2=args['T2']
    max_alpha=args['max_alpha']
    if now_epoch > T1:
        alpha = (now_epoch - T1) / (T2 - T1) * max_alpha
        if now_epoch > T2:
            alpha = max_alpha
    return alpha

# def cal_weight_confidence(prob):
#     a=args['confidence_threshold']
#
#     result = np.ones(len(prob))
#     y=torch.tensor(result, device=torch.device('cuda'), dtype=torch.float)
#
#     #y=nn.Parameter( torch.ones(len(prob),dtype=torch.float) )
#     y[prob<a]=0  #小于置信限的置为0，其他置为1
#
#     # y=-(prob-a)*(prob+a-2)/(a-1)**2
#     # y[y<0]=0
#
#     #a = np.ones(3)
#     #t3 = torch.tensor(a, device=torch.device('cuda'), dtype=torch.float)
#
#     y.require_grad=False
#     #y.to(DEVICE)
#
#     return y

# def get_sample_weight(now_epoch):
#
#     if now_epoch<=args['T1']:
#         return 10000
#     else:
#         return 10000 * args['epoch'] / (990 * (now_epoch-args['T1']) + 10 * args['epoch'])
#     #用双曲线去拟合这个模式，epoch小的时候weight很大，epoch大的时候weight小一些。


def get_DATASET_and_LOADER(now_epoch):

    def get_num_of_known_label(now_epoch):
        result=None
        if now_epoch<=args['T1']:
            result=0
        elif now_epoch<args['T2']:
            T1=args['T1']
            T2=args['T2']

            b=np.log(100)/(T2-T1)
            a=1/np.exp(b*T1)

            result=50*a*np.exp(b*now_epoch)
        else:
            result=5000
        return int(result)
        #return 10000 * args['epoch'] / (990 * (now_epoch - args['T1']) + 10 * args['epoch'])
        # 用双曲线去拟合这个模式，epoch小的时候weight很大，epoch大的时候weight小一些。
        # 按照这个设置，epoch达到T1的时候 weight为1000，此时在一个batch中，known比unknown为10:1，
        # 到最后一个epoch的时候，weight为10，此时在一个batch中，known比unknown为1：:1


    num_of_unknown_label=get_num_of_known_label(now_epoch) #比如刚开始到T1，我需要50个unknown label，比例就是10:1，到最后我需要是5000个，比例就是1:10
    num_of_unknown_label=500
    SSL_dataset = SSL_DATASET(dataset, known_label_index, num_of_unknown_label=num_of_unknown_label)


    dataloader = torch.utils.data.DataLoader(SSL_dataset,
                                             batch_size=args['batch'],
                                             pin_memory=True,
                                             shuffle=True
                                             )
    return dataloader


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
save_file_name='./SSL_training'+nowtime

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

frozen_model=My_read_model(args['frozen_model_path'])
frozen_conv=args['frozen_model_conv']
model=SSL_Model(frozen_model,frozen_conv,use_additional_features=args['use_additional_features'])

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
truth_label=np.array(dataset.targets)   #真实标签
known_label_index=GetLabeledIndex(truth_label)  #know label的样本的index





stats_array=[]

crossentropyloss = nn.CrossEntropyLoss(reduction='none').cuda()
logsoftmax_func=nn.LogSoftmax(dim=1).cuda()
softmax_func = nn.Softmax(dim=1).cuda()



# training convnet with DeepCluster
for big_epoch_iter in range(1,args['epoch']+1):

    # if big_epoch_iter>13:
    #     break

    logger.info('************* EPOCH %d/%d ***************************\n'%(big_epoch_iter,args['epoch']) )



    #这一步相当重要，get data loader
    dataloader=get_DATASET_and_LOADER(big_epoch_iter)
    print("length of dataloader %d"%len(dataloader))
    #


    model.train()

    loss_known_array=[]   #记录已知label部分的loss
    loss_unknown_array=[]  #位置label部分的loss
    total_loss_array=[]  #总loss，等于 loss_known+alpha*loss_unknown
    known_acc_array=[]   #已知label的acc
    unknown_acc_array=[]  #未知label的 acc
    num_of_known_label_array=[] #一个batch里面有多少个已知label
    prob_for_pseudo_label_array=[]   #那些伪标签，他们的最大类的概率是多少

    for step, batch_data in enumerate(dataloader):



        batch_num=len(dataloader)
        data,label,is_known=map(lambda x: x.to(DEVICE), batch_data)
        model.zero_grad() #梯度清零

        outputs= model(data)
        #print("outpus",outputs[0:3,:])
        #print("labels",label)
        #为真标签计算loss
        loss_known = crossentropyloss(outputs, label)
        loss_known=loss_known*is_known
        #print("loss_known1", loss_known)
        loss_known= torch.sum( loss_known) /(1e-5+torch.sum(is_known))
        #print("loss_known2", loss_known)

        #为假标签计算loss
        with torch.no_grad():
            softmax_output = softmax_func(outputs)  #这样得到的就是概率
            prob=torch.max(softmax_output,dim=1)[0]  #最大的那一个概率

            weight_by_confidence=prob.clone()
            weight_by_confidence[weight_by_confidence<args['confidence_threshold'] ]=0   #最大概率小于置信限比如0.2的，就当成0处理。

            pseudo_label = torch.argmax(outputs, dim=1)  #伪标签

        loss_unknown=crossentropyloss(outputs, pseudo_label)  #用伪标签去算损失


        #print(prob)
        #print(weight_by_confidence)
        #print(type(weight_by_confidence))
        #print(len(weight_by_confidence))
        #print(weight_by_confidence.device)

        loss_unknown=loss_unknown*(1-is_known)*weight_by_confidence


        loss_unknown=loss_unknown.sum()/(1e-5+torch.sum(1-is_known))

        #print("prob",(prob*(1-is_known)).data)   #那些伪标签，看看其概率到底是多少。

        prob_for_pseudo_label=torch.sum(prob*(1-is_known))/(1e-5+torch.sum(1-is_known))  #平均一下，看伪标签的概率是多少



        #总loss
        total_loss=loss_known+get_unlabeled_weight(big_epoch_iter)*loss_unknown

        known_acc = torch.sum( torch.eq(torch.argmax(outputs,dim=1),label) * is_known )  /(1e-5+ torch.sum(is_known))
        unknown_acc= torch.sum(  torch.eq(torch.argmax(outputs,dim=1),label) * (1-is_known) ) / (1e-5+torch.sum(1-is_known))
        num_of_known_label=torch.sum(is_known)

        loss_known_array.append(loss_known.item())
        loss_unknown_array.append(loss_unknown.item())
        total_loss_array.append(total_loss.item())
        known_acc_array.append(known_acc.item())
        unknown_acc_array.append(unknown_acc.item())
        num_of_known_label_array.append(num_of_known_label.item())
        prob_for_pseudo_label_array.append(prob_for_pseudo_label.item())

        #传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        if (step+1)% args['batch_print']==0:
            print('************')
            print("batch %d/%d" % (step + 1, len(dataloader)))
            print("total_loss",total_loss.item())
            print("num of known_label",num_of_known_label.item())
            print("loss known",loss_known.item())
            print("loss unknown",loss_unknown.item())
            print("weight for unknown",get_unlabeled_weight(big_epoch_iter))
            print("known acc",known_acc.item())
            print("unknown acc",unknown_acc.item())
            print("prob_for_pseudo_label",prob_for_pseudo_label.item())
            print('\n\n')

        #展示
        # if step % 100 == 0:
        #     # print("batch: %d/%d, loss: %.4f"%(i,batch_num,loss))
        #     logger.info("batch: %d/%d, total_loss: %.4f,num of known label:%d  " % (step, batch_num, total_loss.item(),num_of_known_label.item()))
        #     logger.info('loss known: %f,loss unknown: %f,known_acc :%f,unknown_acc: %f'%(loss_known,loss_unknown,known_acc,unknown_acc))
        #


    #记录统计信息

    stats={'epoch':big_epoch_iter,'num of known label':np.mean(num_of_known_label_array),
           'alpha_for_unknown_Loss':get_unlabeled_weight(big_epoch_iter),
           'avg loss for known':np.mean(loss_known_array),'avg loss for unknown':np.mean(loss_unknown_array),
           'avg loss for total':np.mean(total_loss_array),'avg known acc':np.mean(known_acc_array),'avg unknown acc':np.mean(unknown_acc_array),
           'prob_for_pseudo_label':np.mean(prob_for_pseudo_label_array)}

    stats_array.append(stats)

    logger.info("In this epoch ,the stats are %s"%stats)



logger.info("Fully Complete!!")

df_stats = pd.DataFrame(stats_array)
logger.info("Now writing stats to excel")
df_stats.to_csv(save_file_name+"/Train_Stats.csv",index=False)


save_model_dir = save_file_name + '/model_parameter.pkl'
torch.save(model.state_dict(), save_model_dir)
# print("Successfully Save model to%s" % save_model_dir)
logger.info("Successfully Save model to%s" % save_model_dir)
