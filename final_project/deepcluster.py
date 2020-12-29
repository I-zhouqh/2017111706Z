# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
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


args={'seed':4396,'verbose':False,'use_ssl_kmeans':True,'nmb_cluster':10,'batch':64,'num_workers':0,'big_epoch':100,
      'small_epoch':1,'lr':0.01,'wd':-4,'momentum':0.9,'pca':64,'n_init':5,'lr_classifier':0.05,'known_label_size':100,
      'batches_print':100}

# if args['use_ssl_kmeans'] and args['nmb_cluster']!=10:
#     raise   ValueError("When use ssl kmeans, the nmb_cluster must match the true label num, which is 10")


nowtime=time.strftime("%b%d_%H_%M", time.localtime())   # 'Dec12_16_13'
save_file_name='./model_save_'+nowtime

# Create output directory if needed
if not os.path.exists(save_file_name):
    os.makedirs(save_file_name)



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

def save_model(model,big_epoch_iter):
    try:
        save_model_dir =save_file_name+ '/parameter_epoch_' + str(big_epoch_iter) + '.pkl'
        torch.save(model.state_dict(), save_model_dir)
        #print("Successfully Save model to%s" % save_model_dir)
        logger.info("Successfully Save model to%s" % save_model_dir)
    except:
        save_model_dir = save_file_name+ '/parameter_epoch_' + str(big_epoch_iter) + '.pkl'
        #print("Model Saving Failed to%s" % save_model_dir)
        logger.info("Model Saving Failed to%s" % save_model_dir)

def train(loader, model, crit, opt, epoch):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """


    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args['lr_classifier'],
        weight_decay=10**args['wd'],
    )

    end = time.time()

    for epoch_times in range(args['small_epoch']):

        loss_array=[]
        for i, data in enumerate(loader):
            batch_num=len(loader)
            input_tensor,target = map(lambda x: x.to(DEVICE), data)  #这个target是伪标签

            #target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input_tensor.cuda())
            target_var = torch.autograd.Variable(target)
            target_var=target_var.long()

            output = model(input_var)
            loss = crit(output, target_var)
            loss_array.append(loss.item())


            # compute gradient and do SGD step
            opt.zero_grad()

            optimizer_tl.zero_grad()
            loss.backward()

            opt.step()
            optimizer_tl.step()

            # measure elapsed time

            if i % args['batches_print'] == 0:
                #print("batch: %d/%d, loss: %.4f"%(i,batch_num,loss))
                logger.info("batch: %d/%d, loss: %.4f"%(i,batch_num,loss))
                # print('Epoch: [{0}][{1}/{2}]\t'
                #       'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #       'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                #       'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                #       .format(epoch, i, len(loader), batch_time=batch_time,
                #               data_time=data_time, loss=losses))

    return np.mean(loss_array)

def compute_features(dataloader, model, N):

    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda())

        with torch.no_grad():

            aux = model(input_var).data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * args['batch']: (i + 1) * args['batch'] ]= aux
            else:
                # special treatment for final batch
                features[i * args['batch']:] = aux

    return features

def get_kmeans_model(features):

    kmeans_model=None

    if args['use_ssl_kmeans']:

        np.random.seed(seed=args['seed'])
        known_label_size = args['known_label_size']  #有多少个已知标签
        assert known_label_size % 10 == 0, "known_label_size must be 10的倍数"

        chosen_index = {}

        for classed_label in range(10):
            this_class_index = np.flatnonzero(truth_label == classed_label)
            tmp = np.random.choice(this_class_index, int(known_label_size / 10), replace=False)
            chosen_index[classed_label] = tmp

        clusters_init = np.zeros((10, args['pca']))  #维度与特征的维度匹配
        for classed_label in range(10):
            clusters_init[classed_label, :] = np.mean(features[chosen_index[classed_label], :], axis=0)


        if args['nmb_cluster']==10:  #如果只有10个，那刚好就用这10个类来初始化
            kmeans_model = KMeans(n_clusters=10, init=clusters_init, n_init=1, max_iter=300,
                                  tol=0.0001,
                                  verbose=0, random_state=None, algorithm='auto', precompute_distances=True)

        elif args['nmb_cluster']>10 :

            chosen_index_array=[]
            for item in chosen_index.values():
                chosen_index_array.extend(item.tolist())

            n,p=features.shape

            selected_array=[]

            random_select=chosen_index[0][-1]  #随便拿一个出来作为初始


            for iter in range(10,args['nmb_cluster']):
                distance=np.sum( np.square( features[random_select,:]-features ) ,axis=1)
                distance[chosen_index_array]=0  #那些点被选过了，就不要再出现了
                distance/=np.sum(distance)  #归一化

                while 1:
                    select=np.random.choice(n,p=distance)  #根据距离来抽奖，kmeans++算法
                    if select not in selected_array: #一定要抽到不一样的才行
                        selected_array.append(select)
                        random_select=select
                        break

            clusters_init=np.concatenate((features[selected_array,:],clusters_init),axis=0)

            kmeans_model = KMeans(n_clusters=args['nmb_cluster'], init=clusters_init, n_init=1, max_iter=300,
                                  tol=0.0001,
                                  verbose=0, random_state=None, algorithm='auto', precompute_distances=True)


        else:
            raise ValueError("错误输入 for nmb cluster")



    else:

        kmeans_model = KMeans(n_clusters=args['nmb_cluster'], init='k-means++', n_init=args['n_init'], max_iter=300,
                              tol=0.0001,
                              verbose=0, random_state=None, algorithm='auto', precompute_distances=True)

    return kmeans_model



logger = init_logger(filename=save_file_name+"/diary.log")



DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#一个big_epoch里面进行一次伪标签assign和训练。训练几个epoch呢，small_epoch个。


logger.info("All the information will be stored in %s"%save_file_name)
logger.info("The config are: %s"%str(args))


# fix random seeds
torch.manual_seed(args['seed'])
torch.cuda.manual_seed_all(args['seed'])
np.random.seed(args['seed'])

# CNN

model = VGG(num_classes=args['nmb_cluster'])
model.to(DEVICE)

feature_dimension = model.top_layer.weight.size()[1]

model.top_layer = None
model.features = torch.nn.DataParallel(model.features)
model.cuda()
cudnn.benchmark = True

# create optimizer
optimizer = torch.optim.SGD(
    filter(lambda x: x.requires_grad, model.parameters()),
    lr=args['lr'],
    momentum=args['momentum'],
    weight_decay=10**args['wd'],
)

# define loss function
criterion = nn.CrossEntropyLoss().cuda()

# preprocessing of data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
     ])

# load the data

#dataset = datasets.CIFAR10('./cifar', transform=transforms.Compose(transform))


#  训练数据集
dataset = datasets.CIFAR10(root='./cifar',
                                             train=True,
                                             transform=transform,
                                             download=True)
truth_label=np.array(dataset.targets)

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args['batch'],
                                         num_workers=args['num_workers'],
                                         pin_memory=True)

# clustering algorithm to use



last_time_cluster_label=None
stats_array=[]

# training convnet with DeepCluster
for big_epoch_iter in range(1,args['big_epoch']+1):

    #adjust_learning_rate(optimizer, epoch)   #调整学习率

    #print('************* EPOCH %d/%d ***************************\n'%(big_epoch_iter,args['big_epoch']) )
    logger.info('************* EPOCH %d/%d ***************************\n'%(big_epoch_iter,args['big_epoch']) )


    # 提取特征，因此无需relu，无需接分类层，需要预先把之去掉
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])  #children是取出所有层，将其list化，然后去掉最后一层.

    ###             计算得到特征，并进行Kmeans       ###
    features = compute_features(dataloader, model, len(dataset))




    #进行Kmeans，返回分类的list，计算伪标签，计算loss。
    features=preprocess_features(features,pca=args['pca'])

    # kmeans_model = KMeans(n_clusters=args['nmb_cluster'], init='k-means++', n_init=args['n_init'], max_iter=300,
    #                       tol=0.0001,
    #                       verbose=0, random_state=None, algorithm='auto', precompute_distances=True)

    kmeans_model=get_kmeans_model(features)

    t3=time.time()
    kmeans_model.fit(features)
    logger.info("Time for cluster: %s"%(time.time() - t3))  #打印Kmeans的时间

    print(pd.Series(kmeans_model.labels_).value_counts().sort_values(ascending=False))


    ###             计算NMI       ###
    nmi_diff=None
    if last_time_cluster_label is not None:

        nmi_diff = normalized_mutual_info_score(
            last_time_cluster_label,
            kmeans_model.labels_
        )

        #print('NMI against previous assignment: {0:.3f}'.format(nmi_diff))
        logger.info('NMI against previous assignment: {0:.3f}'.format(nmi_diff))

    last_time_cluster_label = kmeans_model.labels_
    nmi_with_truth=normalized_mutual_info_score(truth_label,kmeans_model.labels_)
    #print('NMI against Truth: {0:.3f}'.format(nmi_with_truth))
    logger.info('NMI against Truth: {0:.3f}'.format(nmi_with_truth))

    ###             整理为dataset，dataloader       ###

    train_dataset = ReassignedDataset(dataset , kmeans_model.labels_)
    train_dataset.check_pseudolabel_distribution()


    #print("now check:")
    logger.info("now check:")

    #print(len(set(kmeans_model.labels_)))
    #print(max(kmeans_model.labels_))
    logger.info (len(set(kmeans_model.labels_)))
    logger.info (max(kmeans_model.labels_))


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args['batch'],
        num_workers=2,
        pin_memory=True,
    )

    #print('伪标签get!')
    logger.info('伪标签get!')


    ###             训练       ###

    # 刚才把人家给去掉了，现在把最后一层给加回去。同时，分类层应该重新初始化！
    mlp = list(model.classifier.children())
    mlp.append(nn.ReLU(inplace=True).cuda())
    model.classifier = nn.Sequential(*mlp)
    model.top_layer = nn.Linear(feature_dimension, len(set(kmeans_model.labels_)) )
    model.top_layer.weight.data.normal_(0, 0.01)
    model.top_layer.bias.data.zero_()
    model.top_layer.cuda()

    # train network with clusters as pseudo-labels，有标签了，训练即可。

    loss = train(train_dataloader, model, criterion, optimizer, big_epoch_iter )


    #print('Clustering loss: %.3f \n'
    #      'ConvNet loss: %.3f'
    #      %(kmeans_model.inertia_, loss))
    logger.info('Clustering loss: %.3f    '
          'ConvNet loss: %.3f'
          %(kmeans_model.inertia_, loss))



    ###             记录统计信息       ###



    stats={'epoch':big_epoch_iter,'cluster_loss':kmeans_model.inertia_,'NMI_true':nmi_with_truth,'NMI_last':nmi_diff,'NNW_loss':loss  }
    stats_array.append(stats)


    if big_epoch_iter%20==0: #每20次保存模型
        save_model(model,big_epoch_iter)


    #print('\n\n')
    logger.info('\n\n')



#print("Fully Complete!!")
logger.info("Fully Complete!!")

df_stats = pd.DataFrame(stats_array)
#df_stats = df_stats.set_index('epoch')
#print(df_stats)
logger.info(df_stats)
df_stats.to_csv(save_file_name+"/Train_Stats.csv",index=False)


    # save running checkpoint
    # torch.save({'epoch': epoch + 1,
    #             'arch': args.arch,
    #             'state_dict': model.state_dict(),
    #             'optimizer' : optimizer.state_dict()},
    #            os.path.join(args.exp, 'checkpoint.pth.tar'))
    #
    # # save cluster assignments
    # cluster_log.log(deepcluster.images_lists)

