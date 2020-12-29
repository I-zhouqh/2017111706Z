import torch
from torch import nn
from torch.utils.data import Dataset
import math
import numpy as np
#未完成的注意事项：前面传过来的x到底维度是多少应该有个准性。然后in channel也应该改变。


class SSL_Model(nn.Module):
    def __init__(self, frozen_model,frozen_conv,use_additional_features):
        super(SSL_Model, self).__init__()

        self.frozen_model=frozen_model  #预训练好的model，冻住
        self.frozen_conv=frozen_conv  #这个预训练好的model要几层

        self.features=self.make_layers()

        hidden_dim=2048 if use_additional_features else 4096
        self.use_additional_features=use_additional_features

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim , hidden_dim),   #这个第一个2048是怎么来的，其实和尺寸有关，比如前面最后一个卷积层之后出来的是64*512*2*2,512是深度，具体查check3点。
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 10)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        #这里非常关键，frozen_model是冻住的，不要梯度，features_not_frozen 和 classifier需要初始化

        # freeze some layers
        for param in self.frozen_model.parameters():
            param.requires_grad = False

        for m in self.frozen_model.modules():
            if isinstance(m,nn.BatchNorm2d):  #把batch norm给关掉 ！！
                m.eval()

        # unfreeze batchnorm scaling

            # if args.train_batchnorm:
            #     for layer in model.modules():
            #         if isinstance(layer, torch.nn.BatchNorm2d):
            #             for param in layer.parameters():
            #                 param.requires_grad = True


        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                #print(y)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                #print(y)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()




    def make_layers(self):
        layers = []
        in_channels = self.get_in_channel_of_features()   #not 3 ！！看fronzen层传过来的维度是多少
        #cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        cfg = [512, 512, 512, 'M']
        #这个cfg只要一部分就好了

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]

                in_channels = v  # mind it ！
        return nn.Sequential(*layers)



    def get_in_channel_of_features(self):
        #我想要知道feature部分的in_channel是多少，即是前面冻住部分的最后一层的out_channel (也就是卷积核个数）

        count = 1
        last_out_channel=None

        for m in self.frozen_model.features.modules():

            if isinstance(m,nn.Conv2d):  #如果这是卷积层，记下它的卷积核个数，即是下一层的in_channel
                last_out_channel=m.out_channels

            if isinstance(m, nn.ReLU):
                if count == self.frozen_conv :
                    break
                count = count + 1

        return last_out_channel



    def forward(self,x):


        #Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        Avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        count = 1

        for m in self.frozen_model.features.modules():
            if not isinstance(m, nn.Sequential):
                x = m(x)
            if isinstance(m, nn.ReLU):
                if count == self.frozen_conv :
                    break
                count = count + 1

        #print("check0", x.size())
        if self.frozen_conv<11:   #如果层数太高了就不再去pool，会导致size太小。
            x = Avgpool(x)

        #print("check1",x.size())

        if self.use_additional_features:
            x=self.features(x)

        #print("check2",x.size())


        x=x.view(x.size(0),-1)

        #print("check3",x.size())

        x=self.classifier(x)

        #print("check4",x.size())

        return x




class SSL_DATASET(Dataset):

    def __init__(self,dataset,known_label_index,num_of_unknown_label):
        """

        :param dataset:  dataset
        :param known_label_index: 那些
        :param only_known_label:  boolen，是否只用known label，如果是的话，样本将会很少500个，用于刚开始
        """


        self.num_of_unknown_label=num_of_unknown_label

        self.dataset=dataset

        self.known_label_index=known_label_index
        self.known_label_num=len(self.known_label_index)

        self.selected_index=self.get_select_index()

    def get_select_index(self):
        all_index=list(range(len(self.dataset)))
        unknown_label_index=[item for item in all_index if item not in self.known_label_index]
        unknown_label_index=np.random.choice(unknown_label_index,self.num_of_unknown_label,replace=False)
        result=self.known_label_index+unknown_label_index.tolist()

        # print(self.known_label_index)
        # print(unknown_label_index.tolist())

        return result

    def __getitem__(self,index):

        data,true_label=self.dataset[self.selected_index[index]]
        if index<self.known_label_num:
            known=1
        else:
            known=0

        return (data,true_label,known)


    def __len__(self):
        return len(self.selected_index)

