import torch
from torch import nn
from torch.utils.data import Dataset
import math
import numpy as np
#未完成的注意事项：前面传过来的x到底维度是多少应该有个准性。然后in channel也应该改变。



class VGG(nn.Module):

    def __init__(self , num_classes=10):
        super(VGG, self).__init__()
        self.features = self.make_layers()
        self.classifier = nn.Sequential(
            nn.Linear(512 , 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )

        self._initialize_weights()


    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for y,m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
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
        in_channels = 3
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)





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

