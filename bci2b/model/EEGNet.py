"""EEGNet (Lawhern et al., J. Neural Eng. 2018) in PyTorch, original layout: F2 pointwise filters and a
flattened classifier. Constructor arguments and docstring follow the reference implementation
https://github.com/vlawhern/arl-eegmodels (CC0 1.0, see THIRD_PARTY_NOTICES.md)."""
import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, num_classes, chans=64, samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()
        
        '''
        Inputs:
        
        nb_classes      : int, number of classes to classify
        Chans, Samples  : number of channels and time points in the EEG data
        dropoutRate     : dropout fraction
        kernLength      : length of temporal convolution in first layer. We found
                            that setting this to be half the sampling rate worked
                            well in practice. 
        F1, F2          : number of temporal filters (F1) and number of pointwise
                            filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
        D               : number of spatial filters to learn within each temporal
                            convolution. Default: D = 2
        dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
        '''
        
        # dropoutRate: within-subjet=0.5 cross-subject=0.25
                
        # 根据dropout类型选择dropout层
        if dropoutType == 'SpatialDropout2D':
            self.dropoutType = nn.Dropout2d(dropoutRate)
        elif dropoutType == 'Dropout':
            self.dropoutType = nn.Dropout(dropoutRate)
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')

        # 第一块：时间卷积 -> 批量归一化 -> 深度卷积 -> 批量归一化 -> 激活 -> 池化 -> dropout
        self.time_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), bias=False),
            nn.BatchNorm2d(F1),
            # DepthwiseConv2d
            nn.Conv2d(F1, D*F1, (chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D*F1),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 4)),
            self.dropoutType
        )
        
        # 第二块：空间可分离卷积 -> 批量归一化 -> 激活 -> 池化 -> dropout
        self.separable_conv = nn.Sequential(
            nn.Conv2d(D*F1, D*F1, (1, 16), groups=D*F1, bias=False),
            nn.Conv2d(D*F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 8)),
            self.dropoutType
        )
        
        out = torch.ones((1, 1, chans, samples))
        out = self.time_conv(out)
        out = self.separable_conv(out)
        n_out_time = out.cpu().data.numpy().shape

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_out_time[-1] * n_out_time[-2] * n_out_time[-3], num_classes),
            nn.Softmax(dim=1)  # 使用dim=1，因为输入的是(batch_size, nb_classes)
        )

    def extract_features(self, x):
        # 获取时间卷积和空间卷积的特征
        x = self.time_conv(x)
        x = self.separable_conv(x)
        return x
    
    def forward(self, x):
        x = self.time_conv(x)
        x = self.separable_conv(x)          # [64, 16, 1, 9]
        x = self.classifier(x)
        return x
