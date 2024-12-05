import torch
import torch.nn as nn
from .basicmodule import BasicModule
from models.resnet_uniform import resnet18
from models.Function_Net import Uniform_net
import torch.nn.functional as F

class Filter_module(nn.Module):
    def __init__(self):
        super(Filter_module, self).__init__()

        kernel_1 = [[0,0,0],[1,-2,1],[0,0,0]]
        kernel_2 = [[0,1,0],[0,-2,0],[0,1,0]]
        kernel_3 = [[0,0,1],[0,-2,0],[1,0,0]]
        kernel_4 = [[1,0,0],[0,-2,0],[0,0,1]]

        kernel_1 = torch.FloatTensor(kernel_1).unsqueeze(0).unsqueeze(0)
        self.weight_1 = nn.Parameter(data=kernel_1, requires_grad=False)

        kernel_2 = torch.FloatTensor(kernel_2).unsqueeze(0).unsqueeze(0)
        self.weight_2 = nn.Parameter(data=kernel_2, requires_grad=False)

        kernel_3 = torch.FloatTensor(kernel_3).unsqueeze(0).unsqueeze(0)
        self.weight_3 = nn.Parameter(data=kernel_3, requires_grad=False)

        kernel_4 = torch.FloatTensor(kernel_4).unsqueeze(0).unsqueeze(0)
        self.weight_4 = nn.Parameter(data=kernel_4, requires_grad=False)
 
    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]

        x1_1 = F.conv2d(x1.unsqueeze(1), self.weight_1, padding=1)
        x1_1 = x1_1**2
        x2_1 = F.conv2d(x2.unsqueeze(1), self.weight_1, padding=1)
        x2_1 = x2_1**2
        x3_1 = F.conv2d(x3.unsqueeze(1), self.weight_1, padding=1)
        x3_1 = x3_1**2

        x1_2 = F.conv2d(x1.unsqueeze(1), self.weight_2, padding=1)
        x1_2 = x1_2**2
        x2_2 = F.conv2d(x2.unsqueeze(1), self.weight_2, padding=1)
        x2_2 = x2_2**2
        x3_2 = F.conv2d(x3.unsqueeze(1), self.weight_2, padding=1)
        x3_2 = x3_2**2

        x1_3 = F.conv2d(x1.unsqueeze(1), self.weight_3, padding=1)
        x1_3 = x1_3**2
        x2_3 = F.conv2d(x2.unsqueeze(1), self.weight_3, padding=1)
        x2_3 = x2_3**2
        x3_3 = F.conv2d(x3.unsqueeze(1), self.weight_3, padding=1)
        x3_3 = x3_3**2

        x1_4 = F.conv2d(x1.unsqueeze(1), self.weight_4, padding=1)
        x1_4 = x1_4**2
        x2_4 = F.conv2d(x2.unsqueeze(1), self.weight_4, padding=1)
        x2_4 = x2_4**2
        x3_4 = F.conv2d(x3.unsqueeze(1), self.weight_4, padding=1)
        x3_4 = x3_4**2

        x_1 = torch.sqrt(x1_1 + x1_2 + x1_3 + x1_4)#.unsqueeze(0).transpose(0,1)
        x_2 = torch.sqrt(x2_1 + x2_2 + x2_3 + x2_4)#.unsqueeze(0).transpose(0,1) 
        x_3 = torch.sqrt(x3_1 + x3_2 + x3_3 + x3_4)#.unsqueeze(0).transpose(0,1) 

        x = torch.cat((x_1,x_2,x_3), dim=1)
        #x = torch.cat((x_3,x_3,x_3), dim=1)
       
        return x


class main_Net(BasicModule):
    def __init__(self):
        super(main_Net, self).__init__()
        self.extract_feature = resnet18(pretrained=False)
        
        self.uniform = Uniform_net(in_channel=3,out_channel=3)
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.BatchNorm2d1 = torch.nn.BatchNorm2d(32)
        self.BatchNorm2d2 = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
        self.Filter = Filter_module()
        #self.fc1 = nn.Linear(8192*2, 8192)
        self.fc2 = nn.Linear(512*2, 2)
        
        #self.gamma = torch.nn.Parameter(torch.ones(1))
        #self.lama = torch.nn.Parameter(torch.ones(1))
            
    def forward(self, x):
        x_u = self.uniform(x)+x
        x_Filter = self.Filter(x)
        
        x_u = self.relu(self.BatchNorm2d1(self.conv1(x_u)))
        x_u = self.relu(self.BatchNorm2d2(self.conv2(x_u)))

        feature = self.extract_feature(x_Filter)
        feature1= self.extract_feature(x_u)

        feature= torch.cat((feature,feature1),dim=1)
        feature_a = feature.view(x.size(0), -1)
        result = self.fc2(feature_a)
        


        return feature_a,result

