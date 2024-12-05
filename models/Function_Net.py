import torch
import torch.nn as nn
import torch.nn.functional as F
from .basicmodule import BasicModule


class SquaredDifferenceConv2d(nn.Module):
    def __init__(self, s, p):
        super(SquaredDifferenceConv2d, self).__init__()
        self.in_c = 1
        self.s = s
        self.p = p
        
        
        self.kernels = {
            'right': nn.Parameter(torch.tensor([[0, 0, 0],
                                               [0, 1, -1],
                                               [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False),
            'left': nn.Parameter(torch.tensor([[0, 0, 0],
                                              [-1, 1, 0],
                                              [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False),
            'down': nn.Parameter(torch.tensor([[0, 0, 0],
                                              [0, 1, 0],
                                              [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False),
            'up': nn.Parameter(torch.tensor([[0, -1, 0],
                                            [0, 1, 0],
                                            [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False),
            'diag_down_right': nn.Parameter(torch.tensor([[0, 0, 0],
                                                          [0, 1, 0],
                                                          [0, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False),
            'diag_down_left': nn.Parameter(torch.tensor([[0, 0, 0],
                                                         [0, 1, 0],
                                                         [-1, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False),
            'diag_up_right': nn.Parameter(torch.tensor([[0, 0, -1],
                                                       [0, 1, 0],
                                                       [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False),
            'diag_up_left': nn.Parameter(torch.tensor([[-1, 0, 0],
                                                       [0, 1, 0],
                                                       [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False)
        }

    def forward(self, x):
        size_ = (x.size(2) - 3 + 2 * self.p) // self.s + 1
        squared_diff_sum = torch.zeros((x.size(0), 3, size_, size_), device=x.device)

        for key in self.kernels:
            kernel = self.kernels[key]
            kernel = kernel.to(x.device)
            #kernel_output1 = F.conv2d(x[:,0].unsqueeze(1), kernel, stride=self.s, padding=self.p )
            #kernel_output2 = F.conv2d(x[:,1].unsqueeze(1), kernel, stride=self.s, padding=self.p )
            kernel_output3 = F.conv2d(x[:,1].unsqueeze(1), kernel, stride=self.s, padding=self.p )
            kernel_output = torch.cat((kernel_output3,kernel_output3,kernel_output3),dim=1)
            squared_diff_sum += torch.square(kernel_output)
        
        return torch.sqrt(squared_diff_sum)



def sub_channel(data):
    result = torch.zeros_like(data)
    for i in range(data.size(1)):
        channel_i = data[:, i, :, :]

        expanded_channel_i = channel_i.unsqueeze(1)

        diff_squared = torch.square(data - expanded_channel_i) 

        channel_result = diff_squared.sum(dim=1)

        result[:, i, :, :] = channel_result

        return torch.sqrt(result)


class UpsampleCNN(nn.Module):
    def __init__(self, in_channels, out_channels,feature_size):
        super(UpsampleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((feature_size // 2, feature_size // 2))

        #self.upsample = nn.ConvTranspose2d(128,128,3,2,1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.adaptive_pool_full = nn.AdaptiveAvgPool2d((feature_size, feature_size))
 
        self.conv_out = nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = self.upsample(x)
        x = self.adaptive_pool_full(x)
        x = self.conv_out(x)
        return x


class Uniform_net(BasicModule):
    def __init__(self,in_channel,out_channel ):
        super(Uniform_net, self).__init__()
        self.conv_1 = SquaredDifferenceConv2d(in_c=in_channel,out_c=out_channel,s=1,p=1)
        self.conv_2 = SquaredDifferenceConv2d(in_c=in_channel,out_c=out_channel,s=2,p=1)
        self.conv_3 = SquaredDifferenceConv2d(in_c=in_channel,out_c=out_channel,s=3,p=1)
        self.conv_channel = sub_channel
        self.advnet2_1 = UpsampleCNN(in_channel,out_channel,299)
        self.advnet3_1 = UpsampleCNN(in_channel,out_channel,299)
            
    def forward(self, feature):
        sub_feature = self.conv_channel(feature)
        #print(sub_feature)
        
        x1 = self.conv_1(feature)
        x2 = self.conv_2(feature)
        x3 = self.conv_3(feature)
        
        x21 = self.advnet2_1 (x2)
        x31 = self.advnet3_1 (x3)
        #print(x1)
        #return 0.5*sub_feature+0.5*x1
        #return torch.cat((sub_feature,x1,x21,x31),dim=1)
        return sub_feature+x1+x21+x31
    

