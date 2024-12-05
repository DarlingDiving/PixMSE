import torch as t
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import random
from data.common_dataloader import CommonDataloader

from models.Network import main_Net
from config import opt
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import os
from tqdm import tqdm
import time

# load old model weights into new model (only for layers with common name)
def load_model_weights(old_model,new_model):
    if opt.use_gpu:
        new_model.cuda()
    pretrained_dict = old_model.state_dict()
    substitute_dict = new_model.state_dict()
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in substitute_dict}
    substitute_dict.update(pretrained_dict)
    new_model.load_state_dict(substitute_dict)
    print('loaded done')
    return new_model

def setup_seed(seed):
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.backends.cudnn.benchmark = False
    t.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


class CombineTrainer():
    def __init__(self, opt):

        self.train_data_root = opt.train_data_root
        self.val_data_root = opt.val_data_root
        self.test_data_root1 = opt.test_data_root

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.constant(m.weight, 1e-2)
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias,0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            #print(m)
            nn.init.kaiming_normal(m.weight, mode="fan_out")
            # nn.init.constant(m.weight, 1e-3)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 2e-1)
            nn.init.constant(m.bias, 0)


    def train(self):
        epoch_times = []
        model_1 = main_Net()
        model_1.apply(self.weight_init)

        if torch.cuda.is_available():
            model_1.cuda()
            device=torch.device('cuda')
        else:
            device=torch.device('cpu')
            
        if opt.load_model and os.path.exists(opt.load_model_path_2+'100_Uniform.pth'):
            print('loading')
            sta =torch.load( opt.load_model_path_2+'100_Uniform.pth',map_location=device)
            model_1.load_state_dict(sta)
        
        model_1.train()

        #  data
        train_data = CommonDataloader(self.train_data_root, train=True,real_name="0_real",fake_name="1_fake")
        val_data   = CommonDataloader(self.val_data_root,   train=False,test = True,real_name="0_real",fake_name="1_fake")
        train_dataloader = DataLoader(train_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)
        val_dataloader = DataLoader(val_data, opt.batch_size,shuffle=True,num_workers=opt.num_workers)
        
        criterion = t.nn.CrossEntropyLoss()
        learning_rate = opt.lr
        optimizer = t.optim.Adam(model_1.parameters(),lr=learning_rate,weight_decay=opt.weight_decay)


        for epoch in range(60):
            start_time = time.perf_counter()
            start_time = time.time()
           
            for ii, (lr_rgb,label), in tqdm(enumerate(train_dataloader),total=len(train_dataloader), desc='Training'):
                
                lr_rgb_input = Variable(lr_rgb, requires_grad=True)
                label_target = Variable(label)
                if opt.use_gpu and torch.cuda.is_available():
                    lr_rgb_input = lr_rgb_input.cuda()
                    label_target = label_target.cuda()
                optimizer.zero_grad()

                _,score = model_1( lr_rgb_input)
                
            
                loss = criterion(score, label_target)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

                if (ii + 1) % 50 == 0:
                    print("state 2 Epoch %03d ,loss: %.7f "%(epoch,loss.item()))

            
            end_time = time.perf_counter()
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)
       
            if epoch%2==0:
                model_1.save(opt.save_model_path+'%s_Uniform.pth'%(epoch+1))
                score_all,label_all,cm_value=self.val(model_1,val_dataloader)
                self.result(score_all,label_all,cm_value)
            print('time %.2fs'%(end_time-start_time))
           

    def val(self,model_1,dataloader):
        model_1.eval()
        confusion_matrix = meter.ConfusionMeter(2)
        score_all = []
        label_all = []
        with t.no_grad():
            for ii,(rgb,label) , in tqdm(enumerate(dataloader),total=len(dataloader), desc='testing'):
                valrgb_input = Variable(rgb)
                val_label = Variable(label.type(t.LongTensor))
                if torch.cuda.is_available():
                    valrgb_input = valrgb_input.cuda()
                    val_label = val_label.cuda()
                _,score= model_1(valrgb_input)
                confusion_matrix.add(score.data, label.long())
                score_all.extend(score[:,1].detach().cpu().numpy())
                label_all.extend(label)
        cm_value = confusion_matrix.value()
        return score_all,label_all,cm_value
    
    def result(self,score_all,label_all,cm_value):

        accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) /(cm_value.sum())
        auc = roc_auc_score(label_all, score_all)
        ap = average_precision_score(label_all,score_all)
        print(f"val acc.{accuracy} | AUC {auc} | AP {ap*100} |", end='')
        print('\n')
        return accuracy,auc*100,ap*100

    def print_parameters(self,model):
        pid = os.getpid()
        total_num = sum(i.numel() for i in model.parameters())
        trainable_num = sum(i.numel() for i in model.parameters() if i.requires_grad)

        print("=========================================")
        print("PID:",pid)
    
        print("\nNum of parameters:%i"%(total_num))
        print("Num of trainable parameters:%i"%(trainable_num))
        print("Save model path:",opt.save_model_path)
        print("seed:",opt.seed)
        print("learning rate:",opt.lr)
        print("batch_size:",opt.batch_size)

        print("=========================================")

