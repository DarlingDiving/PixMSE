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
        self.test_data_root1 = opt.test_data_root_cnn

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
    
    
    def test(self,realname,fakename):

        model = main_Net()
        if torch.cuda.is_available():
            model.cuda()
            device=torch.device('cuda')
        else:
            device=torch.device('cpu')
            
        for e in [41]:    
            if opt.load_model and os.path.exists(opt.load_model_path_2+'%s_Uniform.pth'%e):
                print('loading opt.load_model_path_2+%s_Uniform.pth'%e)
                sta =torch.load( opt.load_model_path_2+'%s_Uniform.pth'%e,map_location=device)
                model.load_state_dict(sta)
   
            for gan in ['Stylegan','Stylegan2']:
                    
                data_root1 = self.test_data_root1+'/'+gan
                print('正在计算：%s'%gan)
                test_data   = CommonDataloader(data_root1,  train=False,test=True,real_name=realname,fake_name=fakename)
                test_dataloader = DataLoader(test_data,4,shuffle=True,num_workers=opt.num_workers)
                score,label,cm_value=self.val(model,test_dataloader)
                acc,auc,ap = self.result(score,label,cm_value)
                f=open('./data/%s.txt'%gan,mode='a')
                f.write('%s: acc %.2f ,auc %.2f ,ap %.2f \n'%(gan,acc,auc,ap))
                f.close()

        return
        # handle.remove()

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

