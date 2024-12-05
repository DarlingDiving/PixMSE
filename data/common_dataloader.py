import os
from PIL import Image
from PIL import ImageFile
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from config import opt
import torch.multiprocessing

from PIL import Image
torch.multiprocessing.set_sharing_strategy('file_system')

class CommonDataloader(data.Dataset):

    def __init__(self, root, train=True, test=False,real_name='FFHQ',fake_name='Stylegan2'):
        self.train   = train
        self.test    = test
        self.real_name = real_name
        self.fake_name = fake_name
        real_dir = root+"/%s"%(self.real_name)
        fake_dir = root+"/%s"%(self.fake_name)
        print(real_dir)
        print(fake_dir)

        # get img list
        imgs =  [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
        imgs += [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]
        #imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        
        self.imgs = imgs

        resize = opt.img_size
        if self.train:
            self.transforms_RGB = T.Compose(
                [T.RandomHorizontalFlip(),
                T.Resize(resize),
                T.RandomCrop(opt.img_size, padding=4),
                T.ToTensor(),
                ]
                )
        else:
            self.transforms_RGB = T.Compose(
                [T.Resize(resize),
                T.CenterCrop(opt.img_size),
                T.ToTensor(),
                ]
                )
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __getitem__(self, index):
        img_path = self.imgs[index]
        
        
        if  self.real_name in img_path.split('/'):
            label = 1
        else:
            label = 0
        data = Image.open(img_path)
        data_ = np.array(data).copy()
        # add noise
        noises = np.random.normal(scale=opt.noise_scale, size=data_.shape)
        noises = noises.round()
        im_noise = data_.astype(np.int16) + noises.astype(np.int16)
        im_noise = im_noise.clip(0, 255).astype(np.uint8)
        im = im_noise
        im = Image.fromarray(im).convert('RGB')
        im_rgb = self.transforms_RGB(im)

        return im_rgb, label

    def __len__(self):
        return len(self.imgs)
