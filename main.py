import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
#from torch.utils.data.dataset import Dataset
from UTGAN import UTGAN
from gen_models import Generator_Unet
from dis_models import Discriminator_steg
from Dataloader import MyDataset
from utils import imsave_singel, show_result
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

Train = True
use_cuda = True
img_nc = 1
payload = 0.4
epochs_train = 600
epochs_eval = 1
lr = 0.0001
batch_size_train = 20
batch_size_eval = 1
bilinear = False
DIR = '/data/BossClf/BOSSBase_256'

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_model = '/data/cuiqi/ASGAN/models/netG_epoch_1.pth'
img_prob_eval_path = './eval_result/'

mytransform = transforms.Compose([transforms.ToTensor(),]) 
if Train:
    dataset = MyDataset(DIR, transform=mytransform )
    dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)
    UTGAN_trainer = UTGAN(device, img_nc, lr, payload, bilinear)
    UTGAN_trainer.train(dataloader, epochs_train)
else:
    dataset = MyDataset(DIR, transform=mytransform )
    dataloader = DataLoader(dataset, batch_size=batch_size_eval, shuffle=False, num_workers=1)
    model = Generator_Unet(img_nc,1, bilinear)
    model.load_state_dict(torch.load(pretrained_model), strict=False)
    model.cuda()
    model.eval()
    for i, data in enumerate(dataloader, start=0):
        images = data['img']
        images = images.to(device)
        prob_pred = model(images)
        imsave_singel(prob_pred, path=img_prob_eval_path + str(i) +'.tif')
