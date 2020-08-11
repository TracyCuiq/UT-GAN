import os
import numpy as np
from torch.utils.data.dataset import Dataset
from glob import glob
import torchvision.transforms as transforms
from PIL import Image
from scipy import ndimage
import torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class MyDataset(Dataset):
  def __init__(self, DATASET_DIR, transform=None):
    self.transform = transform
    self.cover_dir = DATASET_DIR
    self.cover_list = [x.split('/')[-1] for x in glob(self.cover_dir+'/*')]
    assert len(self.cover_list) != 0, "cover_dir is empty"
    
  def __len__(self):
    return len(self.cover_list)

  def __getitem__(self, idx):
    file_index = int(idx)
    cover_path = os.path.join(self.cover_dir, self.cover_list[file_index])
    cover_data = Image.open(cover_path)
    cover_nd = np.array(cover_data)

    cover_trans = cover_nd
    if cover_trans.max() > 1:
        cover_trans = cover_trans / 255
        
    if self.transform:
        data = self.transform(cover_data)
    sample = {'img': data}
    return sample

if __name__ == '__main__':
    DIR = '/data/BossClf/BOSSBase_256'
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    data = MyDataset(DATASET_DIR=DIR, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=16, num_workers=1, shuffle=True)
    for batch_num, sample in enumerate(train_loader):
        #print(sample['img'].shape)
        img = sample['img']
        img_shape = list(img.size())
        img = img.reshape(img_shape[0] * img_shape[1], *img_shape[2:])
        print(img.shape)
