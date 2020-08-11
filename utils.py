import matplotlib.pyplot as plt
plt.switch_backend('agg')
import itertools
import numpy as np 
import torch
from PIL import Image
from torchvision import transforms
from torchvision import utils
import scipy.misc
import scipy
from torchvision.utils import make_grid#,save_image

def show_result(num_epoch, test_images, show=False, save=False, path = 'result.png'):
    size_figure_grid = int(np.sqrt(test_images.size()[0]))
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        if test_images.size()[1] == 1:         
            ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0).squeeze() + 1) / 2, cmap='gray')
        else:
            ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            
    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
        

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    #im.save(filename)
    im.convert('L').save(filename)

def imsave_singel(test_images, path):
    #utils.save_image(test_images[0,:,:,:], path, nrow=1, normalize=True)
    #transform_ = transforms.ToPILImage(mode="L")
    #image = transform_(np.uint8(test_images.detach().numpy()))
    save_image(test_images, path, nrow=1,)

