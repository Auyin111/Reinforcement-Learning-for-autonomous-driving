import torch
import matplotlib.pyplot as plt
import torchvision
from PIL import Image


def plot_img(img, *args, **kwargs):
    plt.figure(figsize=(5, 5))
    plt.imshow(img, *args, **kwargs)
    plt.axis('off')
    plt.show()


def plot_img_from_path(path_img):
    """e.g. plot_img_from_path(os.path.join(dir_auto_img, 'in', '80.png'))"""

    img = torchvision.io.read_image(path_img).permute(1, 2, 0)
    img = Image.fromarray(img.cpu().detach().numpy(), 'RGB')
    img.show()


def load_img(path_img, device):
    """load an image for prediction"""

    img = torchvision.io.read_image(path_img).float().unsqueeze(0).to(device)
    return img


def output_to_arr(outputs):
    """convert the predicted output from tensor to array"""

    _, predicted = torch.max(outputs.data, 1)
    arr_predicted = predicted.detach().cpu().numpy()

    return arr_predicted
