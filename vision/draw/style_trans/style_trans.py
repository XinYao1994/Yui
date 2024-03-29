# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#sphx-glr-advanced-neural-style-tutorial-py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import PIL.Image as Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

# we define two distances, one for the content (DC) and one for the style (DS).
dirs_ = "./watch/data/arts/misheng/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 
imsize = 512 # if torch.cuda.is_available() else 128
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((imsize, imsize))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img

def image_shower(img, img_path):
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img = img.resize(Image.open(img_path).size)
    img.save(img_path + "after", "JPEG")
    # img.show()


# plt.figure()
# imshow(style_img, title='Style Image')

# plt.figure()
# imshow(content_img, title='Content Image')

class Content_Loss(nn.Module):
    def __init__(self, target, weight):
        super(Content_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        # 必须要用detach来分离出target，这时候target不再是一个Variable，这是为了动态计算梯度，否则forward会出错，不能向前传播
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        out = input.clone()
        return out

    def backward(self, retain_variabels=True):
        self.loss.backward(retain_graph=True)
        return self.loss

class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, input):
        a, b, c, d = input.size()
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        gram /= (a * b * c * d)
        return gram

class Style_Loss(nn.Module):
    def __init__(self, target, weight):
        super(Style_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        self.gram = Gram()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        G = self.gram(input) * self.weight
        self.loss = self.criterion(G, self.target)
        out = input.clone()
        return out

    def backward(self, retain_variabels=True):
        self.loss.backward(retain_graph=True)
        return self.loss

cnn = models.vgg19(pretrained=True).features.to(device)

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_loss(style_img,
                             content_img,
                             cnn=cnn,
                             style_weight=1200,
                             content_weight=1,
                             content_layers=content_layers_default,
                             style_layers=style_layers_default):

    content_loss_list = []
    style_loss_list = []

    model = nn.Sequential()
    model = model.cuda()
    gram = Gram()
    gram = gram.cuda()

    i = 1
    for layer in cnn:
        if isinstance(layer, nn.Conv2d):
            name = 'conv_' + str(i)
            model.add_module(name, layer)

            if name in content_layers_default:
                target = model(content_img)
                content_loss = Content_Loss(target, content_weight)
                model.add_module('content_loss_' + str(i), content_loss)
                content_loss_list.append(content_loss)

            if name in style_layers_default:
                target = model(style_img)
                target = gram(target)
                style_loss = Style_Loss(target, style_weight)
                model.add_module('style_loss_' + str(i), style_loss)
                style_loss_list.append(style_loss)

            i += 1
        if isinstance(layer, nn.MaxPool2d):
            name = 'pool_' + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = 'relu' + str(i)
            model.add_module(name, layer)

    return model, style_loss_list, content_loss_list

def get_input_param_optimier(input_img):
    """
    input_img is a Variable
    """
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

def run_style_transfer(content_img, style_img, input_img, num_epoches=300):
    print('Building the style transfer model..')
    model, style_loss_list, content_loss_list = get_style_model_and_loss(
        style_img, content_img)
    input_param, optimizer = get_input_param_optimier(input_img)

    print('Opimizing...')
    epoch = [0]
    while epoch[0] < num_epoches:

        def closure():
            input_param.data.clamp_(0, 1)

            model(input_param)
            style_score = 0
            content_score = 0

            optimizer.zero_grad()
            for sl in style_loss_list:
                style_score += sl.backward()
            for cl in content_loss_list:
                content_score += cl.backward()

            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print('run {}'.format(epoch))
                print('Style Loss: {:.4f} Content Loss: {:.4f}'.format(
                    style_score.data.item(), content_score.data.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

        input_param.data.clamp_(0, 1)

    return input_param.data


from torch.autograd import Variable

# style_img = image_loader(dirs_+"152.jpg")
# style_img = image_loader(dirs_+"../../arts.png")
style_img = image_loader(dirs_+"../../arts2.jpg")
style_img = Variable(style_img).to(device)
content_img = image_loader("./watch/data/20190726123116.jpg")
# content_img = image_loader("./watch/data/20190726123139.jpg")
content_img = Variable(content_img).to(device)

input_img = content_img.clone()

out = run_style_transfer(content_img, style_img, input_img, num_epoches=400)

image_shower(out.cpu(), "./watch/data/20190726123116.jpg")









