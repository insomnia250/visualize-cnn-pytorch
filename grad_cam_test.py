# coding=utf-8
from __future__ import print_function

from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

# Many are borrowed from https://github.com/kazuto1011/grad-cam-pytorch.
# Add forward hook and guided-backprop from target layer and filter, instead of final one hot.

class PropagationBase(object):
    def __init__(self, model, target_layer, cuda):
        self.model = model
        self.model.eval()
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()
        self.target_layer = target_layer
        self.internal = OrderedDict()
        self.all_grads = OrderedDict()
        self.set_hook_func()

    def set_hook_func(self):
        raise NotImplementedError

    def encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.cuda() if self.cuda else one_hot

    def load_image(self, filename, transform):
        print('loading image:', filename)
        self.raw_image = cv2.imread(filename)[:, :, ::-1]
        self.raw_image = cv2.resize(self.raw_image, (224, 224))
        image = transform(self.raw_image).unsqueeze(0)
        image = image.cuda() if self.cuda else image
        self.image = Variable(image, volatile=False, requires_grad=True)

    def forward(self):
        self.model(self.image)

    def backward(self, target_layer, target_filter):
        self.model.zero_grad()
        for key, value in self.internal.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        temp = np.zeros(value.size(), dtype=np.float32)
                        temp[target_filter] = 1
                        mask = Variable(torch.from_numpy(temp), requires_grad=True)
                        # out = mask*value
                        # out.backward(retain_graph=True)
                        value.backward(gradient=mask * value, retain_graph=True)

                        print('"{}"out size:{}'.format(module[0], value.size()))
                        return self.image.grad.cpu().data.numpy()[0].transpose(1, 2, 0)
        raise ValueError('invalid layer name: {}'.format(target_layer))

    def find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('invalid layer name: {}'.format(target_layer))


class BackPropagation(PropagationBase):
    def set_hook_func(self):
        print('setting hook -- BackPropagation')

        def func_f(module, input, output):
            self.internal[id(module)] = output[0].cpu()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_in[0].cpu()

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)
            module[1].register_forward_hook(func_f)

    def generate(self):
        output = self.find(self.all_grads, self.target_layer)
        return output.data.numpy()[0].transpose(1, 2, 0)

    def save(self, filename, data):
        data -= data.min()
        data /= data.max()
        data = np.uint8(data * 255)
        cv2.imwrite(filename, data)


class GuidedBackPropagation(BackPropagation):
    def set_hook_func(self):
        print('setting hook -- GuidedBackPropagation')

        def func_f(module, input, output):
            self.internal[id(module)] = output[0].cpu()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_in[0].cpu()

            # Cut off negative input gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)
            module[1].register_forward_hook(func_f)
