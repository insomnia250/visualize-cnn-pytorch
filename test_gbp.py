# coding: utf-8


from __future__ import print_function

import argparse

import cv2
import numpy as np
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
from grad_cam_test import BackPropagation, GuidedBackPropagation
import torch



img_path = './samples/cat_dog.png'

# Load the synset words
idx2cls = list()
with open('samples/synset_words.txt') as lines:
    for line in lines:
        line = line.strip().split(' ', 1)[1]
        line = line.split(', ', 1)[0].replace(' ', '_')
        idx2cls.append(line)

print('Loading a model...')
model = torchvision.models.vgg19(pretrained=True)
for module in model.named_modules():
    print(module)
    print('==='*20)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


print('\nGuided Backpropagation')
gbp = GuidedBackPropagation(model=model,
                            target_layer='layer4.2.conv2',
                            cuda=1)
gbp.load_image(img_path, transform)
gbp.forward()
filter_vis = gbp.backward(target_layer='features.0',target_filter=0)


img = cv2.imread(img_path)
plt.figure(figsize=(10, 10), facecolor='w')
plt.subplot(2, 2, 1)
plt.title('input')
plt.imshow(img[:,:,::-1])
plt.subplot(2, 2, 2)
plt.title('abs. saliency')
plt.imshow(np.abs(filter_vis).max(axis=-1), cmap='gray')
plt.subplot(2, 2, 3)
plt.title('pos. saliency')
plt.imshow((np.maximum(0, filter_vis) / filter_vis.max()))
plt.subplot(2, 2, 4)
plt.title('neg. saliency')
plt.imshow((np.maximum(0, -filter_vis) / -filter_vis.min()))
plt.show()

