import json
import os

import torch
from PIL import Image
from torchvision import models
import torchvision.transforms.v2 as tfs
from torch import nn

resnet50_weights = models.ResNet50_Weights.DEFAULT

transforms = tfs.Compose([
    tfs.ToImage(),
    tfs.ToDtype(torch.float32, scale=True),
    tfs.Resize(512),
    tfs.CenterCrop(512),
    resnet50_weights.transforms()
])

model = models.resnet50(weights=resnet50_weights)
model.requires_grad_(False)
model.fc = nn.Linear(512 * 4, 2)
model.requires_grad_(True)

st = torch.load('cat_and_dog_classifier.tar')
model.load_state_dict(st)

model.eval()


def predict(path):
    img = Image.open(path)
    img = transforms(img).unsqueeze(0)

    p = model(img)
    p2 = torch.argmax(p, dim=1)
    return p2[0].item()


# dataset/test/dogs/dog_28.jpg
while inp_path := input('Укажите путь к файлу с изображением >> '):
    print('Вероятно, это собака' if predict(inp_path) else 'Вероятно, это кот')
