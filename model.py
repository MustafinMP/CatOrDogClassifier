import json
import os

import torch
from PIL import Image
from torchvision import models
import torchvision.transforms.v2 as tfs
from tqdm import tqdm
from torch import optim, nn
from torch.utils import data

# https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification?select=train


class CatAndDogDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, 'train' if train else 'test')
        self.transform = transform

        with open(os.path.join(self.path, 'format.json'), 'r') as fp:
            self.format = json.load(fp)

        self.length = 0
        self.files = []
        self.targets = torch.eye(2)

        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)
            self.length += len(list_files)
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

    def __getitem__(self, item):
        path_file, target = self.files[item]
        t = self.targets[target]
        img = Image.open(path_file).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, t

    def __len__(self):
        return self.length


resnet50_weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=resnet50_weights)
model.requires_grad_(False)
model.fc = nn.Linear(512 * 4, 2)
model.requires_grad_(True)
transforms = tfs.Compose([
    tfs.ToImage(),
    tfs.ToDtype(torch.float32, scale=True),
    tfs.Resize(512),
    tfs.CenterCrop(512),
    resnet50_weights.transforms()
])


d_train = CatAndDogDataset('dataset', transform=transforms, train=True)
train_data = data.DataLoader(d_train, batch_size=4, shuffle=True)

optimizer = optim.Adam(params=model.fc.parameters(), lr=0.001, weight_decay=0.001)
loss_function = nn.CrossEntropyLoss()

model.train()
epochs = 0

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0
    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_function(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f'Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}')

# st = model.state_dict()
# torch.save(st, 'cat_and_dog_classifier.tar')
st = torch.load('cat_and_dog_classifier.tar')
model.load_state_dict(st)

d_test = CatAndDogDataset('dataset', transform=transforms, train=False)
test_data = data.DataLoader(d_test, batch_size=8, shuffle=False)

Q = 0
P = 0
count = 1
model.eval()

test_tqdm = tqdm(test_data, leave=True)
for x_test, y_test in test_tqdm:
    with torch.no_grad():
        p = model(x_test)
        p2 = torch.argmax(p, dim=1)
        y = torch.argmax(y_test, dim=1)
        P += torch.sum(p2 == y).item()
        Q += loss_function(p, y_test).item()
        count += 1

Q /= count
P /= len(d_test)
print(Q)
print(P)

img = Image.open('dataset/test/dogs/dog_28.jpg')
img = transforms(img).unsqueeze(0)

p = model(img)
p2 = torch.argmax(p, dim=1)
print(p2)
