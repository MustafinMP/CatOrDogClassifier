import tkinter
from tkinter.filedialog import askopenfilename

from PIL import Image, ImageTk

import torch
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


def predict(img):
    img = transforms(img).unsqueeze(0)

    p = model(img)
    p2 = torch.argmax(p, dim=1)
    return p2[0].item()


root = tkinter.Tk()
frame = tkinter.Frame(root)
frame.grid()

canvas = tkinter.Canvas(root, height=550, width=550)

image = Image.open(askopenfilename())
p = predict(image)
image.thumbnail((512, 512))
photo = ImageTk.PhotoImage(image)
tk_image = canvas.create_image(0, 0, anchor='nw', image=photo)
canvas.grid(row=2, column=1)

canvas.create_text(256, 512 + 16, text='Вероятно, это собака' if p else 'Вероятно, это кот')
root.mainloop()
