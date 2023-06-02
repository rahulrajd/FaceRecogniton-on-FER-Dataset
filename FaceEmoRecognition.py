import torch as t
import torch.nn as nn
import torch.nn.functional as F


class FacialEmoRecognition(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 6, 3, 1)
        self.layer2 = nn.Conv2d(6, 16, 3, 1)
        self.layer3 = nn.Conv2d(16, 6, 3, 1)
        self.f1_layer = nn.Linear(6 * 4 * 4, 128)
        self.f2_layer = nn.Linear(128, 48)
        self.f_layer = nn.Linear(48, 7)

    def forward(self, image):
        image = F.relu(self.layer1(image))
        image = F.max_pool2d(image, 2)
        image = F.relu(self.layer2(image))
        image = F.max_pool2d(image, 2)
        image = F.relu(self.layer3(image))
        image = F.max_pool2d(image, 2)
        # print(image.size())
        image = image.view(-1, 6 * 4 * 4)
        image = F.relu(self.f1_layer(image))
        image = F.relu(self.f2_layer(image))
        image = self.f_layer(image)
        return F.log_softmax(image)
