import os
import random
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import torchvision.models as models
import torch.nn as nn
import __main__
import torch.nn.functional as F

data_dir = 'dataset/'
classes = os.listdir(data_dir)

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
dataset = ImageFolder(data_dir, transform=transformations)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch + 1, result['train_loss'], result['val_loss'], result['val_acc']))


class DenseNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.densenet121(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.classifier.in_features
        self.network.classifier = nn.Linear(num_ftrs, len(dataset.classes))

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


# porting gpu/cpu
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device_type):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device_type) for x in data]
    return data.to(device_type, non_blocking=True)


device = get_default_device()
print(f'Using {device}.')

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]


setattr(__main__, "DenseNet", DenseNet)
setattr(__main__,'ResNet',ResNet)
ResNet50_model = torch.load('ResNet50_model_v2.2.pt', map_location='cpu')
DenseNet121_model = torch.load('DenseNet121_model_v2.2.pt', map_location='cpu')
DenseNet201_model = torch.load('DenseNet201_model_v2.2.pt', map_location='cpu')


# 圖片辨識
def predict_external_image(image_name, loaded_model):
    image = Image.open(Path('./' + image_name))
    example_image = transformations(image)
    result = predict_image(example_image, loaded_model)
    return result


# 三模型投票
def vote(model1, model2, model3):
    if model1 == model2:
        return model1
    elif model1 == model3:
        return model1
    elif model2 == model3:
        return model2
    else:
        result = random.choice([model1,model2,model3])
        return result


# 辨識結果
def predict_result(image_name):
    resnet50 = predict_external_image(image_name, ResNet50_model)
    densenet121 = predict_external_image(image_name, DenseNet121_model)
    densenet201 = predict_external_image(image_name, DenseNet201_model)
    print(f"ResNet50：{resnet50}；DenseNet121：{densenet121}；DenseNet201：{densenet201}")
    result = vote(resnet50, densenet121, densenet201)
    print("The image resembles", result + ".")
    return result


# 可回收日
def days(result):
    solid = ['paper_container', 'glass', 'plastic', 'metal']
    flat = ['clothes', 'cardboard', 'paper', 'plastic_bag']
    another = ['battery', 'biological']
    trash = ['shoes', 'trash']
    if result in solid:
        return '二、四、六'
    elif result in flat:
        return '一、五'
    elif result in another:
        return '一、二、四、五、六'
    elif result in trash:
        return '不可回收'
