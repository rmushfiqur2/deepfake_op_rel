import torch
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
from .archs.resnet import resnet50
remove_classifier=True

def freeze_layers(model, num_lst_layers=0):
    for name, param in model.named_parameters():
        if(name.split('.')[1] == "layer4" or name.split('.')[0] == "additional_layer"):
            param.requires_grad = True
            #print(f"Parameter name: {name}, Size: {param.size()}")
        else:
            param.requires_grad = False


# Combine the existing model and the additional layer into a new custom model
class CustomModel(nn.Module):
    def __init__(self, existing_model, additional_layer):
        super(CustomModel, self).__init__()
        self.existing_model = existing_model
        self.additional_layer = additional_layer

    def forward(self, x):
        x = self.existing_model(x)
        x = self.additional_layer(x)
        return x


def get_resnet(device):
    init_weights = torch.load('models/flr_r50_vgg_face.pth', map_location=device)['state_dict']
    converted_weights = {k.replace('module.base_net.', ''):v for k, v in init_weights.items()}

    model_resnet = resnet50(remove_classifier=True)
    model_resnet.load_state_dict(converted_weights, strict=False)

    # Create instances of the existing model and the additional layer
    existing_model = model_resnet
    additional_layer = nn.Linear(2048, 1792)
    custom_model = CustomModel(existing_model, additional_layer)
    custom_model.to(device)
    freeze_layers(custom_model)
    return custom_model