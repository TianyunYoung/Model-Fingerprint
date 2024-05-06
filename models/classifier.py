
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from utils.common import get_input_data

class UnetResNet50(nn.Module):

    def __init__(self, class_num, backbone='resnet50', encoder_weights='imagenet'):
        super(UnetResNet50, self).__init__()

        aux_params = dict(
            classes=class_num, 
        )
        self.model = smp.Unet(backbone, encoder_weights=encoder_weights,\
                              classes=3, activation=None, aux_params=aux_params)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input, denoise_func=None):

        input = get_input_data(input, denoise_func)  

        embedding = self.model.encoder(input)[-1]
        feature = self.pool(embedding)
        feature = feature.view(feature.shape[0], -1)

        cls_prob = self.model(input)[1]

        return cls_prob, feature


    