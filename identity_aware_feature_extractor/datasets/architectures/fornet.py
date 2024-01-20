"""
Video Face Manipulation Detection Through Ensemble of CNNs

Image and Sound Processing Lab - Politecnico di Milano

Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini
"""
from collections import OrderedDict

import torch
from efficientnet_pytorch import EfficientNet
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms

from . import externals

"""
Feature Extractor
"""


class FeatureExtractor(nn.Module):
    """
    Abstract class to be extended when supporting features extraction.
    It also provides standard normalized and parameters
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


"""
EfficientNet
"""


class EfficientNetGen(FeatureExtractor):
    def __init__(self, model: str):
        super(EfficientNetGen, self).__init__()

        self.efficientnet = EfficientNet.from_pretrained(model)
        self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, 1)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        return x

class EfficientNetB4(EfficientNetGen):
    def __init__(self):
        super(EfficientNetB4, self).__init__(model='efficientnet-b4')


"""
EfficientNetAutoAtt
"""


class EfficientNetAutoAtt(EfficientNet):
    def init_att(self, model: str, width: int):
        """
        Initialize attention
        :param model: efficientnet-bx, x \in {0,..,7}
        :param depth: attention width
        :return:
        """
        if model == 'efficientnet-b4':
            self.att_block_idx = 9
            if width == 0:
                self.attconv = nn.Conv2d(kernel_size=1, in_channels=56, out_channels=1)
            else:
                attconv_layers = []
                for i in range(width):
                    attconv_layers.append(
                        ('conv{:d}'.format(i), nn.Conv2d(kernel_size=3, padding=1, in_channels=56, out_channels=56)))
                    attconv_layers.append(
                        ('relu{:d}'.format(i), nn.ReLU(inplace=True)))
                attconv_layers.append(('conv_out', nn.Conv2d(kernel_size=1, in_channels=56, out_channels=1)))
                self.attconv = nn.Sequential(OrderedDict(attconv_layers))
        else:
            raise ValueError('Model not valid: {}'.format(model))

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:

        # Placeholder
        att = None

        # Stem
        x = self._swish(self._bn0(self._conv_stem(x)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self.att_block_idx:
                att = torch.sigmoid(self.attconv(x))
                break

        return att

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self._swish(self._bn0(self._conv_stem(x)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self.att_block_idx:
                att = torch.sigmoid(self.attconv(x))
                x = x * att

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x


class EfficientNetGenAutoAtt(FeatureExtractor):
    def __init__(self, model: str, width: int):
        super(EfficientNetGenAutoAtt, self).__init__()

        self.efficientnet = EfficientNetAutoAtt.from_pretrained(model)
        self.efficientnet.init_att(model, width)
        self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, 1)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        return x

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet.get_attention(x)


class EfficientNetAutoAttB4(EfficientNetGenAutoAtt):
    def __init__(self):
        super(EfficientNetAutoAttB4, self).__init__(model='efficientnet-b4', width=0)


"""
Xception
"""


class Xception(FeatureExtractor):
    def __init__(self):
        super(Xception, self).__init__()
        self.xception = externals.xception()
        self.xception.last_linear = nn.Linear(2048, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.xception.features(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.xception.forward(x)


"""
Siamese tuning
"""


class SiameseTuning(FeatureExtractor):
    def __init__(self, feat_ext: FeatureExtractor, num_feat: int, lastonly: bool = True):
        super(SiameseTuning, self).__init__()
        self.feat_ext = feat_ext()
        if not hasattr(self.feat_ext, 'features'):
            raise NotImplementedError('The provided feature extractor needs to provide a features() method')
        self.lastonly = lastonly
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=num_feat),
            nn.Linear(in_features=num_feat, out_features=1),
        )

    def features(self, x):
        x = self.feat_ext.features(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lastonly:
            with torch.no_grad():
                x = self.features(x)
        else:
            x = self.features(x)
        x = self.classifier(x)
        return x

    def get_trainable_parameters(self):
        if self.lastonly:
            return self.classifier.parameters()
        else:
            return self.parameters()


class EfficientNetB4ST(SiameseTuning):
    def __init__(self):
        super(EfficientNetB4ST, self).__init__(feat_ext=EfficientNetB4, num_feat=1792, lastonly=True)


class EfficientNetAutoAttB4ST(SiameseTuning):
    def __init__(self):
        super(EfficientNetAutoAttB4ST, self).__init__(feat_ext=EfficientNetAutoAttB4, num_feat=1792, lastonly=True)


class XceptionST(SiameseTuning):
    def __init__(self):
        super(XceptionST, self).__init__(feat_ext=Xception, num_feat=2048, lastonly=True)


class IdentityAwareNeuralNetwork(nn.Module):
    def __init__(self, in_features, in_identity):
        super().__init__()
        self.in_features = in_features
        self.in_identity = in_identity
        self.linear_norm_stack = nn.Sequential(
            nn.Linear(in_features+in_identity, 50),
            #nn.ReLU(),
            #nn.BatchNorm1d(50),
        )
        self.linear_identity_stack = nn.Sequential(
            nn.Linear(50+in_identity, 50),
            #nn.ReLU(),
            #nn.BatchNorm1d(50),
        )

    def forward(self, x):
        #traces_processed = self.linear_norm_stack(x[:, :self.in_features])
        traces_processed = self.linear_norm_stack(x)
        #input_merged = torch.cat((traces_processed, x[:, self.in_features:]), dim=1)
        #final_traces = self.linear_identity_stack(input_merged)
        return traces_processed

class IdentityAwareNeuralNetworkNotShared(nn.Module):
    def __init__(self, in_features, in_identity):
        super().__init__()
        self.in_features = in_features
        self.in_identity = in_identity
        self.linear_norm_stack = nn.Sequential(
            nn.Linear(in_features, 50),
            #nn.ReLU(),
            #nn.BatchNorm1d(50),
        )
        self.linear_norm_stack2 = nn.Sequential(
            nn.Linear(in_features, 50),
            #nn.ReLU(),
            #nn.BatchNorm1d(50),
        )
        self.linear_identity_stack1 = nn.Sequential(
            nn.Linear(50, 50),
            #nn.ReLU(),
            #nn.BatchNorm1d(50),
        )
        self.linear_identity_stack2 = nn.Sequential(
            nn.Linear(50, 50),
            #nn.ReLU(),
            #nn.BatchNorm1d(50),
        )
        self.linear_identity_decoder = nn.Sequential(
            nn.Linear(self.in_identity, self.in_features),
        )
        self.linear_identity_decoder2 = nn.Sequential(
            nn.Linear(self.in_identity, self.in_features),
        )

    def forward(self, x):
        x1, x2 = x[0], x[1]
        #traces_processed1 = self.linear_norm_stack(x1[:, :self.in_features])
        #traces_processed2 = self.linear_norm_stack(x2[:, :self.in_features])
        #input_merged1 = torch.cat((traces_processed1, x1[:, self.in_features:]), dim=1)
        #input_merged2 = torch.cat((traces_processed2, x2[:, self.in_features:]), dim=1)
        #final_traces1 = self.linear_identity_stack1(traces_processed1)
        #final_traces2 = self.linear_identity_stack2(traces_processed2)
        final_decoded1 = self.linear_identity_decoder(x1[:, self.in_features:])
        final_decoded2 = self.linear_identity_decoder(x2[:, self.in_features:])
        input_modified1 = final_decoded1 + x1[:, :self.in_features]
        input_modified2 = final_decoded2 + x2[:, :self.in_features]
        final_traces1 = self.linear_norm_stack(input_modified1)
        final_traces2 = self.linear_norm_stack2(input_modified2)
        return [final_traces1, final_traces2]