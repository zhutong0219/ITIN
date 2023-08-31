import os
import time
import random
import numpy as np
from tqdm import tqdm
from math import sqrt
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from transformers import BertTokenizer, BertModel
import cv2
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import collections
import torchtext.vocab as Vocab
import torchvision.models as models
from sklearn.metrics import f1_score
from torchvision import transforms as tfs
from PIL import Image


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.drop=nn.Dropout(0.5)
        self.fc4 = nn.Linear(512, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
 
        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

#####################################################################################################################


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def sum_attention(nnet, query, value, dropout=None):
    scores = nnet(query).transpose(-2, -1)
    # if mask is not None:
    #     scores.data.masked_fill_(mask.data.eq(0), -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def qkv_attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(d_k)
    # if mask is not None:
    #     scores.data.masked_fill_(mask.data.eq(0), -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class SummaryAttn(nn.Module):
    def __init__(self, dim, num_attn, dropout, is_cat=False):
        super(SummaryAttn, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, num_attn),
        )
        self.h = num_attn
        self.is_cat = is_cat
        self.attn = None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, query, value):
        # if mask is not None:
        #     mask = mask.unsqueeze(-2)
        batch = query.size(0)

        weighted, self.attn = sum_attention(self.linear, query, value, dropout=self.dropout)
        weighted = weighted if self.is_cat else weighted.mean(dim=-2)

        return weighted


def EncoderImage(data_name, img_dim, embed_size, finetune=False,
                 cnn_type='resnet152', no_imgnorm=False,
                 self_attention=False):

    img_enc = EncoderImagePrecomp(img_dim, embed_size, no_imgnorm,
                                  self_attention)

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False,
                 self_attention=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.self_attention = self_attention

        self.fc = nn.Linear(img_dim, embed_size)
        if self_attention:
            self.attention_layer = SummaryAttn(embed_size, 1, -1)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        if self.self_attention:
            features = self.attention_layer(features, features)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        # self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.out1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 64)
        self.out2 = nn.Linear(64, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [batch size, sent len, emb dim]

        outs, hidden = self.rnn(embedded)
        outs = (outs[:, :, :outs.size(2) // 2] + outs[:, :, outs.size(2) // 2:]) / 2
        o = torch.mean(outs, dim=1)

        # hidden = [n layers * n directions, batch size, emb dim]
        # _ = [batch size, sent len, hid dim * 2]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        # output = self.out(hidden)
        output = F.relu(self.out1(hidden))
        output = self.out2(output)

        # output = [batch size, out dim]

        return outs, o


class GatedFusionNew(nn.Module):
    def __init__(self, dim, num_attn, dropout=0.01, reduce_func="self_attn", fusion_func="concat"):
        super(GatedFusionNew, self).__init__()
        self.dim = dim
        self.h = num_attn

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

        self.reduce_func = reduce_func
        self.fusion_func = fusion_func

        self.img_key_fc = nn.Linear(dim, dim, bias=False)
        self.txt_key_fc = nn.Linear(dim, dim, bias=False)

        self.img_query_fc = nn.Linear(dim, dim, bias=False)
        self.txt_query_fc = nn.Linear(dim, dim, bias=False)

        self.weighted_img_key_fc = nn.Linear(dim, dim, bias=False)
        self.weighted_txt_key_fc = nn.Linear(dim, dim, bias=False)

        self.weighted_img_query_fc = nn.Linear(dim, dim, bias=False)
        self.weighted_txt_query_fc = nn.Linear(dim, dim, bias=False)

        in_dim = dim
        if fusion_func == "sum":
            in_dim = dim
        elif fusion_func == "concat":
            in_dim = 2 * dim
        else:
            raise NotImplementedError('Only support sum or concat fusion')

        self.fc_1 = nn.Sequential(
            nn.Linear(in_dim, dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), )
        self.fc_2 = nn.Sequential(
            nn.Linear(in_dim, dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), )
        self.fc_out = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

        if reduce_func == "mean":
            self.reduce_layer = torch.mean
        elif reduce_func == "self_attn":
            # self.reduce_layer_1 = SummaryAttn(dim, num_attn, dropout, is_cat=True)
            # self.reduce_layer_2 = SummaryAttn(dim, num_attn, dropout, is_cat=True)
            self.final_reduce_1 = SummaryAttn(dim, num_attn, dropout)
            self.final_reduce_2 = SummaryAttn(dim, num_attn, dropout)

        self.init_weights()
        print("GatedFusion module init success!")

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.dim +
                                  self.dim)
        self.img_key_fc.weight.data.uniform_(-r, r)
        self.txt_key_fc.weight.data.uniform_(-r, r)
        self.fc_1[0].weight.data.uniform_(-r, r)
        # self.fc_1[0].bias.data.fill_(0)
        self.fc_2[0].weight.data.uniform_(-r, r)
        # self.fc_2[0].bias.data.fill_(0)
        self.fc_out[0].weight.data.uniform_(-r, r)
        self.fc_out[0].bias.data.fill_(0)
        self.fc_out[3].weight.data.uniform_(-r, r)
        self.fc_out[3].bias.data.fill_(0)

    def forward(self, v1, v2, get_score=True, keep=None):
        k1 = self.img_key_fc(v1)
        k2 = self.txt_key_fc(v2)
        q1 = self.img_query_fc(v1)
        q2 = self.txt_query_fc(v2)
        batch_size_v1 = v1.size(0)
        batch_size_v2 = v2.size(0)

        weighted_v1, attn_1 = qkv_attention(q2, k1, v1)

        weighted_v2, attn_2 = qkv_attention(q1, k2, v2)

        weighted_v2_q = self.weighted_txt_query_fc(weighted_v2)
        weighted_v2_k = self.weighted_txt_key_fc(weighted_v2)

        weighted_v1_q = self.weighted_img_query_fc(weighted_v1)
        weighted_v1_k = self.weighted_img_key_fc(weighted_v1)

        fused_v1, _ = qkv_attention(weighted_v2_q, weighted_v2_k, weighted_v2)
   
        fused_v2, _ = qkv_attention(weighted_v1_q, weighted_v1_k, weighted_v1)

        fused_v1 = l2norm(fused_v1)
        fused_v2 = l2norm(fused_v2)

        gate_v1 = F.sigmoid((v1 * fused_v1).sum(dim=-1)).unsqueeze(-1)
        gate_v2 = F.sigmoid((v2 * fused_v2).sum(dim=-1)).unsqueeze(-1)

        if self.fusion_func == "sum":
            co_v1 = (v1 + fused_v1) * gate_v1
            co_v2 = (v2 + fused_v2) * gate_v2
        elif self.fusion_func == "concat":
            co_v1 = torch.cat((v1, fused_v1), dim=-1) * gate_v1
            co_v2 = torch.cat((v2, fused_v2), dim=-1) * gate_v2

        co_v1 = self.fc_1(co_v1) + v1
        co_v2 = self.fc_2(co_v2) + v2

        if self.reduce_func == "self_attn":
            co_v1 = self.final_reduce_1(co_v1, co_v1)
            co_v2 = self.final_reduce_2(co_v2, co_v2)
            co_v1 = l2norm(co_v1)
            co_v2 = l2norm(co_v2)
        else:
            co_v1 = self.reduce_func(co_v1, dim=-2)
            co_v2 = self.reduce_func(co_v2, dim=-2)
            co_v1 = l2norm(co_v1)
            co_v2 = l2norm(co_v2)

        return co_v1, co_v2


class SimLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, inner_dim=0, loss_func="BCE"):
        super(SimLoss, self).__init__()
        self.margin = margin
        self.measure = measure
        # if measure == 'gate_fusion_new':
        self.sim = GatedFusionNew(inner_dim, 4, 0.0)

        self.loss_func = loss_func
        self.max_violation = max_violation

    def forward(self, im, s, get_score=False, keep="words"):
        cur_im = im
        cur_s = s
        drive_num = 1

        fused_v1, fused_v2 = self.sim(cur_im, cur_s, keep=keep)

        return fused_v1,fused_v2


class CAMP(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt,bert,
                             HIDDEN_DIM,
                             OUTPUT_DIM,
                             N_LAYERS,
                             BIDIRECTIONAL,
                             DROPOUT):
        # Build Models
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    opt.finetune, opt.cnn_type,
                                    no_imgnorm=opt.no_imgnorm,
                                    self_attention=opt.self_attention)

        self.txt_enc = BERTGRUSentiment(bert,
                             HIDDEN_DIM,
                             OUTPUT_DIM,
                             N_LAYERS,
                             BIDIRECTIONAL,
                             DROPOUT)

        if opt.cross_model:
            self.criterion = SimLoss(margin=opt.margin,
                                     measure=opt.measure,
                                     max_violation=opt.max_violation,
                                     inner_dim=opt.embed_size)
        else:
            self.criterion = SimLoss(margin=opt.margin,
                                     measure=opt.measure,
                                     max_violation=opt.max_violation)

        if torch.cuda.is_available():
            self.img_enc = nn.DataParallel(self.img_enc)
            self.txt_enc = nn.DataParallel(self.txt_enc)
            self.img_enc.cuda()
            self.txt_enc.cuda()
            if opt.cross_model:
                self.criterion.sim = nn.DataParallel(self.criterion.sim)
                self.criterion.sim.cuda()
            cudnn.benchmark = True

        print("Encoders init OK!")
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.module.fc.parameters())
        if opt.self_attention:
            params += list(self.img_enc.module.attention_layer.parameters())

        if opt.finetune:
            params += list(self.img_enc.module.cnn.parameters())

        if opt.cross_model:
            params += list(self.criterion.sim.parameters())

        if opt.measure == "gate_fusion" and not opt.finetune_gate:
            print("Only fc layers and final aggregation layers optimized.")
            params = list(self.criterion.sim.module.fc_1.parameters())
            params += list(self.criterion.sim.module.fc_2.parameters())
            params += list(self.criterion.sim.module.fc_out.parameters())
            params += list(self.criterion.sim.module.reduce_layer_1.parameters())
            params += list(self.criterion.sim.module.reduce_layer_2.parameters())

        if opt.measure == "gate_fusion_new" and not opt.finetune_gate:
            print("Only fc layers and final aggregation layers optimized.")
            params = list(self.criterion.sim.module.fc_1.parameters())
            params += list(self.criterion.sim.module.fc_2.parameters())
            params += list(self.criterion.sim.module.fc_out.parameters())
            params += list(self.criterion.sim.module.final_reduce_1.parameters())
            params += list(self.criterion.sim.module.final_reduce_2.parameters())

        if opt.embed_mask:
            self.embed_mask = np.load(opt.embed_mask)
        else:
            self.embed_mask = None

        self.params = params
        self.Eiters = 0
        print("Model init OK!")

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        if self.opt.cross_model:
            state_dict += [self.criterion.sim.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict[0].items():
            new_state_dict[k] = v
        self.img_enc.load_state_dict(new_state_dict, strict=True)

        new_state_dict = OrderedDict()
        for k, v in state_dict[1].items():
            # name = k.replace('module.', '') # remove `module.`
            new_state_dict[k] = v
        self.txt_enc.load_state_dict(new_state_dict, strict=True)
        new_state_dict = OrderedDict()

        if len(state_dict) > 2:
            new_state_dict = OrderedDict()
            for k, v in state_dict[2].items():
                # name = k.replace('module.', '') # remove `module.`
                new_state_dict[k] = v
            self.criterion.sim.load_state_dict(new_state_dict, strict=False)
            new_state_dict = OrderedDict()

    def forward_emb(self, images, captions, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)
        cap_emb, _ = self.txt_enc(captions)

        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, **kwargs):

        fused_v1, fused_v2 = self.criterion(img_emb, cap_emb)

        return fused_v1, fused_v2


class catNet(nn.Module):
    def __init__(self, model, bertmodel, imagenet):
        super(catNet, self).__init__()
        # defining layers in catNet
        self.model = model
        self.bertmodel = bertmodel
        self.imagenet = imagenet
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, 3)
        self.fc3 = nn.Linear(6, 3)
        self.fc4 = nn.Linear(1536, 64)

    def forward(self, x1, x2,x3):
        img_emb, cap_emb = self.model.forward_emb(x1, x2, volatile=True)
        _, bertout = self.bertmodel(x2)
        imageglobal = self.imagenet(x3)
        
        fused_v1, fused_v2 = self.model.forward_loss(img_emb, cap_emb)

        # fused = torch.cat((fused_v1, fused_v2), 1)
        # fused = 0.5 * fused_v1 + 0.5 * fused_v2
        fused = fused_v1
        h = torch.cat((fused, imageglobal), 1)
        # h = self.drop(h)
        h = F.relu(self.fc4(h))
        h = self.fc2(h)
        h2 = torch.cat((fused, bertout), 1)
        h2 = F.relu(self.fc1(h2))
        h2 = self.fc2(h2)
        h = 0.2*h + 0.8*h2
        # h = torch.cat((h,h2),1)
        # h = self.fc3(h)
       
        return h


# 加载数据
ids = []
labels = []
with open('./Test_label.txt') as f:
    for line in f:
        text_id, label = line.split()
        label = int(label)
        ids.append(text_id)
        labels.append(label)

mvsa_data = []
image = []
imageglobal = []
trans_ops = tfs.Compose([
    tfs.Resize(256),
    tfs.CenterCrop(224),
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])
for fid in tqdm(ids):
    img_feature = np.load(os.path.join('./TestSet/bottom', str(fid) + '.npz'))
    img_feature = img_feature['x']
    image.append(img_feature)

    imgglobal_feature = Image.open(os.path.join('./TestSet/image', str(fid) + '.jpg')).convert('RGB')
    imgglobal_feature = trans_ops(imgglobal_feature)
    imageglobal.append(imgglobal_feature)

    filepath = os.path.join('./TestSet/text', str(fid) + '.txt')
    with open(filepath, 'rb') as f:
        review = f.read()
        review = review.decode('ascii', 'ignore')
        mvsa_data.append(review)

sentencses = ['[CLS] ' + caption + ' [SEP]' for caption in mvsa_data]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_sents = [tokenizer.tokenize(sent) for sent in sentencses]

MAX_LEN=100

input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]

def pad(x):
    if len(x) > MAX_LEN:
        x = x[:MAX_LEN]
    else:
        x = x + [0] * (MAX_LEN - len(x))
    # return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))
    return x

# input_ids=keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
input_ids = [pad(input_id) for input_id in input_ids]

idx_test = [i for i in range(len(ids))]
test_texts = [input_ids[i] for i in idx_test]
test_labels = [labels[i] for i in idx_test]
test_images = [image[i] for i in idx_test]
test_imageglobal = [imageglobal[i] for i in idx_test]

test_texts = torch.tensor(test_texts)
test_labels = torch.tensor(test_labels)
test_images = torch.tensor(test_images)
test_imageglobal = torch.stack(test_imageglobal,0)

test_data = TensorDataset(test_texts, test_images, test_imageglobal,test_labels)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)


def test(mymodel,test_loader,device,test_data_length):
    mymodel.eval()
    test_correct = 0
    F1_ma = 0
    F1_mi = 0

    with torch.no_grad():
        for batch_idx, (txt, image,imageglobal, target) in enumerate(test_loader):
            txt, image, imageglobal, target = txt.to(device),image.to(device),imageglobal.to(device),target.to(device)
            # output = model(txt, attention_mask=mask)
            output = mymodel(image, txt, imageglobal)
            target = target.squeeze()
     
            _, txt_preds = torch.max(output, 1)
            test_correct += torch.sum(txt_preds == target.data)

    val_acc_image = test_correct.double() / test_data_length
    print('Test_image_acc:{:.4f}'.format(val_acc_image))
 
    return val_acc_image

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")

    HIDDEN_DIM = 1024
    OUTPUT_DIM = 3
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    lr = 0.001
    weight_decay = 1e-5
    epochs = 200

    checkpoint = torch.load("./checkpoint_59.pth.tar")
    opt = checkpoint['opt']

    opt.distributed = False
    opt.use_all = True
    opt.instance_loss = False
    opt.attention = False

    bert = BertModel.from_pretrained('bert-base-uncased')
    bertmodel = BERTGRUSentiment(bert,HIDDEN_DIM,OUTPUT_DIM,N_LAYERS,BIDIRECTIONAL,DROPOUT)

    for name, param in bertmodel.named_parameters():
        if name.startswith('bert'):
            param.requires_grad = False

    imagenet = resnet18().to(device)

    model = CAMP(opt, bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

    mymodel = catNet(model=model, bertmodel=bertmodel, imagenet=imagenet)

    resnet_dict = models.resnet18(pretrained=True).state_dict()
    image_dict = mymodel.imagenet.state_dict()
    resnet_dict = {k: v for k, v in resnet_dict.items() if k in image_dict}
    image_dict.update(resnet_dict)
    mymodel.imagenet.load_state_dict(image_dict)
 
    mymodel.load_state_dict(torch.load("./ITIN.pt"))

    mymodel = mymodel.to(device)

    val_acc_image = test(mymodel, test_loader, device, len(test_data))
    print('test_score: {:.4f}'.format(val_acc_image))


if __name__ == '__main__':
    main()
