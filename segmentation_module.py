import torch
import torch.nn as nn
from torch import distributed
import torch.nn.functional as functional
import inplace_abn
from inplace_abn import InPlaceABNSync, InPlaceABN, ABN

from functools import partial, reduce

import models
from modules import DeeplabV3
from torch.nn import functional as F
from utils.non_local_embedded_gaussian import NONLocalBlock2D
from cc import CC_module
def make_model(opts, classes=None):
    if opts.norm_act == 'iabn_sync':
        norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'iabn':
        norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'abn':
        norm = partial(ABN, activation="leaky_relu", activation_param=.01)
    else:
        norm = nn.BatchNorm2d  # not synchronized, can be enabled with apex
    head_channels = 256

    if not opts.no_pretrained:
        body = models.__dict__[f'net_{opts.backbone}'](norm_act=norm, output_stride=opts.output_stride)
        pretrained_path = f'pretrained/{opts.backbone}_{opts.norm_act}.pth.tar'
        pre_dict = torch.load(pretrained_path, map_location='cpu')
        # print(pre_dict['state_dict'])
        del pre_dict['state_dict']['module.classifier.fc.weight']
        del pre_dict['state_dict']['module.classifier.fc.bias']

        # print(pre_dict['state_dict'].keys())
        kep = list(pre_dict['state_dict'].keys())

        for key in kep:
            # print(key)
            key_n=key[7:]
            # print(key_n)
            pre_dict['state_dict'][key_n]=pre_dict['state_dict'].pop(key)
        body.load_state_dict(pre_dict['state_dict'])
        del pre_dict  # free memory


        head = DeeplabV3(body.out_channels, head_channels, 256, norm_act=norm,
                         out_stride=opts.output_stride, pooling_size=opts.pooling)

    if classes is not None:
        model = IncrementalSegmentationModule(body, head, head_channels, classes=classes, fusion_mode=opts.fusion_mode)
    else:
        model = SegmentationModule(body, head, head_channels, opts.num_classes, opts.fusion_mode)

    return model


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalSegmentationModule(nn.Module):

    def __init__(self, body, head, head_channels, classes, ncm=False, fusion_mode="mean"):
        super(IncrementalSegmentationModule, self).__init__()
        self.body = body
        self.head = head
        # classes must be a list where [n_class_task[i] for i in tasks]
        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"
        self.cls = nn.ModuleList(
            [nn.Conv2d(head_channels, c, 1) for c in classes]
        )
        self.classes = classes
        self.head_channels = head_channels
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.means = None
        self.softmax = nn.Softmax(dim=1)
        ### SE module
        self.selayer_2048 = SELayer(channel=2048).cuda()
        self.selayer_256 = SELayer(channel=256).cuda()
        # self.selayer_4096 = SELayer(channel=4096).cuda()
        ### Spatial attention layer
        self.splayer_2048 = SPLayer(channel=2048).cuda()
        self.splayer_256 = SPLayer(channel=256).cuda()
        ### multi-head module self attention
        self.multihead_att_2048= multi_head_attenionLayerN(2048).cuda()
        self.multihead_att_256= multi_head_attenionLayer(256).cuda()

    def att_map(self,x):
        ### sptial attention
        a = torch.sum(x ** 2, dim=1)
        ### channel attention
        for i in range(a.shape[0]):
            a[i] = a[i] / torch.norm(a[i])
        a = torch.unsqueeze(a, 1)
        x = a.detach() * x
        return x
    def satt_map(self,x):
        # ### channel attention
        bs, W, h, w = x.size()
        if W == 2048:
            x = x+self.selayer_2048(x)
        else:
            x = x+self.selayer_256(x)


        ### sptial attention
        # if W == 2048:
        #     x = self.splayer_2048(x) + x
        # else:
        #     x = self.splayer_256(x) + x
        # a = torch.sum(x ** 2, dim=1)
        # for i in range(a.shape[0]):
        #     a[i] = a[i] / torch.norm(a[i])
        # a = torch.unsqueeze(a, 1)
        # x = a.detach() * x

        return x

    def _network(self, x,x_b_old=None,x_pl_old=None, ret_intermediate=False):
        # x_b.shape=[bs,2048,32,32] x_pl.shape=[bs,256,32,32] x_o.shape=[bs,ch_out+1,32,32]
        ### for origin and reproduce
        x_b = self.body(x)
        ### multi-head
        # if x_b_old is not None:
        #     p=torch.randn(1)
        #     ### modal dropout
        #     if p>0.3:
        #         x_b = self.multihead_att_2048(x_b_old, x_b)
        #     else:
        #         x_b = self.multihead_att_2048(None, x_b)
        # else:
        # x_b = self.multihead_att_2048(x_b_old, x_b)
        x_pl = self.head(x_b)
        # x_pl = self.multihead_att_256(x_pl_old, x_pl)

        out = []
        for mod in self.cls:
            out.append(mod(x_pl))
        x_o = torch.cat(out, dim=1)

        # print(x_o.shape)
        if ret_intermediate:

            ### Attentive Feature Distillation(AFD)
            # x_b=self.satt_map(x_b)
            # x_pl=self.satt_map(x_pl)

            return x_o, x_b,  x_pl
        return x_o

    def init_new_classifier(self, device):
        cls = self.cls[-1]
        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x, x_b_old=None,x_pl_old=None, scales=None, do_flip=False, ret_intermediate=False):
        out_size = x.shape[-2:]

        out = self._network(x, x_b_old, x_pl_old,ret_intermediate)

        sem_logits = out[0] if ret_intermediate else out

        sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)

        # if ret_intermediate:
        return sem_logits, {"body": out[1], "pre_logits": out[2]}

        # return sem_logits, {}

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SPLayer(nn.Module):
    def __init__(self, channel):
        super(SPLayer, self).__init__()
        self.bn = nn.BatchNorm2d(1)
        self.splayer = nn.Conv2d(channel, 1, 1)
    def forward(self, x):
        y = self.splayer(x**2)
        y = self.bn(y)
        return x * y.expand_as(x)

class multi_head_attenionLayer(nn.Module):
    def __init__(self, channel):
        super(multi_head_attenionLayer, self).__init__()
        self.self_att_old=NONLocalBlock2D(in_channels=channel)
        self.self_att_new=NONLocalBlock2D(in_channels=channel)

        self.splayer = nn.Sequential(
                                    nn.Conv2d(2 * channel, 1, 1),
                                    nn.BatchNorm2d(1)
                                    )
        self.chlayer=nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(2*channel, channel , bias=False),
            nn.Linear(channel, channel // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 16, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x_old, x_new):
        if x_old is None:
            x_old=x_new
        b, c, _, _ = x_old.size()
        z_old=self.self_att_old(x_old)
        z_new=self.self_att_new(x_new)
        z_ = torch.cat((z_old, z_new), dim=1)
        att_ch=self.fc(self.chlayer(z_).view(b,2*c)).view(b, c, 1, 1)
        att_sp=self.splayer(z_**2)
        out=att_ch*(att_sp*x_new)+x_new
        return out

class multi_head_attenionLayerN(nn.Module):
    def __init__(self, channel):
        super(multi_head_attenionLayerN, self).__init__()
        self.self_att_old=NONLocalBlock2D(in_channels=channel)
        self.self_att_new=NONLocalBlock2D(in_channels=channel)

        self.splayer = nn.Sequential(
                                    nn.Conv2d(channel, 1, 1),
                                    nn.BatchNorm2d(1)
                                    )
        self.chlayer=nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 16, channel, bias=False),
            nn.Sigmoid()
        )
        self.confuslayer=nn.Conv2d(2*channel, channel, 1)

    def forward(self,x_old, x_new):
        b, c, _, _ = x_new.size()

        if x_old is None:
            z_new = self.self_att_new(x_new)
            z_=z_new
        else:
            b, c, h, w = x_new.size()
            # x_old=torch.zeros(b, c, h, w).cuda()
            # x_old=torch.randn(b, c, h, w).cuda()
            # z_old=self.self_att_old(x_old)
            z_new=self.self_att_new(x_new)
            z_ = torch.cat((z_new, z_new), dim=1)
            z_ =self.confuslayer(z_)
        att_ch=self.fc(self.chlayer(z_).view(b,c)).view(b, c, 1, 1)
        att_sp=self.splayer(z_**2)
        att_sp= att_sp/torch.max(att_sp)
        out=att_ch*(att_sp*x_new)+x_new
        return out
