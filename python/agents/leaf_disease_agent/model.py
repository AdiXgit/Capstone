"""Backbone construction for the three transfer-learning regimes.

The three modes exist so the ablation can attribute accuracy to pretraining
rather than to architecture:

    scratch   randomly initialised, everything trainable
    frozen    ImageNet weights, backbone frozen, linear probe on the head
    finetune  ImageNet weights, head warm-up then last block unfrozen
"""
import torch.nn as nn
from torchvision import models

MODES = ("scratch", "frozen", "finetune")

_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
}


def _replace_head(net, arch, num_classes, dropout):
    if arch == "resnet18":
        in_features = net.fc.in_features
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return net.fc
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return net.classifier[-1]


def _last_block(net, arch):
    return net.layer4 if arch == "resnet18" else net.features[-3:]


def build_model(mode, arch="resnet18", num_classes=3, dropout=0.3):
    """Return (model, head_params, block_params).

    block_params is empty unless mode == "finetune"; callers use the split to
    give the pretrained block a smaller learning rate than the fresh head.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if arch not in _BACKBONES:
        raise ValueError(f"arch must be one of {sorted(_BACKBONES)}, got {arch!r}")

    ctor, weights = _BACKBONES[arch]
    net = ctor(weights=None if mode == "scratch" else weights)
    head = _replace_head(net, arch, num_classes, dropout)

    if mode == "scratch":
        return net, list(net.parameters()), []

    for p in net.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True

    if mode == "frozen":
        return net, list(head.parameters()), []

    block = _last_block(net, arch)
    for p in block.parameters():
        p.requires_grad = True
    return net, list(head.parameters()), list(block.parameters())


def set_block_trainable(block_params, trainable):
    for p in block_params:
        p.requires_grad = trainable


def freeze_batchnorm_where_params_frozen(net):
    """Put BatchNorm layers into eval() mode wherever their params are frozen.

    requires_grad=False stops gradient flow but NOT the running mean/variance
    updates, which happen in the forward pass whenever the module is in train()
    mode. Without this, a "frozen" backbone still drifts its BatchNorm
    statistics on the new data and is not actually frozen -- which would make
    the frozen-vs-finetune comparison measure the wrong thing.

    Must be called after every net.train() and after any change to
    requires_grad, since train() resets every module's mode.
    """
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if not any(p.requires_grad for p in m.parameters(recurse=False)):
                m.eval()
