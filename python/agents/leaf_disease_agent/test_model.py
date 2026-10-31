import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import (
    MODES, build_model, freeze_batchnorm_where_params_frozen, set_block_trainable,
)


def _bn_modes(net):
    bns = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]
    return [m.training for m in bns]


def test_all_modes_build():
    for mode in MODES:
        net, head, block = build_model(mode)
        assert sum(p.numel() for p in net.parameters()) > 0
        assert head, f"{mode} has no trainable head params"


def test_scratch_trains_everything():
    net, head, block = build_model("scratch")
    assert block == []
    assert all(p.requires_grad for p in net.parameters())


def test_frozen_trains_only_the_head():
    net, head, block = build_model("frozen")
    assert block == []
    trainable = [p for p in net.parameters() if p.requires_grad]
    assert len(trainable) == len(head)


def test_finetune_exposes_block_separately():
    net, head, block = build_model("finetune")
    assert block, "finetune must expose the last block for a lower LR"
    assert all(p.requires_grad for p in block)


def test_frozen_backbone_batchnorm_is_actually_frozen():
    """requires_grad=False does not stop running-stat updates; eval() does.

    Without this, a "frozen" backbone keeps adapting its BatchNorm statistics
    to the new data during training and the frozen-vs-finetune comparison
    measures something other than what it claims.
    """
    net, _, _ = build_model("frozen")
    net.train()
    assert all(_bn_modes(net)), "sanity: train() should put every BN in train mode"

    freeze_batchnorm_where_params_frozen(net)
    assert not any(_bn_modes(net)), "every BN in a frozen backbone must be eval()"


def test_scratch_batchnorm_stays_trainable():
    net, _, _ = build_model("scratch")
    net.train()
    freeze_batchnorm_where_params_frozen(net)
    # Nothing is frozen, so no BN should have been switched off.
    assert all(_bn_modes(net))


def test_finetune_unfreezes_block_batchnorm_after_warmup():
    net, _, block = build_model("finetune")

    set_block_trainable(block, False)  # warm-up phase
    net.train()
    freeze_batchnorm_where_params_frozen(net)
    assert not any(_bn_modes(net)), "during warm-up the whole backbone is frozen"

    set_block_trainable(block, True)  # block now adapting
    net.train()
    freeze_batchnorm_where_params_frozen(net)
    modes = _bn_modes(net)
    assert any(modes), "the unfrozen block's BN must update once it trains"
    assert not all(modes), "BN outside the unfrozen block must stay frozen"


def test_rejects_bad_arguments():
    for bad in [{"mode": "nope"}, {"mode": "frozen", "arch": "nope"}]:
        try:
            build_model(**bad)
        except ValueError:
            continue
        raise AssertionError(f"should have rejected {bad}")
