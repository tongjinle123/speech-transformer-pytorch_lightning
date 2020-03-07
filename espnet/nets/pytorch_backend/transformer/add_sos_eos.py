#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Unility funcitons for Transformer."""

import torch as t


def add_sos_eos(ys_pad, ys_length, sos, eos, ignore_id):

    input_ = t.nn.functional.pad(ys_pad, (1, 0), value=sos)
    target_ = t.nn.functional.pad(ys_pad, (0, 1), value=ignore_id)
    indices = t.LongTensor([[i, v.item()] for i, v in enumerate(ys_length)]).to(ys_pad.device)
    values = t.LongTensor([eos for i in ys_length]).to(ys_pad.device)
    target_ = target_.index_put(tuple(indices.t()), values=values)
    return input_.detach(), target_.detach()


    # """Add <sos> and <eos> labels.
    #
    # :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    # :param int sos: index of <sos>
    # :param int eos: index of <eeos>
    # :param int ignore_id: index of padding
    # :return: padded tensor (B, Lmax)
    # :rtype: torch.Tensor
    # :return: padded tensor (B, Lmax)
    # :rtype: torch.Tensor
    # """
    # from espnet.nets.pytorch_backend.nets_utils import pad_list
    # print(ys_pad.shape)
    #
    # _sos = ys_pad.new([sos])
    # _eos = ys_pad.new([eos])
    # print(_sos.shape)
    # ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    # ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    # ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)
