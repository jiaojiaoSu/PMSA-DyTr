# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.misc import _get_clones, _get_activation_fn
from models.ops.modules import MSDeformAttn
from models.modules.multibranch import MultiBranch
from models.modules.lightweight_convolution import LightweightConv
from models.modules.dynamic_convolution import DynamicConv


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, args, encoder_layer, num_layers):
        super().__init__()
        self.args = args
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        assert num_layers == self.args.enc_layers

    @staticmethod
    #获取参考点的坐标信息
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, args, activation='relu'):
        super().__init__()

        self.args = args
        self.d_model = args.hidden_dim
        self.nheads = args.nheads
        self.num_queries = args.num_queries
        # Note: Multiscale encoder's dim_feedforward halved for memory efficiency
        self.dim_feedforward = args.dim_feedforward // 2
        self.dropout = args.dropout

        # Hard-coded Hyper-parameters
        self.n_feature_levels = 4#3
        self.n_points = 4

        # self attention

        # self.self_attn = MSDeformAttn(self.d_model, self.n_feature_levels, self.nheads, self.n_points)
        # self attention
        # 此处根据args.encoder_brach_type的类型进行self-attention的选取  根据通道维度一分为二


        # 此处做2/3 1/3实验
        d_model=self.d_model
        n_heads=self.nheads
        # args.encoder_branch_type = ["attn" + ":" + "1" + ":" + str(d_model // 2) + ":" + str(n_heads // 2),
        #                             "lightweight" + ":" + "default" + ":" + str(d_model // 2) + ":" + str(n_heads// 2)]
        # args.encoder_branch_type = ["attn" + ":" + "1" + ":" + str(d_model // 2) + ":" + str(n_heads // 2),
        #                             "dynamic'" + ":" + "default" + ":" + str(d_model // 2) + ":" + str(n_heads // 2)]
        args.encoder_branch_type = ["attn" + ":" + "1" + ":" + str(d_model // 4) + ":" + str(n_heads // 4),
                                    "dynamic'" + ":" + "default" + ":" + str(3 * d_model // 4) + ":" + str(
                                        3 * n_heads // 4)]  # 后面也需修改
        # args.encoder_branch_type = ["attn" + ":" + "1" + ":" + str(3 *d_model // 4) + ":" + str(3 *n_heads // 4),
        #                             "dynamic'" + ":" + "default" + ":" + str( d_model // 4) + ":" + str( n_heads // 4)]  # 后面也需修改
        print(args.encoder_branch_type)
        self.num_heads = n_heads
        self.embed_dim = d_model
        if args.encoder_branch_type is None:
            self.self_attn = MSDeformAttn(self.d_model, self.n_feature_levels, self.nheads, self.n_points)
        else:
            layers = []
            embed_dims = []
            heads = []
            num_types = len(args.encoder_branch_type)  # 分支按通道分成2个部分，一个是捕获长程依赖的全局注意力，另一个是捕获局部的线性卷积
            for layer_type in args.encoder_branch_type:
                embed_dims.append(int(layer_type.split(':')[2]))  # d_model//2 [:]->[,]
                heads.append(int(layer_type.split(':')[3]))  # nhead//2
                layers.append(self.get_layer(args, 6, embed_dims[-1], heads[-1], layer_type))  # index=6,表示encoder的数目为6
            assert sum(embed_dims) == self.embed_dim, (sum(embed_dims), self.embed_dim)
            print("layers:", layers)
            self.self_attn = MultiBranch(layers, embed_dims)
        self.dropout1 = nn.Dropout(self.dropout)
        self.norm1 = nn.LayerNorm(self.d_model)

        # ffn
        self.linear1 = nn.Linear(self.d_model, self.dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, self.d_model)
        self.dropout3 = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(self.d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

        # add  sjj 20220830
    def get_layer(self, args, index, out_dim, num_heads, layer_type):
        # print("index",index)
        kernel_size = layer_type.split(':')[1]
        if kernel_size == 'default':
            kernel_size = args.encoder_kernel_size_list[index]
        else:
            kernel_size = int(kernel_size)
        padding_l = kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2)
        if 'lightweight' in layer_type:
            layer = LightweightConv(
                out_dim, kernel_size, padding_l=padding_l, weight_softmax=args.weight_softmax,
                num_heads=num_heads, weight_dropout=args.weight_dropout,
                with_linear=args.conv_linear,
            )
        elif 'dynamic' in layer_type:
            layer = DynamicConv(
                out_dim, kernel_size, padding_l=padding_l,
                weight_softmax=args.weight_softmax, num_heads=num_heads,
                weight_dropout=args.weight_dropout, with_linear=args.conv_linear,
                glu=args.encoder_glu,
            )
        elif 'attn' in layer_type:
           # layer =MSDeformAttn(128, self.n_feature_levels, self.nheads, self.n_points)#按通道分为二#此处为3个level
           # layer = MSDeformAttn(192, self.n_feature_levels, self.nheads, self.n_points)  # 按通道分为二#此处为3个level
           layer = MSDeformAttn(64, self.n_feature_levels, self.nheads, self.n_points)  # 按通道分为二#此处为3个level

        else:
            raise NotImplementedError

        return layer


    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos),
                              reference_points,
                              src,
                              spatial_shapes,
                              level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src
