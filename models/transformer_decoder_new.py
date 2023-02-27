# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision

from models.misc import _get_clones, _get_activation_fn, MLP
from models.position_encoding import gen_sineembed_for_position
from models.attention import MultiheadAttention
from util.box_ops import box_cxcywh_to_xyxy


class TransformerDecoder(nn.Module):
    def __init__(self, args, decoder_layer, num_layers):
        super().__init__()
        self.args = args
        self.multiscale = args.multiscale
        self.num_layers = num_layers
        self.layers = _get_clones(decoder_layer, num_layers)
        assert num_layers == self.args.dec_layers
        self.box_embed = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                memory_h=None,
                memory_w=None,
                grid=None):
        output = tgt

        intermediate = []
        intermediate_reference_boxes = []

        for layer_id, layer in enumerate(self.layers):

            #modified by sjj 20221129
            if layer_id == 0 :
                scale_level =3#减少初始定位的搜索空间
            elif layer_id == 1 or layer_id == 2:
                scale_level = 2#更精确的定位
            elif layer_id==3 or layer_id == 4:
                scale_level = 1
            elif  layer_id == 5:
                scale_level = 0# 小物体
            else:
                assert False

            #参考框learnable refernece box，参考框进行相关语义的引导
            if layer_id == 0:
                # 在decoder的第一层   代表object query的初始化位置
                reference_boxes_before_sigmoid = query_pos  # [num_queries, batch_size, 4]
                reference_boxes = reference_boxes_before_sigmoid.sigmoid().transpose(0, 1)#第一层

            else:
               # print(self.bbox_embed[layer_id - 1])
                tmp = self.bbox_embed[layer_id - 1](output)
                reference_boxes_before_sigmoid = tmp + reference_boxes_before_sigmoid#add上一层结果
                reference_boxes = reference_boxes_before_sigmoid.sigmoid().transpose(0, 1)
                reference_boxes_before_sigmoid = reference_boxes_before_sigmoid.detach()
                reference_boxes = reference_boxes.detach()

           # print("reference_boxes :",reference_boxes.shape)#[1,300,4]

            obj_center = reference_boxes[..., :2].transpose(0, 1)      # [num_queries, batch_size, 2]
           # print("onj_center:",obj_center.shape)#[300,1,2]


            # get sine embedding for the query vector sine embedding
            query_ref_boxes_sine_embed = gen_sineembed_for_position(obj_center)
            #print("query_ref_boxes_sine_embed :",query_ref_boxes_sine_embed.shape)#[300，1，256]

            if self.multiscale:
                memory_ = memory[scale_level]
                # print("memory_.shape",memory_ .shape)
                memory_h_ = memory_h[scale_level]
                memory_w_ = memory_w[scale_level]
                memory_key_padding_mask_ = memory_key_padding_mask[scale_level]
                pos_ = pos[scale_level]
                grid_ = grid[scale_level]
            else:
                memory_ = memory
                memory_h_ = memory_h
                memory_w_ = memory_w
                memory_key_padding_mask_ = memory_key_padding_mask
                pos_ = pos
                grid_ = grid

            output = layer(output,
                           memory_,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask_,
                           pos=pos_,
                           query_ref_boxes_sine_embed=query_ref_boxes_sine_embed,
                           reference_boxes=reference_boxes,
                           memory_h=memory_h_,
                           memory_w=memory_w_,
                           grid=grid_,)

            intermediate.append(output)
            intermediate_reference_boxes.append(reference_boxes)

        return torch.stack(intermediate).transpose(1, 2), \
               torch.stack(intermediate_reference_boxes)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args, activation="relu"):
        super().__init__()
        self.args = args
        self.d_model = args.hidden_dim
        self.nheads = args.nheads
        self.num_queries = args.num_queries
        self.dim_feedforward = args.dim_feedforward
        self.dropout = args.dropout
        self.activation = _get_activation_fn(activation)

        # Decoder Self-Attention sa
        self.sa_qcontent_proj = nn.Linear(self.d_model, self.d_model)
        self.sa_qpos_proj = nn.Linear(self.d_model, self.d_model)
        self.sa_kcontent_proj = nn.Linear(self.d_model, self.d_model)
        self.sa_kpos_proj = nn.Linear(self.d_model, self.d_model)
        self.sa_v_proj = nn.Linear(self.d_model, self.d_model)
        self.self_attn = MultiheadAttention(self.d_model, self.nheads, dropout=self.dropout, vdim=self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)
        self.norm1 = nn.LayerNorm(self.d_model)

        # Decoder Cross-Attention ca
        self.ca_qcontent_proj = nn.Linear(self.d_model, self.d_model)
        self.ca_kcontent_proj = nn.Linear(self.d_model, self.d_model)
        self.ca_v_proj = nn.Linear(self.d_model, self.d_model)
        self.ca_qpos_sine_proj = MLP(self.d_model, self.d_model, self.d_model, 2)
        self.ca_kpos_sine_proj = MLP(self.d_model, self.d_model, self.d_model, 2)
        self.cross_attn = MultiheadAttention(self.nheads * self.d_model, self.nheads, dropout=self.dropout, vdim=self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(self.d_model)

        self.point1 = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        if self.args.smca:
            self.point2 = nn.Sequential(
                nn.Linear(self.d_model // 4 * 7 * 7, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.nheads * 4),
            )
            nn.init.constant_(self.point2[-1].weight.data, 0)
            nn.init.constant_(self.point2[-1].bias.data, 0)
        else:
            self.point2 = nn.Sequential(
                nn.Linear(self.d_model // 4 * 7 * 7, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.nheads * 2),
            )
            nn.init.constant_(self.point2[-1].weight.data, 0)
            nn.init.constant_(self.point2[-1].bias.data, 0)


        self.attn1 = nn.Linear(self.d_model, self.d_model * self.nheads)
        self.attn2 = nn.Linear(self.d_model, self.d_model * self.nheads)

        # FFN
        self.linear1 = nn.Linear(self.d_model, self.dim_feedforward)
        self.dropout88 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, self.d_model)
        self.dropout3 = nn.Dropout(self.dropout)
        self.norm3 = nn.LayerNorm(self.d_model)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h, valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_ref_boxes_sine_embed = None,
                reference_boxes: Optional[Tensor] = None,
                memory_h=None,
                memory_w=None,
                grid=None):

        num_queries = tgt.shape[0]
        bs = tgt.shape[1]
        c = tgt.shape[2]
        n_model = c
        valid_ratio = self.get_valid_ratio(memory_key_padding_mask.view(bs, memory_h, memory_w))

        #print("memory:",memory.shape)#memory: torch.Size([2500, 1, 256])  torch.Size([10000, 1, 256]) --([1634, 2, 256])
        memory_2d = memory.view(memory_h, memory_w, bs, c)
        #print("memory_2d:", memory_2d.shape)#torch.Size([50, 50, 1, 256]) torch.Size([100, 100, 1, 256])--([38, 43, 2, 256])
        memory_2d = memory_2d.permute(2, 3, 0, 1)
        #print("memory_2d:", memory_2d.shape)#memory_2d: torch.Size([1, 256, 50, 50]) torch.Size([1, 256, 100, 100])--[2, 256, 38, 43])

        # ========== Begin of Self-Attention =============
        q_content = self.sa_qcontent_proj(tgt)#content
        q_pos = self.sa_qpos_proj(query_ref_boxes_sine_embed)#position
        #query==key
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_ref_boxes_sine_embed)
        v = self.sa_v_proj(tgt)#对tgt进行线性映射得到calue
        q = q_content + q_pos#包含内容和位置
        k = k_content + k_pos#包含内容和位置
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        #print("tgt2.shape",tgt2.shape) #([300, 1, 256]  --[300, 2, 256])
        # ========== End of Self-Attention =============
        tgt = tgt + self.dropout1(tgt2)#此处是否去掉dropout
        tgt = self.norm1(tgt)

         #参考框
        reference_boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
       # print("refernence_boxes:",reference_boxes.shape)--[2, 300, 4]
       #  print("reference_boxes_xyxy:",reference_boxes_xyxy.shape)--[2, 300, 4]
        reference_boxes_xyxy[:, :, 0] *= memory_w
        reference_boxes_xyxy[:, :, 1] *= memory_h
        reference_boxes_xyxy[:, :, 2] *= memory_w
        reference_boxes_xyxy[:, :, 3] *= memory_h
        # print("reference_boxes_xyxy:", reference_boxes_xyxy.shape)--[2, 300, 4])
        reference_boxes_xyxy = reference_boxes_xyxy * valid_ratio.view(bs, 1, 4)#torch.Size([1, 300, 4])

        #ysemantic-aligned matching从已编码的图像特征中重采样新的object query embedding
        #ROIAlign从相应的参考矿中从编码图像特征中提取区域特征 N*7*7*d
        #ROALign
        q_content = torchvision.ops.roi_align(
            memory_2d,
            list(torch.unbind(reference_boxes_xyxy, dim=0)),
            output_size=(7, 7),
            spatial_scale=1.0,
            aligned=True)  # (bs * num_queries, c, 7, 7)

        q_content_points = torchvision.ops.roi_align(
            memory_2d,
            list(torch.unbind(reference_boxes_xyxy, dim=0)),
            output_size=(7, 7),
            spatial_scale=1.0,
            aligned=True)  # (bs * num_queries, c, 7, 7)
        #print("  q_content_points:",  q_content_points.shape)#torch.Size([300, 256, 7, 7])--【600，256，7，7】

        q_content_index = q_content_points.view(bs * num_queries, -1, 7, 7)
        #print(" q_content_index:", q_content_index.shape)#torch.Size([300, 256, 7, 7])--【600，256，7，7】

        #self.point1  self.point2  是利用conv+MLP进行M个关键点的空间位置的预测，表示对识别和定位潜在目标的至关重要的信息。预测坐标被限制在坐标框内
        #M个关键点  M个预测头
        points = self.point1(q_content_index)#conv2d  relu
        #print("points:",points.shape)#points: torch.Size([300, 64, 7, 7])
        points = points.reshape(bs * num_queries, -1)
        #print("points:", points.shape)#[300,3136]--[600, 3136]
        points = self.point2(points)#linear relu
        #print("points:",points.shape)#points: torch.Size([300, 32])--[600,16] no smca
        if not self.args.smca:#空间调制的co-attention  spatially modelated co-attention
            points = points.view(bs * num_queries, 1, self.nheads, 2).tanh()
        else:
            points_scale = points[:, 2 * self.nheads:].reshape(bs, num_queries, self.nheads, 2).permute(1, 0, 2, 3)
            points = points[:, :2 * self.nheads].view(bs * num_queries, 1, self.nheads, 2).tanh()
        #print("points:", points.shape)#: torch.Size([300, 1, 8, 2])

        #ROI_align
        q_content = F.grid_sample(q_content, points, padding_mode="zeros", align_corners=False).view(bs * num_queries, -1)
       # print("q_content:",q_content.shape)#q_content: torch.Size([300, 1, 8, 256]) -- [600,2048]
        q_content = q_content.view(bs, num_queries, -1, 8).permute(1, 0, 3, 2)   # (num_query, bs, n_head, 256)
        q_content = q_content * self.attn1(tgt).view(num_queries, bs, self.nheads, n_model).sigmoid()#attn1 linearsigmoid
        #print("q_content:", q_content.shape)# torch.Size([300, 2048])--[300,2,8,256]

       #采样点，中心，scale,delta
        q_pos_center = reference_boxes[:, :, :2].reshape(bs, num_queries, 1, 2).expand(-1, -1, self.nheads, -1)
        q_pos_scale = reference_boxes[:, :, 2:].reshape(bs, num_queries, 1, 2).expand(-1, -1, self.nheads, -1) * 0.5
        q_pos_delta = points.reshape(bs, num_queries, self.nheads, 2)
        q_pos = q_pos_center + q_pos_scale * q_pos_delta

        q_pos = q_pos.permute(1, 0, 2, 3)   # (num_query, bs, n_head, 2)
        q_pos = q_pos.reshape(num_queries, bs * self.nheads, 2)

        if self.args.smca:
            # SMCA: start
            #估计每个object query 的位置，用二维高斯权重来约束不同注意头的注意响应
            gau_point = torch.clone(q_pos)
            gau_point[:, :, 0] *= memory_w
            gau_point[:, :, 1] *= memory_h
            gau_point = gau_point.reshape(num_queries, bs, self.nheads, 2)
            gau_point = gau_point * valid_ratio[:, :2].reshape(1, bs, 1, 2)
            gau_point = gau_point.reshape(num_queries, bs * self.nheads, 2)
            gau_distance = (gau_point.unsqueeze(1) - (grid + 0.5).unsqueeze(0)).pow(2)
            gau_scale = points_scale
            gau_scale = gau_scale * gau_scale
            gau_scale = gau_scale.reshape(num_queries, -1, 2).unsqueeze(1)
            gau_distance = (gau_distance * gau_scale).sum(-1)
            gaussian = -(gau_distance - 0).abs() / 8.0         # 8.0 is the number used in SMCA-DETR
            # SMCA: end
        else:
            gaussian = None

        q_pos = gen_sineembed_for_position(q_pos).reshape(num_queries, bs, self.nheads, c)
        q_pos = q_pos * self.attn2(tgt).view(num_queries, bs, self.nheads, n_model).sigmoid()##attn2 linearsigmoid 合并以前的有用信息
        #至此，新的query与pos实现了与encoded image feature语义对其

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(q_content)#对齐之后的
        k_content = self.ca_kcontent_proj(memory).view(-1, bs, 1, 256).expand(-1, -1, self.nheads, -1)
        v = self.ca_v_proj(memory).view(-1, bs, n_model)

        num_queries, bs, n_head, n_model = q_content.shape
        hw, _, _, _ = k_content.shape

        q = q_content
        k = k_content

        query_sine_embed = self.ca_qpos_sine_proj(q_pos)
        q = (q + query_sine_embed).view(num_queries, bs, self.nheads * n_model)

        k = k.view(hw, bs, self.nheads, n_model)
        k_pos = self.ca_kpos_sine_proj(pos)
        k_pos = k_pos.view(hw, bs, 1, n_model).expand(-1, -1, self.nheads, -1)
        k = (k + k_pos).view(hw, bs, self.nheads * n_model)

        if self.args.smca:
            tgt2 = self.cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   gaussian=[gaussian])[0]
        else:
            tgt2 = self.cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   gaussian=None)[0]
        # ========== End of Cross-Attention =============
        # print("tg2:",tgt2.shape)#tg2: torch.Size([300, 1, 256])
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout88(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt
