import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.ops.modules import MSDeformAttn


class MultiBranch(nn.Module):
    def __init__(self, branches, embed_dim_list):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.embed_dim_list = embed_dim_list#[128,128]


    # def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, need_weights=True, static_kv=False, attn_mask=None):
    # def forward(self, query,  key_padding_mask=None, incremental_state=None):
    # src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
    #                       padding_mask)
    def forward(self,src_pos, reference_points, src,spatial_shapes, level_start_index, padding_mask,incremental_state=None):
        bsz,tgt_len,embed_size=src.size()#[2,10584,256]
        # print(reference_points.shape)#torch.Size([2, 10584, 3, 2])
        # print(src.shape)#torch.Size([2, 10584, 256])
        # print(spatial_shapes)#tensor([[84, 84],[42, 42],[42, 42]], device='cuda:0')
        # print(level_start_index)#tensor([   0, 7056, 8820], device='cuda:0')
        assert sum(self.embed_dim_list) == embed_size
        out = []
        attn = None
        start = 0
        for idx, embed_dim in enumerate(self.embed_dim_list):
            branch = self.branches[idx]
            branch_type = type(branch)
            # print(branch_type)# <class 'models.ops.modules.ms_deform_attn.MSDeformAttn'>
            src_new= src[...,start:start+embed_dim]#选取前128个通道q: torch.Size([2, 8928, 128])
            src_pos_new=src_pos[...,start:start+embed_dim]#也需要选 torch.Size([2, 8928, 128])
            start += embed_dim
            #输入src的前100个通道运用MSDeformAttn
            if branch_type == MSDeformAttn:
                 # print("1")
                 x=branch(src_pos_new, reference_points,  src_new, spatial_shapes, level_start_index, padding_mask)#[2, 8928, 128])
                 # print("x0:",x.shape)
            else:
                mask = padding_mask
                # print(mask)#[2,8192]
                if mask is not None:
                    src_new = src_new.masked_fill(mask.unsqueeze(2), 1)#masked_fill在时序任务中，mask到当前时刻后面时刻的序列信息
                x = branch(src_new.contiguous(), incremental_state=incremental_state)#[2,12240,128]
                # print("x1:",x.shape)

            out.append(x)
        out = torch.cat(out, dim=-1)#此处实现通道的合并
        return out