import os

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from matplotlib import pyplot as plt


class MSE(nn.Module): #使用注意力以及交叉，在教师和量化后的学生之间进行自蒸馏
    def __init__(self, args):
        super(MSE, self).__init__()
        self.guide_layers = args.guide_layers
        self.hint_layers = args.hint_layers
        self.qk_dim = args.qk_dim

    def forward(self, g_s, g_t, cross_features, Cross=False):
        # g_s是学生的中间特征图，g_t是教师的中间特征图
        # 如何处理这两
        # 蒸馏损失目前选用的是MSE
        g_t = [g_t[i] for i in self.guide_layers]
        g_s = [g_s[i] for i in self.hint_layers]

        loss = []
        for i, h_t in enumerate(g_t):
            h_s = g_s[i]
            diff = self.cal_diff(h_s, h_t)
            loss.append(diff)

        if Cross:
            cross_loss, cross_loss_index = [], []
            for i in range(len(loss)):
                cross = cross_features[i]
                h_t = g_t[i]
                h_s = g_s[i]
                tea_channel_att = h_t.mean(3).mean(2)  # (batch_size, channels)
                stu_channel_att = h_s.mean(3).mean(2)  # (batch_size, channels)
                # # 计算批次内的点积（得到一个batch_size大小的向量）
                dot_product = torch.sum(tea_channel_att * stu_channel_att, dim=1)  # (batch_size,)
                # # 计算缩放点积，将其平均得到一个标量权重
                scaled_dot_product = dot_product / torch.sqrt(torch.tensor(tea_channel_att.size(1), dtype=torch.float32, device=tea_channel_att.device))  # (batch_size,)
                average_weight = torch.mean(scaled_dot_product)  # 标量
                # # 对这个标量应用Sigmoid以确保权重在(0, 1)之间
                # final_weight = torch.sigmoid(average_weight)
                final_weight = 1 - torch.sigmoid(average_weight)

                differ = self.cal_diff(F.normalize(h_s), F.normalize(cross))
                differ = differ * final_weight
                cross_loss.append(differ)
            return sum(loss), sum(cross_loss)
        else:
            return sum(loss), 0

    def cal_diff(self, v_s, v_t):
        diff = (v_s - v_t).pow(2).mean(1).mean()
        return diff
