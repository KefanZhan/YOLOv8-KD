import os

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from matplotlib import pyplot as plt

def visualize_feature(features, save_dir):
    os.makedirs(save_dir, exist_ok=True)  # 如果文件夹不存在，则创建
    # 遍历 features 列表，依次处理每个特征图
    for idx, feature in enumerate(features):
        if idx == len(features) - 1:
            break
        # 假设 feature 的形状为 (batch_size, channels, height, width)
        # 选择第一个样本的特征图进行可视化，feature[0] 表示第一个 batch
        single_feature = feature[0]
        # 对通道进行均值处理，生成一个二维特征图
        combined_feature = single_feature.mean(dim=0).detach().cpu().numpy()  # 对通道取均值
        # 使用伪彩色映射（如 'jet'）将二维特征图保存为 RGB 图像
        save_path = os.path.join(save_dir, f'feature_map_{idx + 1}.png')
        plt.imsave(save_path, combined_feature, cmap='jet')  # 使用 'jet' 显示 RGB 图像


class MyCriterion(nn.Module): #使用注意力以及交叉，在教师和量化后的学生之间进行自蒸馏
    def __init__(self, args):
        super(MyCriterion, self).__init__()
        self.guide_layers = args.guide_layers
        self.hint_layers = args.hint_layers
        self.qk_dim = args.qk_dim

    def forward(self, g_s, g_t, cross_features, Cross=False):
        # g_s是学生的中间特征图，g_t是教师的中间特征图
        # 如何处理这两
        # 蒸馏损失目前选用的是MSE
        g_t = [g_t[i] for i in self.guide_layers]
        g_s = [g_s[i] for i in self.hint_layers]

        # visualize_feature(g_t,'original_student_feature_maps')
        # visualize_feature(g_s,'original_teacher_feature_maps')
        # visualize_feature(cross_features,'original_cross_feature_maps')
        a_t = []

        loss = []
        for i, h_t in enumerate(g_t):
            cross = cross_features[i]
            h_s = g_s[i]

            self.ReLU = nn.ReLU()
            q = self.ReLU(cross)
            k = self.ReLU(h_t)
            attention_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(k.size()[1]), dim=-1)

            # attention_weights = F.softmax(torch.matmul(cross, h_t.transpose(-2, -1)) / np.sqrt(h_t.size()[1]), dim=-1)

            # # 将注意力权重应用于学生特征图，以突出与教师特征图相关的部分
            a_t.append((h_s + torch.matmul(attention_weights, h_s))/2)
            h_s = F.normalize((h_s + torch.matmul(attention_weights, h_s))/2) # 如果将注意力权重作用于学生特征图(h_s)，那么学生特征图只是自我强化，而无法真正从教师特征图中获取新的信息。这违背了通过知识蒸馏让学生从教师中学习的初衷
            h_t = F.normalize(h_t) #实验证明，有正则化的效果比无正则化的效果好

            diff = self.cal_diff(h_s, h_t)
            loss.append(diff)


        # visualize_feature(a_t, 'original_abp_feature_maps')

        if Cross:
            cross_loss, cross_loss_index = [], []
            for i in range(len(loss)):
                # if i == 0:
                #     continue
                # pre = i - 1
                # pre_stu_attention = g_s[pre].mean(3).mean(2)
                # pre_tea_attention = g_t[pre].mean(3).mean(2)
                # pre_attention = torch.matmul(pre_stu_attention, pre_tea_attention.transpose(-2,-1)) / np.sqrt(pre_tea_attention.size()[1])
                # cur_stu_attention = g_s[i].mean(3).mean(2)
                # cur_tea_attention = g_t[i].mean(3).mean(2)
                # cur_attention = torch.matmul(cur_stu_attention, cur_tea_attention.transpose(-2,-1)) / np.sqrt(cur_tea_attention.size()[1])
                # pre_score = pre_attention.sum()
                # cur_score = cur_attention.sum()
                # if pre_score >= cur_score: # 注意力开关
                #     h_t = g_t[i]
                #     h_s = g_s[i]
                #     cross = cross_features[i]
                #
                #     # 计算MSE前先对特征图采用通道注意力，效果不好
                #     # bt = h_t.size(0)
                #     # bs = h_s.size(0)
                #     # bc = cross.size(0)
                #     # h_t = h_t.pow(2).mean(1).view(bt, -1)
                #     # h_s = h_s.pow(2).mean(1).view(bs, -1)
                #     # cross = cross.pow(2).mean(1).view(bc, -1)
                #
                #     if self.CrossMethod == 1:
                #         differ = self.cal_diff(F.normalize(cross), F.normalize(h_t))
                #     elif self.CrossMethod == 2:
                #         differ = self.cal_diff(F.normalize(h_s), F.normalize(cross))
                #     # MSE(g_t[i],cross_features[i])  Cross第一种损失
                #     # MSE(g_s[i],cross_features[i])  Cross第二种损失
                #     cross_loss.append(differ)
                #     cross_loss_index.append(i)
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
