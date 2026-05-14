import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from matplotlib import pyplot as plt

def visualize_feature(features, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for idx, feature in enumerate(features):
        if idx == len(features) - 1:
            break

        single_feature = feature[0]
        # 对通道进行均值处理，生成一个二维特征图
        combined_feature = single_feature.mean(dim=0).detach().cpu().numpy()  # 对通道取均值
        # 使用伪彩色映射（如 'jet'）将二维特征图保存为 RGB 图像
        save_path = os.path.join(save_dir, f'feature_map_{idx + 1}.png')
        plt.imsave(save_path, combined_feature, cmap='jet')  # 使用 'jet' 显示 RGB 图像


class CSAKD(nn.Module):
    def __init__(self, args):
        super(CSAKD, self).__init__()
        self.guide_layers = args.guide_layers
        self.hint_layers = args.hint_layers
        self.qk_dim = args.qk_dim

    def forward(self, g_s, g_t, cross_features):
        # g_s是学生的中间特征图，g_t是教师的中间特征图
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


            # # 将注意力权重应用于学生特征图，以突出与教师特征图相关的部分
            a_t.append((h_s + torch.matmul(attention_weights, h_s))/2)
            h_s = F.normalize((h_s + torch.matmul(attention_weights, h_s))/2)
            h_t = F.normalize(h_t)

            diff = self.cal_diff(h_s, h_t)
            loss.append(diff)


        # visualize_feature(a_t, 'original_abp_feature_maps')


        cross_loss, cross_loss_index = [], []
        for i in range(len(loss)):
            cross = cross_features[i]
            h_t = g_t[i]
            h_s = g_s[i]
            tea_channel_att = h_t.mean(3).mean(2)
            stu_channel_att = h_s.mean(3).mean(2)

            dot_product = torch.sum(tea_channel_att * stu_channel_att, dim=1)

            scaled_dot_product = dot_product / torch.sqrt(torch.tensor(tea_channel_att.size(1), dtype=torch.float32, device=tea_channel_att.device))
            average_weight = torch.mean(scaled_dot_product)

            final_weight = 1 - torch.sigmoid(average_weight)

            differ = self.cal_diff(F.normalize(h_s), F.normalize(cross))
            differ = differ * final_weight
            cross_loss.append(differ)
        return sum(loss), sum(cross_loss)

    def cal_diff(self, v_s, v_t):
        diff = (v_s - v_t).pow(2).mean(1).mean()
        return diff
