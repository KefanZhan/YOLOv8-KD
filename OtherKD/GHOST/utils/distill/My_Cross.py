import torch.nn.functional as F
import torch


def Cross(g_s,g_t, cross_features, guide_layers, hint_layers):
    g_t = [g_t[i] for i in guide_layers]
    g_s = [g_s[i] for i in hint_layers]
    len_layers = len(g_t)

    cross_loss, cross_loss_index = [], []
    for i in range(len_layers):
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

        differ = cal_diff(F.normalize(h_s), F.normalize(cross))
        differ = differ * final_weight
        cross_loss.append(differ)
    return sum(cross_loss)

def cal_diff(v_s, v_t):
    diff = (v_s - v_t).pow(2).mean(1).mean()
    return diff