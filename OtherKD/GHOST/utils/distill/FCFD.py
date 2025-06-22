import torch.nn as nn
import torch.nn.functional as F
import torch
from ultralytics.utils.tal import make_anchors, dist2bbox

def build_bridges():
    in_channels = [128,256,512]
    out_channels = [128,256,512]

    bridges = {"s2t": [], "t2s": []}
    for stage in range(len(in_channels)):
        bridge_s2t = nn.Sequential(*[
            nn.Conv2d(in_channels[stage], out_channels[stage], kernel_size=3, stride=1, padding=1,
                            bias=False),
            nn.BatchNorm2d(out_channels[stage]),
        ])
        bridges['s2t'].append(bridge_s2t)
        bridge_t2s = nn.Sequential(*[
            nn.Conv2d(out_channels[stage], in_channels[stage], kernel_size=3, stride=1, padding=1,
                            bias=False),
            nn.BatchNorm2d(in_channels[stage]),
        ])
        bridges['t2s'].append(bridge_t2s)
    for key in bridges:
        bridges[key] = nn.ModuleList(bridges[key])
    bridges = nn.ModuleDict(bridges)
    for m in bridges.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return bridges


class Fcfd(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, model):
        super(Fcfd, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def forward(self, g_s, g_t): # 按YOLO格式区分分类和回归的结果，再进行FCFD损失计算
        stu_feats = g_s[1] if isinstance(g_s, tuple) else g_s
        tea_feats = g_t[1] if isinstance(g_t, tuple) else g_t
        stu_pred_distri, stu_pred_score = torch.cat(
            [xi.view(stu_feats[0].shape[0], self.no, -1) for xi in stu_feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        tea_pred_distri, tea_pred_score = torch.cat(
            [xi.view(tea_feats[0].shape[0], self.no, -1) for xi in tea_feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        stu_pred_score = stu_pred_score.permute(0, 2, 1).contiguous()
        tea_pred_score = tea_pred_score.permute(0, 2, 1).contiguous()  # 分类结果
        stu_pred_distri = stu_pred_distri.permute(0, 2, 1).contiguous()
        tea_pred_distri = tea_pred_distri.permute(0, 2, 1).contiguous()

        stu_anchor_points, stu_stride_tensor = make_anchors(stu_feats, self.stride, 0.5)
        tea_anchor_points, tea_stride_teasor = make_anchors(tea_feats, self.stride, 0.5)
        stu_pred_bboxes = self.bbox_decode(stu_anchor_points, stu_pred_distri)
        tea_pred_bboxes = self.bbox_decode(tea_anchor_points, tea_pred_distri)

        loss_cls_kd = self.loss_cls_kd(stu_pred_score, tea_pred_score, T=1.0)
        loss_reg_kd = self.loss_reg_kd(stu_pred_bboxes, tea_pred_bboxes)
        return loss_cls_kd, loss_reg_kd

    # FCFD:
    # APP_WEIGHT: 1.0
    # KD_WEIGHT: 1.0
    # ROIL2_WEIGHT: 1.0
    # T: 1.0

    def loss_cls_kd(self, stu_logits, tea_logits, T=1.0):
        return F.kl_div(F.log_softmax(stu_logits / T, dim=1), F.softmax(tea_logits.detach() / T, dim=1), reduction='batchmean') * T * T * 1.0

    def loss_reg_kd(self, stu_bbox_offsets, tea_bbox_offsets):
        return F.mse_loss(stu_bbox_offsets, tea_bbox_offsets.detach()) * 1.0

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype).to(self.device))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)


