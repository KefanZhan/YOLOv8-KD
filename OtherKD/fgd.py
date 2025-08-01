import torch.nn as nn
import torch.nn.functional as F
import torch

def xywh2xyxy(xywh):
    """
    将xywh格式的张量转换为xyxy格式
    :param xywh: 边界框的xywh格式张量，形状为 (n, 4)
    :return: 边界框的xyxy格式张量，形状为 (n, 4)
    """
    x, y, w, h = xywh.unbind(-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)

class FeatureLoss(nn.Module):

    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(list): Number of channels in the student's feature map.
        teacher_channels(list): Number of channels in the teacher's feature map.
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        device:
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 device,
                 # name,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.0005,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005,
                 ):
        super(FeatureLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        # if student_channels != teacher_channels:
        #     self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        # else:
        #     self.align = None

        self.align = []
        for i in range(len(teacher_channels)):
            student_channel, teacher_channel = student_channels[i], teacher_channels[i]
            if student_channel != teacher_channel:
                self.align.append(nn.Conv2d(student_channel, teacher_channel, kernel_size=1, stride=1, padding=0).to(device))


        # self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        # self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)

        self.conv_mask_s = [nn.Conv2d(teacher_channels[i], 1, kernel_size=1).to(device) for i in range(len(teacher_channels))]
        self.conv_mask_t = [nn.Conv2d(teacher_channels[i], 1, kernel_size=1).to(device) for i in range(len(teacher_channels))]


        # self.channel_add_conv_s = nn.Sequential(
        #     nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
        #     nn.LayerNorm([teacher_channels//2, 1, 1]),
        #     nn.ReLU(inplace=True),  # yapf: disable
        #     nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))
        # self.channel_add_conv_t = nn.Sequential(
        #     nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
        #     nn.LayerNorm([teacher_channels//2, 1, 1]),
        #     nn.ReLU(inplace=True),  # yapf: disable
        #     nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))

        self.channel_add_conv_s = [nn.Sequential(
            nn.Conv2d(teacher_channels[i], teacher_channels[i] // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels[i] // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels[i] // 2, teacher_channels[i], kernel_size=1)).to(device) for i in range(len(teacher_channels))]
        self.channel_add_conv_t = [nn.Sequential(
            nn.Conv2d(teacher_channels[i], teacher_channels[i] // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels[i] // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels[i] // 2, teacher_channels[i], kernel_size=1)).to(device) for i in range(len(teacher_channels))]

        self.reset_parameters()


    def forward(self,
                preds_S,
                preds_T,
                gt_bboxes,
                img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """

        # assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'

        if self.align is not None:
            # preds_S = self.align(preds_S)
            for i in range(len(preds_S)):
                preds_S[i] = self.align[i](preds_S[i])

        tmp = [[] for i in range(preds_S[0].shape[0])]
        for i in range(len(img_metas['batch_idx'])):
            index = int(img_metas['batch_idx'][i])
            gt_bbox_xyxy = xywh2xyxy(gt_bboxes[i]) # 将xywh转换成xyxy
            tmp[index].append(gt_bbox_xyxy)
        tmp = [torch.stack(boxes_list, dim=0) if boxes_list else [] for boxes_list in tmp]
        gt_bboxes = tmp

        total_loss = []
        for index in range(len(preds_S)):
            N,C,H,W = preds_S[index].shape

            S_attention_t, C_attention_t = self.get_attention(preds_T[index], self.temp)
            S_attention_s, C_attention_s = self.get_attention(preds_S[index], self.temp)

            Mask_fg = torch.zeros_like(S_attention_t)
            Mask_bg = torch.ones_like(S_attention_t)
            wmin,wmax,hmin,hmax = [],[],[],[]
            for i in range(N):
                if gt_bboxes[i] == []:  # 空 Tensor 或 []
                    # 填充占位符（避免后续索引越界）
                    hmin.append(torch.empty(0, dtype=torch.int32))
                    hmax.append(torch.empty(0, dtype=torch.int32))
                    wmin.append(torch.empty(0, dtype=torch.int32))
                    wmax.append(torch.empty(0, dtype=torch.int32))
                    continue  # 跳过后续处理
                new_boxxes = torch.ones_like(gt_bboxes[i]) # 如果某个批次的图片没有gt_box呢？
                # new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
                # new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
                # new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
                # new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H
                new_boxxes[:, 0] = gt_bboxes[i][:, 0]  * W
                new_boxxes[:, 2] = gt_bboxes[i][:, 2]  * W
                new_boxxes[:, 1] = gt_bboxes[i][:, 1]  * H
                new_boxxes[:, 3] = gt_bboxes[i][:, 3]  * H

                wmin.append(torch.floor(new_boxxes[:, 0]).int())
                wmax.append(torch.ceil(new_boxxes[:, 2]).int())
                hmin.append(torch.floor(new_boxxes[:, 1]).int())
                hmax.append(torch.ceil(new_boxxes[:, 3]).int())

                area = 1.0/(hmax[i].view(1,-1)+1-hmin[i].view(1,-1))/(wmax[i].view(1,-1)+1-wmin[i].view(1,-1))

                for j in range(len(gt_bboxes[i])):
                    Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = \
                            torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0][j])

                Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
                if torch.sum(Mask_bg[i]):
                    Mask_bg[i] /= torch.sum(Mask_bg[i])

            fg_loss, bg_loss = self.get_fea_loss(preds_S[index], preds_T[index], Mask_fg, Mask_bg,
                               C_attention_s, C_attention_t, S_attention_s, S_attention_t)
            mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
            rela_loss = self.get_rela_loss(preds_S[index], preds_T[index], index=index)


            loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss \
                   + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss
            total_loss.append(loss)
            
        return sum(total_loss)


    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W= preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False)
        C_attention = C * F.softmax(channel_map/temp, dim=1)

        return S_attention, C_attention


    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')
        
        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t= torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t)/len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t)/len(Mask_bg)

        return fg_loss, bg_loss


    def get_mask_loss(self, C_s, C_t, S_s, S_t):

        mask_loss = torch.sum(torch.abs((C_s-C_t)))/len(C_s) + torch.sum(torch.abs((S_s-S_t)))/len(S_s)

        return mask_loss
     
    
    def spatial_pool(self, x, in_type, index):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s[index](x)
        else:
            context_mask = self.conv_mask_t[index](x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context


    def get_rela_loss(self, preds_S, preds_T, index):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_S, 0, index)
        context_t = self.spatial_pool(preds_T, 1, index)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s[index](context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t[index](context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t)/len(out_s)
        
        return rela_loss


    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    
    def reset_parameters(self):
        # kaiming_init(self.conv_mask_s, mode='fan_in')
        # kaiming_init(self.conv_mask_t, mode='fan_in')
        for i in range(len(self.conv_mask_s)):
            kaiming_init(self.conv_mask_s[i], mode='fan_in')
            kaiming_init(self.conv_mask_t[i], mode='fan_in')

        # self.conv_mask_s.inited = True
        # self.conv_mask_t.inited = True

        # self.last_zero_init(self.channel_add_conv_s)
        # self.last_zero_init(self.channel_add_conv_t)
        for i in range(len(self.channel_add_conv_s)):
            self.last_zero_init(self.channel_add_conv_s[i])
            self.last_zero_init(self.channel_add_conv_t[i])


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)