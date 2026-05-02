import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_descriptors_at_keypoints(descriptors, keypoints, stride=8):
    """
    在关键点位置采样描述子
    descriptors: [B, D, H, W] 描述子图
    keypoints: [N, 2] 关键点坐标 (x, y)
    stride: 特征图的stride
    """
    B, D, H, W = descriptors.shape
    device = descriptors.device

    # 将关键点坐标映射到特征图尺寸
    # keypoints是原图尺寸 (640x640)，需要映射到特征图尺寸 (HxW)
    keypoints_norm = (keypoints + 0.5) / torch.tensor([W * stride, H * stride], device=device)
    keypoints_norm = keypoints_norm * 2 - 1  # 归一化到 [-1, 1]

    # grid_sample需要的格式: [B, H_out, W_out, 2]
    grid = keypoints_norm.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)  # [B, 1, N, 2]

    # 采样
    B_out, D_out, H_out, W_out = grid.shape[0], D, grid.shape[2], grid.shape[1]

    # 调整descriptors格式以适配grid_sample
    descriptors_unsqueezed = descriptors.unsqueeze(1)  # [B, 1, D, H, W]

    # 使用grid_sample采样
    # grid格式需要是 [B, H, W, 2]，所以要转置
    grid_t = grid.permute(0, 2, 1, 3)  # [B, N, 1, 2]
    grid_t = grid_t.reshape(B * H_out, 1, 2)  # [B*N, 1, 2]

    # 扩展descriptors到batch维度
    descriptors_expanded = descriptors.unsqueeze(1).expand(B, H_out, D, H, W)
    descriptors_flat = descriptors_expanded.reshape(B * H_out, D, H, W)

    # 采样 (需要reshape回[B, N, D])
    sampled = F.grid_sample(descriptors_flat, grid_t, align_corners=False)
    sampled = sampled.squeeze(-1).t().reshape(B, D, H_out)

    return sampled


class OutputDistillationLoss(nn.Module):
    """
    输出级蒸馏损失：
    1. 检测热图对齐 (KL散度)
    2. 描述子对齐 (在关键点位置采样后对齐)
    """
    def __init__(self, temperature=3.0, desc_weight=0.5, heatmap_weight=1.0, stride=8):
        super().__init__()
        self.temp = temperature
        self.desc_weight = desc_weight
        self.heatmap_weight = heatmap_weight
        self.stride = stride
        self.mse = nn.MSELoss()

    def _align_heatmap(self, student_scores, teacher_scores):
        """
        对齐热图尺寸并计算损失
        student_scores: [B, 1, H, W] 学生模型输出的粗粒度热图
        teacher_scores: [B, 1, H', W'] 教师模型输出的细粒度热图
        """
        # 将教师热图resize到学生热图大小
        if teacher_scores.shape[2:] != student_scores.shape[2:]:
            teacher_scores = F.interpolate(
                teacher_scores,
                size=student_scores.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # 用softmax归一化，让响应值变成概率分布
        student_prob = F.softmax(student_scores / self.temp, dim=1)
        teacher_prob = F.softmax(teacher_scores / self.temp, dim=1)

        # KL散度
        loss_heatmap = F.kl_div(
            torch.log(student_prob + 1e-8),
            teacher_prob,
            reduction='batchmean'
        ) * (self.temp ** 2)

        return loss_heatmap

    def _align_descriptors(self, student_descs, teacher_keypoints, teacher_descs):
        """
        描述子对齐损失
        teacher_descs: [B, D, N] 关键点处的描述子，N是关键点数量
        teacher_keypoints: [N, 2] 关键点坐标
        student_descs: [B, D, H, W] 学生模型的描述子图

        在关键点位置采样学生描述子，然后对齐
        """
        B, D, H, W = student_descs.shape

        # 关键点数量N
        N = teacher_keypoints.shape[0] if teacher_keypoints.dim() == 1 else teacher_keypoints.shape[1]
        if N == 0:
            # 没有关键点，返回0损失（保持梯度流）
            return student_descs.new_zeros(())

        # 在关键点位置采样学生描述子
        # teacher_keypoints: [N, 2] -> [N, 2]
        student_sampled = sample_descriptors_at_keypoints(
            student_descs,
            teacher_keypoints.view(-1, 2),
            stride=self.stride
        )  # [B, D, N]

        # 对齐描述子
        # 教师描述子已经是归一化的 [B, D, N]
        # 学生描述子也需要归一化
        student_sampled_norm = F.normalize(student_sampled, p=2, dim=1)

        # 计算余弦相似度损失 (让关键点处的描述子相似)
        # similarity = (student_norm * teacher).sum(dim=1).mean()
        # 转换为距离损失
        cos_sim = (student_sampled_norm * teacher_descs).sum(dim=1)
        loss_desc = 1.0 - cos_sim.mean()

        return loss_desc

    def forward(self, student_scores, student_descs, teacher_outputs):
        """
        student_scores: [B, 1, H, W] 学生检测热图
        student_descs: [B, 256, H, W] 学生描述子图
        teacher_outputs: dict，教师模型输出
        """
        # 1. 热图对齐
        teacher_scores = teacher_outputs.get('scores_for_distill', None)
        if teacher_scores is not None:
            loss_heatmap = self._align_heatmap(student_scores, teacher_scores)
        else:
            loss_heatmap = student_scores.new_zeros(())

        # 2. 描述子对齐
        teacher_descs_list = teacher_outputs.get('descriptors', [])
        teacher_keypoints_list = teacher_outputs.get('keypoints', [])

        if len(teacher_descs_list) > 0 and len(teacher_keypoints_list) > 0:
            # 检查是否有有效的关键点
            valid_batch = False
            for i, (descs, kps) in enumerate(zip(teacher_descs_list, teacher_keypoints_list)):
                if descs.shape[1] > 0 and kps.shape[0] > 0:
                    valid_batch = True
                    break

            if valid_batch:
                # 合并batch维度
                teacher_descs = torch.stack([d.t() for d in teacher_descs_list], dim=0)  # [B, D, N]
                teacher_keypoints = torch.stack([k.float() for k in teacher_keypoints_list], dim=0)  # [B, N, 2]

                loss_desc = self._align_descriptors(student_descs, teacher_keypoints, teacher_descs)
            else:
                loss_desc = student_descs.new_zeros(())
        else:
            loss_desc = student_descs.new_zeros(())

        # 总损失
        total_loss = self.heatmap_weight * loss_heatmap + self.desc_weight * loss_desc

        return {
            'total': total_loss,
            'heatmap': loss_heatmap,
            'descriptor': loss_desc
        }


class DistillationLoss(nn.Module):
    """
    保留原有的中间特征蒸馏损失，作为备用或辅助损失
    """
    def __init__(self, temperature=3.0):
        super().__init__()
        self.temp = temperature
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, student_feat, teacher_feat):
        loss_mse = self.mse(student_feat, teacher_feat)

        B, C, H, W = student_feat.shape
        s_logits = student_feat.permute(0, 2, 3, 1).reshape(B, -1, C)
        t_logits = teacher_feat.permute(0, 2, 3, 1).reshape(B, -1, C)

        loss_kl = self.kl(
            F.log_softmax(s_logits / self.temp, dim=-1),
            F.log_softmax(t_logits / self.temp, dim=-1)
        ) * (self.temp ** 2)

        total_loss = loss_mse + 0.1 * loss_kl
        return total_loss
