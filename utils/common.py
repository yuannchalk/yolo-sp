"""Common utility functions shared across the project."""
import torch
import torch.nn.functional as F


def batched_nms(scores, nms_radius: int):
    """批处理非极大值抑制 (NMS)

    Args:
        scores: 置信度热图 [B, H, W] 或 [H, W]
        nms_radius: NMS 半径

    Returns:
        经过 NMS 后的热图
    """
    def max_pool(x):
        return F.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def decode_keypoints(scores, threshold=0.005, nms_radius=4, border=4, stride=8):
    """解码热图为关键点坐标

    Args:
        scores: 模型输出的原始分数 [B, C, H', W'] 或已处理的热图
        threshold: 检测阈值
        nms_radius: NMS 半径
        border: 边界丢弃像素数
        stride: 特征图 stride

    Returns:
        keypoints: 关键点坐标 [N, 2] (x, y)
    """
    # 处理输入格式
    if scores.dim() == 4 and scores.shape[1] == 65:
        # SuperPoint 原始输出格式 [B, 65, H, W]
        scores = scores[:, 0, :, :]
    elif scores.dim() == 4 and scores.shape[1] == 1:
        scores = scores.squeeze(1)
    elif scores.dim() == 3:
        pass
    else:
        raise ValueError(f"Unknown scores format: {scores.shape}")

    if scores.dim() == 2:
        scores = scores.unsqueeze(0)

    # 应用 NMS
    scores = batched_nms(scores.squeeze(0), nms_radius).unsqueeze(0)

    # 边界处理
    scores[:, :border, :] = -1
    scores[:, :, :border] = -1
    scores[:, -border:, :] = -1
    scores[:, :, -border:] = -1

    # 提取关键点
    scores = scores.squeeze(0)
    idxs = torch.where(scores > threshold)
    keypoints = torch.stack(idxs[::-1], dim=-1).float()
    return keypoints.cpu().numpy()


def letterbox(img, size=640):
    """Letterbox 图像预处理（保持宽高比）

    Args:
        img: 输入图像 [H, W, C]
        size: 输出尺寸（正方形）

    Returns:
        padded: 填充后的图像
        scale: 缩放比例
        pad_left: 左侧填充像素
        pad_top: 上侧填充像素
    """
    import cv2
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    pad_w, pad_h = size - nw, size - nh
    pl, pr = pad_w // 2, pad_w - pad_w // 2
    pt, pb = pad_h // 2, pad_h - pad_h // 2
    padded = cv2.copyMakeBorder(resized, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, pl, pt
