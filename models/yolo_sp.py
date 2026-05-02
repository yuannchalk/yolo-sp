import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from .adapter import StrongAdapter
import os

# 导入你的SuperPoint定义
import sys

sys.path.append("..")
from superpoint_lib.superpoint_offical import SuperPoint


class YOLOSP_Distiller(nn.Module):
    def __init__(self, yolo_weights=None, sp_weights=None, device='cuda'):
        super().__init__()
        self.device = device

        # 1. 构建/加载 YOLO
        if yolo_weights is not None and os.path.exists(yolo_weights):
            target_yolo_path = yolo_weights
        else:
            # 推理模式：直接用你已有的火焰权重来搭结构
            target_yolo_path = "D:\\python_file\\YOLO_SP_Porject\\weights\\yolov8n-fire.pt"
            if not os.path.exists(target_yolo_path):
                raise FileNotFoundError(f"找不到 {target_yolo_path}！请把你的yolo权重放在weights文件夹下。")

        print(f"[Init] Loading YOLO structure from: {target_yolo_path}")
        self.yolo = YOLO(target_yolo_path).model.to(device)
        self.yolo.eval()

        # 2. 构建 SuperPoint
        print(f"[Init] Building SP structure...")
        self.sp_teacher = SuperPoint().to(device)

        # 只有在训练模式且传了sp_weights时才加载SP权重
        if sp_weights is not None and os.path.exists(sp_weights):
            print(f"[Init] Loading SP weights for training...")
            sp_ckpt = torch.load(sp_weights, map_location=device)
            state_dict = sp_ckpt.get('model', sp_ckpt)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k[7:] if k.startswith('module.') else k
                new_state_dict[new_k] = v
            self.sp_teacher.load_state_dict(new_state_dict, strict=True)
            self._freeze_params()
        else:
            print("[Init] SP weight loading skipped (inference mode).")

        self.sp_teacher.eval()

        # 3. 初始化 Adapter
        self.adapter = StrongAdapter(in_channels=256, out_channels=128).to(device)

        # 4. 提取 Head
        self.sp_detector = self.sp_teacher.detector
        self.sp_descriptor = self.sp_teacher.descriptor

        # 5. 注册 Hook
        self.yolo_features = []
        self.sp_teacher_feat = None
        self._register_hooks()

    def _freeze_params(self):
        for p in self.yolo.parameters(): p.requires_grad = False
        for p in self.sp_teacher.parameters(): p.requires_grad = False
        for p in self.sp_detector.parameters(): p.requires_grad = False
        for p in self.sp_descriptor.parameters(): p.requires_grad = False
        print("[Init] Freezing all weights except Adapter.")

    def _register_hooks(self):
        def hook_yolo(module, input, output):
            if isinstance(output, torch.Tensor) and len(output.shape) == 4:
                self.yolo_features.append(output)

        for i, m in enumerate(self.yolo.model):
            m.register_forward_hook(hook_yolo)

        def hook_sp(module, input, output):
            self.sp_teacher_feat = output

        try:
            self.sp_teacher.backbone.register_forward_hook(hook_sp)
        except Exception as e:
            print(f"Warning: Failed to register SP hook: {e}")
        print("[Hook] All hooks attached.")

    def forward(self, x, mode='infer'):
        self.yolo_features = []
        self.sp_teacher_feat = None

        # 1. 过 YOLO
        yolo_out = self.yolo(x)

        # 2. 筛选 P3
        p3_feat = None
        candidates = []
        for feat in self.yolo_features:
            if feat.shape[1] == 256:
                candidates.append(feat)

        if len(candidates) > 0:
            candidates.sort(key=lambda x: x.shape[2], reverse=True)
            p3_feat = candidates[0]
        else:
            raise RuntimeError("Cannot find 256-channel feature.")

        # 3. 过 Adapter
        adapted_feat = self.adapter(p3_feat)

        # 4. 过 SP Head (学生模型)
        scores = self.sp_detector(adapted_feat)
        descs = F.normalize(self.sp_descriptor(adapted_feat), p=2, dim=1)

        # 5. 教师模型输出 (仅训练模式 distill 时)
        teacher_outputs = None
        if mode == 'distill' and self.training:
            with torch.no_grad():
                if x.shape[1] == 3:
                    scale = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
                    gray = (x * scale).sum(1, keepdim=True)
                else:
                    gray = x

                # 获取教师模型完整输出
                sp_out = self.sp_teacher({'image': gray})

                # 获取backbone特征
                teacher_feat = self.sp_teacher_feat

                # 获取检测热图 (中间结果，需要手动计算)
                teacher_backbone_feat = self.sp_teacher.backbone(gray)
                stride = self.sp_teacher.stride
                teacher_scores_raw = self.sp_teacher.detector(teacher_backbone_feat)
                teacher_scores = teacher_scores_raw[:, :1, :, :]

                # 处理成最终热图格式
                b, _, h, w = teacher_scores.shape
                teacher_scores = teacher_scores.permute(0, 2, 3, 1).reshape(b, h, w, stride, stride)
                teacher_scores = teacher_scores.permute(0, 1, 3, 2, 4).reshape(b, h * stride, w * stride)

                # 对齐尺寸
                if teacher_feat is not None and teacher_feat.shape[2:] != adapted_feat.shape[2:]:
                    teacher_feat = F.interpolate(teacher_feat, size=adapted_feat.shape[2:], mode='bilinear')

                teacher_outputs = {
                    'backbone_feat': teacher_feat,
                    'scores_for_distill': teacher_scores,
                    'descriptors': sp_out['descriptors'],
                    'keypoints': sp_out['keypoints'],
                    'keypoint_scores': sp_out['keypoint_scores']
                }

        return {
            "yolo_out": yolo_out,
            "p3_feat": p3_feat,
            "adapted_feat": adapted_feat,
            "teacher_outputs": teacher_outputs,
            "sp_scores": scores,
            "sp_descs": descs
        }
