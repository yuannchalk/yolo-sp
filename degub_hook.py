import torch
from models.yolo_sp import YOLOSP_Distiller

# 初始化模型
model = YOLOSP_Distiller(
    yolo_weights='weights/yolov8n-fire.pt',
    sp_weights='weights/superpoint_v6_from_tf.pth',
    device='cpu'  # 先用cpu调试
)

# 伪造输入
dummy_input = torch.randn(1, 3, 640, 640)

# 前向传播
outputs = model(dummy_input)

print("Success!")
print(f"P3 Feature Shape: {outputs['p3_feat'].shape}")
print(f"Adapted Feature Shape: {outputs['adapted_feat'].shape}")
if outputs['teacher_outputs'] is not None:
    print(f"Teacher Feature Shape: {outputs['teacher_outputs']['backbone_feat'].shape}")
else:
    print("ERROR: Teacher outputs is None. Check SP hook.")