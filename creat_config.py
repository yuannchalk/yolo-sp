import yaml
import os

# 定义配置字典
config_dict = {
    "weights": {
        "yolo": "D:\\python_file\\YOLO_SP_Porject\\weights\\yolov8n-fire.pt",
        "superpoint": "D:\\python_file\\YOLO_SP_Porject\\weights\\superpoint_v6_from_tf.pth"
    },
    "model": {
        "input_size": 640
    },
    "data": {
        "train_img_dir": "datasets/fire/images/train"
    },
    "train": {
        "distill_epochs": 20,
        "batch_size": 8,
        "distill_lr": 0.001,
        "temperature": 3.0
    }
}

# 确保 configs 文件夹存在
os.makedirs("configs", exist_ok=True)

# 写入文件
save_path = "configs/config.yaml"
with open(save_path, 'w', encoding='utf-8') as f:
    # 使用 yaml.dump 写入
    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"配置文件已成功生成至: {save_path}")