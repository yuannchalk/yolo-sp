import os
import sys
import torch
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import batched_nms, letterbox


def decode_keypoints(scores, threshold=0.02, nms_radius=4, border=4):
    """
    解码热图为关键点坐标
    学生模型 sp_scores 输出格式: [B, 65, H, W] (detector原始输出)
    """
    stride = 8

    # 处理输入格式
    if scores.dim() == 4 and scores.shape[1] == 65:
        scores = scores[:, 0, :, :]
    elif scores.dim() == 4 and scores.shape[1] == 1:
        scores = scores.squeeze(1)
    elif scores.dim() == 3:
        pass
    else:
        raise ValueError(f"Unknown scores format: {scores.shape}")

    if scores.dim() == 2:
        scores = scores.unsqueeze(0)

    scores = batched_nms(scores.squeeze(0), nms_radius).unsqueeze(0)
    scores[:, :border, :] = -1
    scores[:, :, :border] = -1
    scores[:, -border:, :] = -1
    scores[:, :, -border:] = -1

    scores = scores.squeeze(0)
    idxs = torch.where(scores > threshold)
    keypoints = torch.stack(idxs[::-1], dim=-1).float()
    return keypoints.cpu().numpy()


def load_model_weights(model, checkpoint_path, device):
    """加载模型权重"""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading weights from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if 'adapter' in ckpt:
        model.adapter.load_state_dict(ckpt['adapter'], strict=True)
        print(f"Loaded adapter from epoch {ckpt.get('epoch', 'unknown')}")
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
        print("Loaded full model state_dict")
    else:
        print(f"Warning: Unknown checkpoint format")


# ==========================================
# 主程序
# ==========================================
def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    YOLO_WEIGHTS = os.path.join(PROJECT_ROOT, "weights", "yolov8n-fire.pt")
    ADAPTER_WEIGHTS = os.path.join(PROJECT_ROOT, "tools", "checkpoints", "best_adapter.pth")
    IMG_PATH = os.path.join(PROJECT_ROOT, "datasets", "testfire.jpg")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 第一步：纯YOLO检测
    # ==========================================
    print("[1/3] Running pure YOLO detection...")
    from ultralytics import YOLO
    yolo_model = YOLO(YOLO_WEIGHTS)
    yolo_results = yolo_model.predict(IMG_PATH, conf=0.25, verbose=False)

    orig_img = cv2.imread(IMG_PATH)
    boxes_orig = []
    confs_orig = []
    if len(yolo_results[0].boxes) > 0:
        boxes_orig = yolo_results[0].boxes.xyxy.cpu().numpy()
        confs_orig = yolo_results[0].boxes.conf.cpu().numpy()
    print(f"Detected {len(boxes_orig)} fire(s).")

    # ==========================================
    # 第二步：融合模型特征点提取
    # ==========================================
    print("[2/3] Running feature point extraction...")
    from models.yolo_sp import YOLOSP_Distiller

    model = YOLOSP_Distiller(yolo_weights=YOLO_WEIGHTS, device=DEVICE)
    load_model_weights(model, ADAPTER_WEIGHTS, DEVICE)
    model.eval()
    model.to(DEVICE)

    padded_img, scale, pl, pt = letterbox(orig_img)
    img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor, mode='infer')

    kps_640 = decode_keypoints(outputs['sp_scores'], threshold=0.02)
    print(f"Extracted {len(kps_640)} keypoints in total.")

    # ==========================================
    # 第三步：可视化
    # ==========================================
    print("[3/3] Visualizing...")
    draw_img = orig_img.copy()

    # 画YOLO检测框
    for i, (x1, y1, x2, y2) in enumerate(boxes_orig):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(draw_img, f"Fire {confs_orig[i]:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if len(kps_640) > 0:
            kps_resized = kps_640.copy()
            kps_resized[:, 0] = (kps_resized[:, 0] - pl) / scale
            kps_resized[:, 1] = (kps_resized[:, 1] - pt) / scale

            mask = (kps_resized[:, 0] >= x1) & (kps_resized[:, 0] <= x2) & \
                   (kps_resized[:, 1] >= y1) & (kps_resized[:, 1] <= y2)
            fire_kps = kps_resized[mask]

            for (x, y) in fire_kps:
                cv2.circle(draw_img, (int(x), int(y)), 2, (0, 255, 0), -1)
            print(f"  - Fire {i + 1}: {len(fire_kps)} keypoints inside")

    output_path = os.path.join(PROJECT_ROOT, "tools", "checkpoints", "final_result_robust.jpg")
    cv2.imwrite(output_path, draw_img)
    print(f"Done! Saved to {output_path}")

    cv2.imshow("Final Result", draw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
