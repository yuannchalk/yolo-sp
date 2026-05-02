import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import batched_nms, letterbox


def decode_keypoints(scores, threshold=0.005, nms_radius=4, border=4, stride=8):
    """解码热图为关键点坐标"""
    b, _, h, w = scores.shape
    scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
    scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, stride, stride)
    scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * stride, w * stride)
    scores = batched_nms(scores, nms_radius)
    scores[:, :border, :] = -1
    scores[:, :, :border] = -1
    scores[:, -border:, :] = -1
    scores[:, :, -border:] = -1
    scores = scores.squeeze(0)
    idxs = torch.where(scores > threshold)
    keypoints = torch.stack(idxs[::-1], dim=-1).float()
    return keypoints.cpu().numpy()


# ==========================================
# 配置区域
# ==========================================
YOLO_WEIGHTS = "D:\\python_file\\YOLO_SP_Porject\\weights\\yolov8n-fire.pt"
FUSION_WEIGHTS = "D:\\python_file\\YOLO_SP_Porject\\tools\\checkpoints\\yolo_sp_slim.pth"
IMG_PATH = "D:\\python_file\\YOLO_SP_Porject\\datasets\\testfire.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # ==========================================
    # 第一步：用纯 YOLO 跑检测
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
    # 第二步：用融合模型跑特征点
    # ==========================================
    print("[2/3] Running feature point extraction...")
    from models.yolo_sp import YOLOSP_Distiller

    model = YOLOSP_Distiller(yolo_weights=YOLO_WEIGHTS, device=DEVICE)
    ckpt = torch.load(FUSION_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    model.to(DEVICE)

    padded_img, scale, pl, pt = letterbox(orig_img)
    img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor, mode='infer')

    kps_640 = decode_keypoints(outputs['sp_scores'])
    print(f"Extracted {len(kps_640)} keypoints in total.")

    # ==========================================
    # 第三步：可视化融合 (展示所有点)
    # ==========================================
    print("[3/3] Visualizing...")
    draw_img = orig_img.copy()

    all_kps_orig = np.array([])
    if len(kps_640) > 0:
        all_kps_orig = kps_640.copy()
        all_kps_orig[:, 0] = (all_kps_orig[:, 0] - pl) / scale
        all_kps_orig[:, 1] = (all_kps_orig[:, 1] - pt) / scale

    if len(all_kps_orig) > 0:
        for (x, y) in all_kps_orig:
            cv2.circle(draw_img, (int(x), int(y)), 1, (255, 0, 0), -1)

    for i, (x1, y1, x2, y2) in enumerate(boxes_orig):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(draw_img, f"Fire {confs_orig[i]:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if len(all_kps_orig) > 0:
            mask = (all_kps_orig[:, 0] >= x1) & (all_kps_orig[:, 0] <= x2) & \
                   (all_kps_orig[:, 1] >= y1) & (all_kps_orig[:, 1] <= y2)
            fire_kps = all_kps_orig[mask]

            for (x, y) in fire_kps:
                cv2.circle(draw_img, (int(x), int(y)), 2, (0, 255, 0), -1)

            print(f"  - Fire {i + 1}: {len(fire_kps)} keypoints inside")

    output_path = os.path.join(os.path.dirname(FUSION_WEIGHTS), "final_result_all_keypoints.jpg")
    cv2.imwrite(output_path, draw_img)
    print(f"Done! Saved to {output_path}")

    cv2.imshow("Final Result (All Keypoints)", draw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
