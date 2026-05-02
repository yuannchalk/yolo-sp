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


def load_teacher_model():
    """直接使用你的SuperPoint类加载权重"""
    from superpoint_pytorch import SuperPoint

    WEIGHT_PATH = 'weights/superpoint_official.pth'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Initializing model and loading weights from: {WEIGHT_PATH}")

    MODEL_CONF = {
        "nms_radius": 4,
        "max_num_keypoints": 500,
        "detection_threshold": 0.005,
        "remove_borders": 4,
    }

    model = SuperPoint(**MODEL_CONF).to(DEVICE).eval()

    checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)

    state_dict = checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    print("Teacher Model Loaded Successfully.")
    return model, DEVICE


def preprocess_image(img_gray):
    """预处理: numpy灰度图 -> 模型输入字典"""
    img = img_gray.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return {"image": img_tensor}


def get_random_homography(img_shape):
    """生成随机透视变换矩阵"""
    h, w = img_shape
    pts_orig = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    margin = min(w, h) * 0.15
    pts_warped = pts_orig + np.random.uniform(-margin, margin, (4, 2)).astype(np.float32)

    H = cv2.getPerspectiveTransform(pts_orig, pts_warped)
    return H


def warp_keypoints_back(keypoints, H):
    """将变换图中检测到的关键点，通过 H逆矩阵 投影回原图坐标系"""
    if len(keypoints) == 0:
        return np.array([])

    kp_homo = np.concatenate([keypoints, np.ones((len(keypoints), 1))], axis=1)
    H_inv = np.linalg.inv(H)
    kp_proj_homo = (H_inv @ kp_homo.T).T
    kp_proj = kp_proj_homo[:, :2] / (kp_proj_homo[:, [2]] + 1e-10)
    return kp_proj


# ==========================================
# 配置区域
# ==========================================
IMAGE_ROOT = 'datasets/fire_dataset/images/train'
SAVE_ROOT = 'datasets/fire_dataset/keypoints/train'
MATCH_DIST_THRESH = 3.0
os.makedirs(SAVE_ROOT, exist_ok=True)


def main():
    model, DEVICE = load_teacher_model()

    image_list = sorted([f for f in os.listdir(IMAGE_ROOT) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"Found {len(image_list)} images to process.")

    pbar = tqdm(image_list)
    for img_name in pbar:
        img_path = os.path.join(IMAGE_ROOT, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue
        H_orig, W_orig = img.shape[:2]

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        H_new = (H_orig // 8) * 8
        W_new = (W_orig // 8) * 8
        if H_new != H_orig or W_new != W_orig:
            img_gray = cv2.resize(img_gray, (W_new, H_new))

        H_mat = get_random_homography(img_gray.shape)
        img_warped = cv2.warpPerspective(img_gray, H_mat, (W_new, H_new))

        with torch.no_grad():
            data_orig = preprocess_image(img_gray)
            data_orig['image'] = data_orig['image'].to(DEVICE)
            pred_orig = model(data_orig)
            kp_orig = pred_orig['keypoints'][0].cpu().numpy()

            data_warped = preprocess_image(img_warped)
            data_warped['image'] = data_warped['image'].to(DEVICE)
            pred_warped = model(data_warped)
            kp_warped = pred_warped['keypoints'][0].cpu().numpy()

        final_ground_truth_kp = kp_orig

        if len(kp_orig) > 20 and len(kp_warped) > 20:
            kp_proj_to_orig = warp_keypoints_back(kp_warped, H_mat)

            N = min(len(kp_orig), 1000)
            M = min(len(kp_proj_to_orig), 1000)

            A = kp_orig[:N]
            B = kp_proj_to_orig[:M]

            dist = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)

            min_idx_1to2 = np.argmin(dist, axis=1)
            min_dist_1to2 = dist[np.arange(len(A)), min_idx_1to2]

            min_idx_2to1 = np.argmin(dist, axis=0)

            mutual_mask = (min_idx_2to1[min_idx_1to2] == np.arange(len(A)))
            dist_mask = (min_dist_1to2 < MATCH_DIST_THRESH)
            valid_mask = mutual_mask & dist_mask

            if np.sum(valid_mask) > 10:
                final_ground_truth_kp = A[valid_mask]

        save_dict = {
            'image_name': img_name,
            'image_size': (H_new, W_new),
            'keypoints': final_ground_truth_kp
        }

        base_name = Path(img_name).stem
        np.save(os.path.join(SAVE_ROOT, f"{base_name}.npy"), save_dict)

        if int(base_name[-3:]) % 50 == 0 or len(image_list) < 100:
            vis_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            for (x, y) in final_ground_truth_kp:
                cv2.circle(vis_img, (int(x), int(y)), 1, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(SAVE_ROOT, f"vis_{base_name}.jpg"), vis_img)

        pbar.set_description(f"Pts: {len(final_ground_truth_kp)}")


if __name__ == "__main__":
    main()
