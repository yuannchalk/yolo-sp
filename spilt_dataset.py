import os
import shutil
import random
from tqdm import tqdm

# ================= 配置区域 =================
# 【修改这里】你的原始数据所在的文件夹路径
SOURCE_IMG_DIR = "datasets/integrate fire_image(25448)/images"  # 原始图片文件夹
SOURCE_LABEL_DIR = "datasets/integrate fire_image(25448)/labels"  # 原始标签文件夹

# 【修改这里】输出的新数据集路径
OUTPUT_DIR = "datasets/fire"

# 分割比例 (总和必须为1.0)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1  # 如果不需要测试集，改成 0.0

# 随机种子，保证每次运行分割结果一样
SEED = 42


# ===========================================

def main():
    random.seed(SEED)

    # 1. 检查源文件夹是否存在
    if not os.path.exists(SOURCE_IMG_DIR) or not os.path.exists(SOURCE_LABEL_DIR):
        print("错误：找不到源图片或标签文件夹，请检查路径配置！")
        return

    # 2. 获取所有图片文件
    print(f"正在读取源数据: {SOURCE_IMG_DIR} ...")
    img_files = [f for f in os.listdir(SOURCE_IMG_DIR)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"共发现 {len(img_files)} 张图片。")

    # 3. 筛选出有对应标签的图片
    valid_files = []
    for img_file in img_files:
        # 获取文件名前缀 (不含后缀)
        prefix = os.path.splitext(img_file)[0]
        # 对应的标签文件名
        label_file = f"{prefix}.txt"

        if os.path.exists(os.path.join(SOURCE_LABEL_DIR, label_file)):
            valid_files.append(prefix)
        else:
            # print(f"警告: 找不到图片 {img_file} 对应的标签 {label_file}，已跳过。")
            pass

    print(f"其中 {len(valid_files)} 张图片有对应标签，将用于分割。")

    if len(valid_files) == 0:
        print("错误：没有找到匹配的图片-标签对！")
        return

    # 4. 打乱并分割
    random.shuffle(valid_files)

    total = len(valid_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_list = valid_files[:train_end]
    val_list = valid_files[train_end:val_end]
    test_list = valid_files[val_end:]

    print(f"分割结果:")
    print(f"  - 训练集 (Train): {len(train_list)} 张")
    print(f"  - 验证集 (Val):   {len(val_list)} 张")
    print(f"  - 测试集 (Test):  {len(test_list)} 张")

    # 5. 创建输出文件夹结构
    dirs = [
        os.path.join(OUTPUT_DIR, 'images', 'train'),
        os.path.join(OUTPUT_DIR, 'images', 'val'),
        os.path.join(OUTPUT_DIR, 'images', 'test'),
        os.path.join(OUTPUT_DIR, 'labels', 'train'),
        os.path.join(OUTPUT_DIR, 'labels', 'val'),
        os.path.join(OUTPUT_DIR, 'labels', 'test'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 6. 定义复制函数
    def copy_files(prefix_list, split_name):
        print(f"\n正在复制 {split_name} 集...")
        for prefix in tqdm(prefix_list):
            # 找图片文件 (可能是jpg也可能是png)
            img_src = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                temp_path = os.path.join(SOURCE_IMG_DIR, prefix + ext)
                if os.path.exists(temp_path):
                    img_src = temp_path
                    img_ext = ext
                    break

            if img_src is None:
                continue

            label_src = os.path.join(SOURCE_LABEL_DIR, prefix + '.txt')

            # 目标路径
            img_dst = os.path.join(OUTPUT_DIR, 'images', split_name, prefix + img_ext)
            label_dst = os.path.join(OUTPUT_DIR, 'labels', split_name, prefix + '.txt')

            # 复制 (使用 copy2 保留文件元数据)
            shutil.copy2(img_src, img_dst)
            shutil.copy2(label_src, label_dst)

    # 7. 执行复制
    copy_files(train_list, 'train')
    copy_files(val_list, 'val')
    if TEST_RATIO > 0:
        copy_files(test_list, 'test')

    print(f"\n完成！数据集已保存至: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()