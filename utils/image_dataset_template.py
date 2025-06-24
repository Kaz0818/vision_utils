# ===== データ準備テンプレート（コピペ用） =====
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

def setup_image_dataset(base_path_str, test_size=0.2):
    """
    画像データセット準備の完全自動化
    引数: base_path_str (str) - データセットのパス
    戻り値: 訓練・検証データとクラス情報
    """
    base_path = Path(base_path_str)
    
    # Step 1: 画像とラベル収集
    image_paths, labels = [], []
    for class_folder in base_path.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            for img_file in class_folder.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_paths.append(img_file)
                    labels.append(class_name)
    
    # Step 2: 分割
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42)
    
    # Step 3: クラスマッピング
    unique_classes = sorted(list(set(labels)))
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    train_labels_idx = [class_to_idx[label] for label in train_labels]
    val_labels_idx = [class_to_idx[label] for label in val_labels]
    
    # 結果表示
    print(f"総画像数: {len(image_paths)}")
    print(f"クラス数: {len(unique_classes)}")
    print(f"訓練用: {len(train_paths)}, 検証用: {len(val_paths)}")
    
    return {
        'train_paths': train_paths,
        'val_paths': val_paths, 
        'train_labels': train_labels_idx,
        'val_labels': val_labels_idx,
        'class_to_idx': class_to_idx,
        'num_classes': len(unique_classes)
    }

# ===== 使用方法（毎回これだけ） =====
# data_info = setup_image_dataset('/kaggle/input/syour-dataset/folder')

