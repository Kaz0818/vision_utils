# パスだけ受け取るのでメモリ効率がいい
class MyDataset(Dataset):
    def __init__(self, image_paths, labels, tranform=None):
        self.image_paths = imagespaths # <- パスだけ保存(軽い)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 必要な時だけ画像を読み込む
        image_path = self.image_paths[idx]
        labels = self.labels[idx]

        # ここで初めて画像を読み込み
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, label