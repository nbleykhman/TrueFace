import os
from torch.utils.data import Dataset
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        for label_str, class_idx in [('real', 0), ('fake', 1)]:
            folder = os.path.join(root_dir, label_str)
            for fn in os.listdir(folder):
                if fn.lower().endswith(('.jpg', '.png')):
                    self.images.append(os.path.join(folder, fn))
                    self.labels.append(class_idx)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
