import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class CellsDataset(Dataset):
    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        self.data = [
            f for f in os.listdir(self.data_dir) if os.path.splitext(f)[1] == ".jpg"
        ]
        self.transform = transforms.ToTensor()

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        cell_map = {"RBC": 0, "WBC": 1, "Platelets": 2, "nan": 3, "FBC": 3}

        img_name = self.data[index]
        target = cell_map[img_name.split("_")[-2]]

        img = Image.open(os.path.join(self.data_dir, img_name))
        img = img.resize((100, 100))
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
