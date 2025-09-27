from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from kornia.color import rgb_to_lab
from datasets import load_dataset
import os


def image_preprocessing(img, augment):
    aug = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=5, translate=(0.05,0.05), scale=(0.95,1.05)),
        transforms.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.06),
        # transforms.RandomErasing(p=0.15, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2)
    ]
    
    if augment:
        img = transforms.Compose(aug)(img)

    lab = rgb_to_lab(to_tensor(img))
    lab[0] /= 100.0
    lab[1:] = (lab[1:] + 128) / 255.0 * 2 - 1.0

    L, ab = lab[0:1], lab[1:]

    # L = (L - 0.449) / 0.226
    return L, ab


class HfDataset(Dataset):
    def __init__(self, dataset, split="train", augment=True, img_key="image"):
        self.dataset = load_dataset(dataset, split=split)
        self.augment = augment
        self.img_key = img_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx][self.img_key]
        return image_preprocessing(img, self.augment)
    

def load_coco_dataset(batchsize):
    train_dataset = HfDataset("ayushshah/coco-2017-image-colorization-224")
    train_loader = DataLoader(
        train_dataset, batch_size=batchsize, shuffle=True, num_workers=min(8, os.cpu_count()), pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )

    val_dataset = HfDataset("ayushshah/coco-2017-image-colorization-224", split="validation", augment=False)
    val_loader = DataLoader(
        val_dataset, batch_size=batchsize, shuffle=False, num_workers=min(8, os.cpu_count()), pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )

    return train_loader, val_loader
