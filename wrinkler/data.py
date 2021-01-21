import torch
from albumentations.pytorch import ToTensorV2
from datasets.WrinklerDataset import WrinklerDataset
from datasets.AugmentedDataset import DatasetAugmenter
import albumentations as A


def get_dataloaders(datapath, use_cache=True):
    train_data = WrinklerDataset(datapath, split="train", use_cache=use_cache)
    val_data = WrinklerDataset(datapath, split="val", use_cache=use_cache)
    test_data = WrinklerDataset(datapath, split="test", use_cache=use_cache)

    image_height = 1792
    image_width = 2048
    batch_size = 2

    train_transform = A.Compose([
        A.PadIfNeeded(min_height=image_height,
                      min_width=image_width,
                      always_apply=True,
                      border_mode=0),
        A.CropNonEmptyMaskIfExists(
            image_height,
            image_width,
            always_apply=True,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(transpose_mask=True)
    ])
    val_transform = A.Compose([
        A.PadIfNeeded(min_height=image_height,
                      min_width=image_width,
                      always_apply=True,
                      border_mode=0),
        A.CropNonEmptyMaskIfExists(image_height, image_width),
        ToTensorV2(transpose_mask=True)
    ])

    test_transform = A.Compose([ToTensorV2(transpose_mask=True)])

    train_data = DatasetAugmenter(train_data, train_transform)
    val_data = DatasetAugmenter(val_data, val_transform)
    test_data = DatasetAugmenter(test_data, test_transform)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               num_workers=8,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             num_workers=8,
                                             shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              num_workers=8,
                                              shuffle=False)

    return train_loader, val_loader, test_loader
