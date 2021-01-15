import numpy as np
import torch


class AugmentedDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset.__getitem__(idx)
        image = np.array(image)
        mask = np.array(mask)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        image = image.float()
        mask = mask.float()

        image = image / 255.

        mask = torch.transpose(mask, 0, 2)
        mask = torch.transpose(mask, 1, 2)
        if image.size()[1:] != mask.size()[1:]:
            import pdb
            pdb.set_trace()

        return image, mask


class DatasetAugmenter(AugmentedDataset):
    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)

        mask[mask > 0] = 1.

        return image, mask
