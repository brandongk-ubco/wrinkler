import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from datasets.WrinklerDataset import WrinklerDataset
from datasets.AugmentedDataset import DatasetAugmenter
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import albumentations as A
import sys

pl.seed_everything(42)
matplotlib.use('Agg')

datapath = "/mnt/d/work/datasets/wrinkler/"


class Segmenter(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = self.get_model()
        self.loss = self.get_loss()
        self.optimizer = self.get_optimizer()

    def get_model(self):
        return smp.UnetPlusPlus(encoder_name="efficientnet-b2",
                                encoder_weights="imagenet",
                                in_channels=3,
                                classes=4,
                                activation='softmax2d')

    def get_loss(self):
        return lambda y_hat, y: smp.losses.DiceLoss(
            smp.losses.constants.MULTILABEL_MODE, log_loss=True)(
                y_hat, y) + smp.losses.FocalLoss(smp.losses.constants.
                                                 MULTILABEL_MODE)(y_hat, y)

    def get_optimizer(self):
        return torch.optim.Adam

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

        imgs = x.cpu()
        predicted_masks = y_hat.cpu()
        masks = y.cpu()
        for i in range(imgs.shape[0]):
            img = imgs[i, :, :, :].numpy()

            mask = masks[i, :, :, :].numpy()
            mask_img = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
            mask_img[mask[1, :, :] == 1] = 50
            mask_img[mask[2, :, :] == 1] = 100
            mask_img[mask[3, :, :] == 1] = 200

            predicted_mask = predicted_masks[i, :, :, :].numpy()
            predicted_mask_img = np.zeros((img.shape[1], img.shape[2]),
                                          dtype=np.uint8)
            predicted_mask_img[predicted_mask[1, :, :] >= 0.9] = 50
            predicted_mask_img[predicted_mask[2, :, :] >= 0.9] = 100
            predicted_mask_img[predicted_mask[3, :, :] >= 0.9] = 200

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

            ax1.imshow(img.swapaxes(0, 2).swapaxes(0, 1))
            ax2.imshow(mask_img)
            ax3.imshow(predicted_mask_img)

            plt.savefig("{}_{}.png".format(batch_idx, i))
            plt.close()

        return {"val_loss", loss}

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=40,
                                                               min_lr=1e-5,
                                                               verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": 'val_loss'
        }


def get_dataloaders():
    train_data = WrinklerDataset(datapath, split="train")
    val_data = WrinklerDataset(datapath, split="val")
    batch_size = 4

    image_height = 1024
    image_width = 1024

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
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.PadIfNeeded(min_height=image_height,
                      min_width=image_width,
                      always_apply=True,
                      border_mode=0),
        A.CropNonEmptyMaskIfExists(image_height, image_width),
        ToTensorV2()
    ])

    train_data = DatasetAugmenter(train_data, train_transform)
    val_data = DatasetAugmenter(val_data, val_transform)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               num_workers=16,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             num_workers=16,
                                             shuffle=False)

    return train_loader, val_loader


if __name__ == '__main__':

    train_loader, val_loader = get_dataloaders()

    callbacks = [
        pl.callbacks.EarlyStopping('val_loss', patience=80),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ]

    try:
        callbacks.append(pl.callbacks.GPUStatsMonitor())
    except pl.utilities.exceptions.MisconfigurationException:
        pass

    trainer = pl.Trainer(gpus=1,
                         callbacks=callbacks,
                         min_epochs=10,
                         deterministic=True,
                         max_epochs=sys.maxsize)

    trainer.fit(Segmenter(), train_loader, val_loader)
