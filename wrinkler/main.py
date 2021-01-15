import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from datasets.WrinklerDataset import WrinklerDataset
from datasets.AugmentedDataset import DatasetAugmenter
import matplotlib
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
        return smp.Unet(encoder_name="efficientnet-b5",
                        encoder_weights="imagenet",
                        in_channels=3,
                        classes=4,
                        activation='softmax')

    def get_loss(self):
        return smp.utils.losses.BCELoss() + smp.utils.losses.DiceLoss()

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
        return {"val_loss", loss}

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=40,
                                                               min_lr=1e-5,
                                                               verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": 'val_loss'
        }


if __name__ == '__main__':

    train_data = WrinklerDataset(datapath, split="train")
    val_data = WrinklerDataset(datapath, split="val")
    batch_size = 10

    image_size = 512

    train_transform = A.Compose([
        A.PadIfNeeded(min_height=image_size,
                      min_width=image_size,
                      always_apply=True,
                      border_mode=0),
        A.CropNonEmptyMaskIfExists(
            image_size,
            image_size,
            always_apply=True,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.PadIfNeeded(min_height=image_size,
                      min_width=image_size,
                      always_apply=True,
                      border_mode=0),
        A.CropNonEmptyMaskIfExists(image_size, image_size),
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
