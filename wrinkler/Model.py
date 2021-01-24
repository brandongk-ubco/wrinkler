import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import numpy as np
import os
from matplotlib import pyplot as plt


class Segmenter(pl.LightningModule):
    def __init__(self, num_classes, patience=10):
        super().__init__()
        self.loss = self.get_loss()
        self.optimizer = self.get_optimizer()
        self.patience = 10
        self.num_classes = num_classes + 1
        self.intensity = 255 // self.num_classes
        self.model = self.get_model()
        self.batches_to_write = 2

    def get_model(self):
        return smp.Unet(encoder_name="efficientnet-b0",
                        encoder_weights="imagenet",
                        in_channels=3,
                        classes=self.num_classes,
                        activation='softmax2d')

    def get_loss(self):
        return lambda y_hat, y: smp.losses.DiceLoss("multilabel")(
            y_hat, y) + smp.losses.FocalLoss("multilabel")(y_hat, y)

    def get_optimizer(self):
        return torch.optim.Adam

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.write_predictions(x, y, y_hat, batch_idx)
        return {"val_loss", loss}

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.patience, min_lr=1e-5, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": 'val_loss'
        }

    def write_predictions(self, x, y, y_hat, batch_idx):
        if batch_idx > self.batches_to_write:
            return

        imgs = x.clone().detach().cpu()
        predicted_masks = y_hat.clone().detach().cpu()
        masks = y.clone().detach().cpu()
        for i in range(imgs.shape[0]):
            img = imgs[i, :, :, :].numpy()

            mask = masks[i, :, :, :].numpy()
            mask_img = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
            for j in range(1, self.num_classes):
                mask_img[mask[j, :, :] == 1] = self.intensity * j

            predicted_mask = predicted_masks[i, :, :, :].numpy()
            predicted_mask_img = np.zeros((img.shape[1], img.shape[2]),
                                          dtype=np.uint8)
            for j in range(1, self.num_classes):
                predicted_mask_img[
                    predicted_mask[j, :, :] > 0.5] = self.intensity * j

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

            ax1.imshow(img.swapaxes(0, 2).swapaxes(0, 1))
            ax2.imshow(mask_img)
            ax3.imshow(predicted_mask_img)
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')

            plt.savefig(
                os.path.join(self.logger.log_dir,
                             "{}_{}.png".format(batch_idx, i)))
            plt.close()
