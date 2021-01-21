import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import numpy as np
import os
from matplotlib import pyplot as plt


class Segmenter(pl.LightningModule):
    def __init__(self, patience=10):
        super().__init__()
        self.model = self.get_model()
        self.loss = self.get_loss()
        self.optimizer = self.get_optimizer()
        self.patience = 10

    def get_model(self):
        return smp.DeepLabV3Plus(encoder_name="efficientnet-b0",
                                 encoder_weights="imagenet",
                                 in_channels=3,
                                 classes=4,
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
        imgs = x.clone().detach().cpu()
        predicted_masks = y_hat.clone().detach().cpu()
        masks = y.clone().detach().cpu()
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
            predicted_mask_img[predicted_mask[1, :, :] > 0.5] = 50
            predicted_mask_img[predicted_mask[2, :, :] > 0.5] = 100
            predicted_mask_img[predicted_mask[3, :, :] > 0.5] = 200

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
