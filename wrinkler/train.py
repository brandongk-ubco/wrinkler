import pytorch_lightning as pl
import matplotlib
import sys
from datasets import cityscapes
from Model import Segmenter
import albumentations as A
from albumentations.pytorch import ToTensorV2

pl.seed_everything(42)
matplotlib.use('Agg')

patience = 10


def get_augments(image_height, image_width):
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

    return (train_transform, val_transform, test_transform)


if __name__ == '__main__':

    callbacks = [
        pl.callbacks.EarlyStopping('val_loss', patience=2 * patience),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ]

    try:
        callbacks.append(pl.callbacks.GPUStatsMonitor())
    except pl.utilities.exceptions.MisconfigurationException:
        pass

    trainer = pl.Trainer(gpus=1,
                         callbacks=callbacks,
                         min_epochs=patience,
                         deterministic=True,
                         max_epochs=sys.maxsize,
                         auto_scale_batch_size='binsearch')

    model = Segmenter(cityscapes, get_augments, patience=10)

    trainer.fit(model)
