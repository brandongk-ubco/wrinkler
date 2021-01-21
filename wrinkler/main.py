import pytorch_lightning as pl
import matplotlib
import sys
from data import get_dataloaders
from Model import Segmenter

pl.seed_everything(42)
matplotlib.use('Agg')

datapath = "/mnt/d/work/datasets/wrinkler/"
patience = 10

if __name__ == '__main__':

    train_loader, val_loader, test_loader = get_dataloaders(datapath)

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
                         min_epochs=10,
                         deterministic=True,
                         max_epochs=sys.maxsize)

    trainer.fit(Segmenter(patience=10), train_loader, val_loader)
