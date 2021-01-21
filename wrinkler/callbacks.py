from pytorch_lightning.callbacks import Callback
import numpy as np
import os
from matplotlib import pyplot as plt


class ValidationImageWriter(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx):
        import pdb
        pdb.set_trace()
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
