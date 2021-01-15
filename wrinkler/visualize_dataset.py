from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from main import get_dataloaders

matplotlib.use('Agg')

if __name__ == '__main__':
    train_loader, val_loader = get_dataloaders()

    for imgs, masks in train_loader:
        for i in range(imgs.shape[0]):
            img = imgs[i, :, :, :].numpy()
            mask = masks[i, :, :, :].numpy()

            mask_img = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)

            mask_img[mask[1, :, :] == 1] = 50
            mask_img[mask[2, :, :] == 1] = 100
            mask_img[mask[3, :, :] == 1] = 200

            fig, (ax1, ax2) = plt.subplots(1, 2)

            ax1.imshow(img.swapaxes(0, 2).swapaxes(0, 1))
            ax2.imshow(mask_img)

            plt.savefig("{}.png".format(i))
            plt.close()
        input("Press ENTER to continue")
