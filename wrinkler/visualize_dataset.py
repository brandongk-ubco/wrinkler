from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from train import get_dataloaders, datapath, train_transform, val_transform, test_transform

matplotlib.use('Agg')

if __name__ == '__main__':
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        datapath, (train_transform, val_transform, test_transform))

    intensity = 255 // num_classes

    for j, (imgs, masks) in enumerate(train_loader):
        for i in range(imgs.shape[0]):
            img = imgs[i, :, :, :].numpy()
            mask = masks[i, :, :, :].numpy()

            mask_img = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
            for i in range(1, num_classes):
                mask_img[mask[i, :, :] > 0.5] = intensity * i

            fig, (ax1, ax2) = plt.subplots(2, 1)

            ax1.imshow(img.swapaxes(0, 2).swapaxes(0, 1))
            ax2.imshow(mask_img)

            ax1.axis('off')
            ax2.axis('off')

            plt.savefig("{}.png".format(j))
            plt.close()
        input("Press ENTER to continue")
