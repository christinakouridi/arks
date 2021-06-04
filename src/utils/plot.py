from matplotlib import pyplot as plt

FASHION_LABELS = {
        0: 't-shirt',
        1: 'trouser',
        2: 'pullover',
        3: 'dress',
        4: 'coat',
        5: 'sandal',
        6: 'shirt',
        7: 'sneaker',
        8: 'bag',
        9: 'boot',
    }


CIFAR_LABELS = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck',
    }


def plot_images(X, y, yp, M, N, data='fashion_mnist', task='multi'):
    """
    Plots a grid of Fashion-MNIST, CIFAR-10 or CelebA images with model predictions. Blue frames correspond to
    correct predictions, and red frames wrong predictions. The true label is indicated at the top of each image.

    Parameters
    ----------
    X : images
    y : labels
    yp : predictions
    M : number of rows for the grid of images
    N: number of columns for the grip of images
    data : dataset name
    task : classification task (we use 'multi' for all datasets)

    Returns
    -------
    fig : figure with grid of images
    """

    f, ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N, M * 1.3))

    for i in range(M):
        for j in range(N):

            predicted_label = get_label(yp[i * N + j], task=task)
            if data == 'fashion_mnist':
                predicted_label = FASHION_LABELS[predicted_label.item()]
                sample = 1 - X[i * N + j][0].cpu().numpy()
                cmap = 'gray'

            elif data in ['cifar_10', 'celeba']:
                if data == 'cifar_10':
                    predicted_label = CIFAR_LABELS[predicted_label.item()]
                sample = X[i * N + j].cpu().numpy().transpose((1, 2, 0))
                cmap = None

            ax[i][j].imshow(sample, cmap=cmap, interpolation=None)

            title = ax[i][j].set_title(f"{predicted_label}")
            plt.setp(title, color=('b' if compare_label(yp[i * N + j], y[i * N + j], task=task) else 'r'))

            for label in ax[i][j].get_xticklabels():
                label.set_visible(False)
            for label in ax[i][j].get_yticklabels():
                label.set_visible(False)

            # put a red border around the wrong predictions, and a blue border around the correct ones
            if compare_label(yp[i * N + j], y[i * N + j], task=task):
                c_frame = 'b'
            else:
                c_frame = 'r'

            for axis in ['top', 'bottom', 'left', 'right']:
                ax[i][j].spines[axis].set_linewidth(3.0)
                ax[i][j].spines[axis].set_color(c_frame)

    plt.tight_layout()
    plt.show()
    return f


def compare_label(yp, y, task='binary'):
    """
    Compares the true label and predicted label for a batch of samples

    Parameters
    ----------
    yp : predictions
    y : labels
    task : classification task (we use 'multi' for all datasets)

    Returns
    -------
    res : for example batch sample, it returns True if the predicted label equals the true label, otherwise False
    """

    if task == 'binary':
        res = 1 - int((yp > 0) * (y == 0) + (yp < 0) * (y == 1))
    elif task == 'multi':
        res = yp.max(dim=0)[1] == y
    else:
        raise NotImplementedError
    return res


def get_label(yp, task='binary'):
    """
    Gets the predicted labels for a batch of samples

    Parameters
    ----------
    yp : predictions
    task : classification task (we use 'multi' for all datasets)

    Returns
    -------
    res : predicted labels
    """

    if task == 'binary':
        res = int(yp > 0)
    elif task == 'multi':
        res = yp.max(dim=0)[1]
    else:
        raise NotImplementedError
    return res