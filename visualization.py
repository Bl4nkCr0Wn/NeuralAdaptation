import matplotlib.pyplot as plt
import numpy as np


def plot_data(generator, n_images, class_names):
    """
    Plots random data from dataset
    Args:
    generator: a generator instance
    n_images : number of images to plot
    """
    i = 1
    images, labels = generator.next()
    labels = labels.astype('int32')

    plt.figure(figsize=(14, 15))

    for image, label in zip(images, labels):
        plt.subplot(4, 3, i)
        plt.imshow(image)
        plt.title(class_names[np.argmax(label)])
        plt.axis('off')
        i += 1
        if i == n_images:
            break

    plt.show()

    # plot_data(train_generator, 7)
    # plot_data(validation_generator, 7)
    # plot_data(test_generator, 10)