import numpy as np
import matplotlib.pyplot as plt
import skimage.io

import os

def value_histogram(objects):
    from skimage.exposure import histogram
    hist, hist_centers = histogram(objects)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(objects, cmap=plt.cm.gray)
    axes[0].axis('off')
    axes[1].plot(hist_centers, hist, lw=2)
    axes[1].set_title('nb of pixels relative to their gray value')
    #Show result
    plt.show()

def thresholding(objects):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

    axes[0].imshow(objects > 0.12, cmap=plt.cm.gray)
    axes[0].set_title('obj > 0.12')

    axes[1].imshow(objects > 0.16, cmap=plt.cm.gray)
    axes[1].set_title('obj > 0.16')

    for a in axes:
        a.axis('off')

    plt.tight_layout()
    plt.show()

def edge_contour(objects):
    from skimage.feature import canny
    edges = canny(objects, sigma=1.0)
    return edges

def edge_fill(edges, disk_closing=1.5):
    from scipy import ndimage as ndi
    from skimage import morphology
    from skimage.morphology import disk  # noqa

    footprint = disk(disk_closing)
    fill_obj = morphology.closing(edges, footprint)

    #filling pass
    fill_obj = ndi.binary_fill_holes(fill_obj)
    return fill_obj

def fill_filter(fill):
    from skimage import morphology
    # small objects removal
    fill_obj = morphology.remove_small_objects(fill, min_size=256)
    return fill_obj

def edge_segment(objects):
    edges = edge_contour(objects)
    filled = edge_fill(edges, 1.5)
    filtered = fill_filter(filled)

    fig, axes = plt.subplots(1, 4, figsize=(12,4))
    axes[0].imshow(objects, cmap=plt.cm.gray)
    axes[0].axis('off')
    axes[0].set_title('original')

    axes[1].imshow(edges, cmap=plt.cm.gray)
    axes[1].set_title('Canny edge')
    axes[1].axis('off')

    axes[2].imshow(filled, cmap=plt.cm.gray)
    axes[2].set_title('Filling')
    axes[2].axis('off')

    axes[3].imshow(filtered, cmap=plt.cm.gray)
    axes[3].set_title('Filtered')
    axes[3].axis('off')

    return fig

def main():
    it=0
    for filename in os.listdir("Images"):
        objects = skimage.io.imread("Images/" + filename, as_gray=True)
    #    value_histogram(objects)
    #    thresholding(objects)
        plt.figure(it)
        fig = edge_segment(objects)
        it+=1
    plt.show()


main()