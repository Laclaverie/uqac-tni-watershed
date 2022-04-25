import numpy as np
from skimage.morphology import h_maxima
from scipy import ndimage as ndi

from skimage.segmentation import watershed


def watershedSegmentation(image):
    # utilisation fonction distance avec distance euclidienne
    distance = ndi.distance_transform_edt(image)

    # Selection des marqueurs en prenant les pics de distances les plus hauts (correspondant aux centres des grains)
    coords = h_maxima(distance, np.amax(distance) * 0.1)
    np.zeros(distance.shape, dtype=bool)

    # creation des marqueurs

    markers, _ = ndi.label(coords)

    # labellisation de l'image
    labels = watershed(-distance, markers, mask=image, watershed_line=True)

    return labels, distance
