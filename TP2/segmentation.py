import numpy as np
import cv2
import skimage
from matplotlib import pyplot as plt
from skimage import segmentation
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries
from skimage import exposure
import argparse

from dataclasses import dataclass


### Fonctions ###
def affiche_image(titre, image, cmap=None):
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.title(titre)


### Structures de données ###
@dataclass
class Grain:
    """Classe pour garder les attributs d'un grain (couleur moy, aire)"""
    avg_color: []
    area_covered: float

    def __init__(self):
        self.avg_color = [0, 0, 0]  # [r, g, b]
        self.area_covered = 0.0  # l'aire qu'il occupe sur l'image (si besoin)

    def __str__(self):
        return f"color: [ r={self.avg_color[0]}, g={self.avg_color[1]}, b={self.avg_color[2]}] \t aire : {self.area_covered}"


### Fonctions de pré-traitement ###

def preprocessing(img_init):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_erode = cv2.erode(img_init, kernel, iterations=1)
    img_dilate = cv2.dilate(img_erode, kernel, iterations=3)
    img = cv2.pyrMeanShiftFiltering(img_dilate, 15, 30)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def CLAHE(image_gray):  # Demande l'image pré-traitée en entrée
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(7, 7))
    cl1 = clahe.apply(image_gray)
    return cl1


def adjust_brigthness_contrast(img, alpha=2.2, beta=40):
    new_image = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_image[y, x, c] = np.clip(alpha * img[y, x, c] + beta, 0, 255)
    return new_image


def histogramme_equalize(img_gray):  # Demande l'image pré-traitée en entrée
    eq_img = exposure.equalize_adapthist(img_gray)
    plt.hist(eq_img.flat, bins=100, range=(0, 1))
    return eq_img


### Affichage des thresholds ###
def All_threshold(img_gray):  # Affichage pur  -  # Demande l'image pré-traitée en entrée
    im = skimage.filters.try_all_threshold(img_gray)
    skimage.io.show()


def histogramm(img_gray):  # Affichage pur - Affiche l'histogramme # Demande l'image pré-traitée en entrée
    hist, hist_centers = skimage.exposure.histogram(img_gray)
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 3.2), sharex=True, sharey=True)
    ax1.plot(hist_centers, hist, lw=2)
    fig.tight_layout()
    plt.show()


def binary_threshold(img_gray):  # Affichage pur # Demande l'image pré-traitée en entrée
    fig_s, ((ax1_s, ax2_s), (ax3_s, ax4_s)) = plt.subplots(2, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1_s.imshow(img_gray > 20, cmap='gray')
    ax1_s.axis('off')
    ax1_s.set_title('gray>20')
    ax2_s.imshow(img_gray > 40, cmap='gray')
    ax2_s.axis('off')
    ax2_s.set_title('gray>40')
    ax3_s.imshow(img_gray > 50, cmap='gray')
    ax3_s.axis('off')
    ax3_s.set_title('gray>50')
    ax4_s.imshow(img_gray > 80, cmap='gray')
    ax4_s.axis('off')
    ax4_s.set_title('gray>80')
    fig_s.tight_layout()
    plt.show()


### Méthodes de segmentation ###

# fonction qui retourne l'image de segmentation binaire avec seuil =50
def threshold(img_gray, seuil=50):  # Demande l'image pré-traitée en entrée
    # im = mark_boundaries(image,gray)
    gray = img_gray > seuil
    return skimage.measure.label(gray)


def felzenswalb(image_rgb):  # Demande l'image pré-traitée en entrée RGB
    seg = skimage.segmentation.felzenszwalb(image_rgb, scale=200, sigma=3, min_size=200)
    return skimage.measure.label(seg)


def random_walker(img_gray):  # Demande l'image pré-traitée en entrée Gris
    markers = np.zeros_like(img_gray, dtype=np.uint8)
    markers[(img_gray < 50)] = 1
    markers[(img_gray > 55)] = 2
    labels = segmentation.random_walker(img_gray, markers, beta=10, mode='bf')
    # im=mark_boundaries(labels,markers)
    return skimage.measure.label(labels)


def canny_fill(img_gray):  # Demande l'image pré-traitée en entrée Gris
    from skimage.feature import canny
    from scipy import ndimage as ndi
    from skimage import morphology
    from skimage.morphology import disk

    # Détection des contours
    edges = canny(img_gray, sigma=1.0)
    # Opération de closing
    disk_closing = 2.0
    footprint = disk(disk_closing)
    closed = morphology.closing(edges, footprint)
    # Remplissage
    fill_obj = ndi.binary_fill_holes(closed)
    # Suppression des petits objets
    filtered = morphology.remove_small_objects(fill_obj, min_size=256)

    return skimage.measure.label(filtered)


def watershed(img_raw):  # Demande l'image non traitée en entrée
    from skimage import io
    from skimage import morphology
    from skimage.morphology import square
    from skimage.morphology import h_maxima
    from scipy import ndimage as ndi

    img_gray = skimage.color.rgb2gray(img_raw)
    laplace = skimage.filters.laplace(img_gray)

    # Rehaussement des contours en soustrayant le laplacien de l'image a l'image d'origine
    gray = img_gray - laplace
    # affiche_image("Contours réhaussés", gray)

    # Binarisation de l'image avec algorithme triangle
    thresh = skimage.filters.threshold_triangle(gray)
    binary = gray > thresh
    # affiche_image("Threshold", binary)

    # Application d'une érosion permettant de distinguer les grains
    binary = morphology.erosion(binary, square(3))

    # Segmentation avec la ligne de partage des eaux
    # utilisation fonction distance avec distance euclidienne
    distance = ndi.distance_transform_edt(binary)
    # Selection des marqueurs en prenant les pics de distances les plus hauts (correspondant aux centres des grains)
    coords = h_maxima(distance, np.amax(distance) * 0.1)
    np.zeros(distance.shape, dtype=bool)
    # creation des marqueurs
    markers, _ = ndi.label(coords)
    # labellisation de l'image
    labels = skimage.segmentation.watershed(-distance, markers, mask=binary, watershed_line=True)

    affiche_image("Distances", distance)
    return labels


def slic_and_quickshift_pretraitment(img):
    from skimage.util import img_as_float
    kernel = np.ones((5, 5), np.uint8)
    image = threshold(img)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.dilate(image, kernel, iterations=1)
    image = CLAHE(image, grid=(7, 7), cliplimit=3.0)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return img_as_float(image)


# SLIC - entrée = raw image cv2
def slic(img):
    from skimage.segmentation import slic
    from skimage.segmentation import mark_boundaries
    segments_slic = slic(img, n_segments=100, compactness=10, sigma=2.0, start_label=1, convert2lab=True, max_iter=30,
                         enforce_connectivity=True)

    return segments_slic  # mark_boundaries(imgRef, segments_slic)


# QuickShift - entrée = raw image cv2
def quickshift(img):
    from skimage.segmentation import quickshift
    from skimage.segmentation import mark_boundaries
    segments_quick = quickshift(img, kernel_size=9, max_dist=22, ratio=1.2, sigma=1.0)
    return segments_quick  # mark_boundaries(imgRef, segments_quick)


### Fonctions du dataframe ###

# Repertorier les couleurs moyennes des grains
def moyenne_grains(labels, img_base):
    # Repertorier les couleurs moyennes des grains
    nb_grains = int(np.amax(labels))
    print(nb_grains, "grains identifiés sur l'image")
    grains = []  # Contiendra les données de grains (dataclass)

    for i in range(nb_grains):
        label_i = (labels == i)  # matrice de vrai/faux
        nb_pixels = np.count_nonzero(label_i)  # diviseur
        grain = Grain()
        if nb_pixels == 0: continue
        # pour chaque couleur du grain
        for j in range(3):
            col_array = label_i * img_base[:, :, j]  # multiplier par le masque de vrai/faux
            grain.avg_color[j] = int(np.sum(col_array) / nb_pixels)  # Moyenne (int) (0-255)
            grain.area_covered = nb_pixels / labels.size  # Supplémentaire: trouver l'aire couverte
        # Ajout de la donnee
        grains.append(grain)

    return grains


# Affichage des données sur les grains sous forme de DataFrame
def releve_grains(img_base, labels, save_to_file=False, out_csv_name=""):
    import pandas as pd
    # Liste de dataclass
    grains = moyenne_grains(labels, img_base)

    # Construction du dataframe
    df = pd.DataFrame()
    cpt = 1
    for grain in grains:
        a = {'Moyenne de B': [grain.avg_color[2]], 'Moyenne de G': [grain.avg_color[1]],
             'Moyenne de R': [grain.avg_color[0]]}
        transit = pd.DataFrame(a, index=[f"Grain isolé {cpt}"])
        df = pd.concat([df, transit])
        cpt += 1
    print(df)

    # Sauvegarde dans un fichier (si booléen actif)
    if (save_to_file):
        filename = out_csv_name
        filename = filename.replace('Images/', 'CSV_')
        filename = filename.replace('.png', '.csv')
        df.to_csv("resultats/" + filename, sep=';', encoding="ISO-8859-1")
        print("Fichier", "'resultats/" + filename + "'", "sauvegardé")


### Execution du programme ###
def cv2_to_skimage(cv2_bgr):
    # https://stackoverflow.com/questions/66360041/how-to-convert-cv2-image-to-skimage
    return np.uint8(0.2125 * cv2_bgr[..., 2] + 0.7154 * cv2_bgr[..., 1] + 0.0721 * cv2_bgr[..., 0])


def launch_segmentation(seg, cv2_processed_tuple, ski_raw):
    from skimage import img_as_ubyte
    # Images cv2
    cv2_raw, cv2_rgb_preprocess, cv2_gray_preprocess = cv2_processed_tuple
    # Images skimage
    ski_gray_preprocess = img_as_ubyte(cv2_gray_preprocess)

    if seg == 'thresholding' or seg == 'threshold': return threshold(ski_gray_preprocess)
    if seg == 'rwalker': return random_walker(cv2_gray_preprocess)
    if seg == 'watershed': return watershed(ski_raw)
    if seg == 'canny': return canny_fill(ski_gray_preprocess)
    if seg == 'slic': return slic(cv2_raw)
    if seg == 'quickshift': return quickshift(cv2_raw)
    if seg == 'felzen' or seg == 'felzenszwalb': return threshold(ski_gray_preprocess)
    return


def main():
    import time  # pour benchmark
    import skimage.io
    # Recuperation des arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("img_num", help="Enter the number of the image : 301, 302, 416 ...")
    parser.add_argument("seg_method",
                        help="Segmentation method: thresholding, rwalker, watershed, canny, slic, quickshift, felzen")

    args = parser.parse_args()
    print(args)
    img_num = args.img_num
    seg_method = args.seg_method

    # Ouverture de l'image
    img_path = f"Images/Echantillion1Mod2_{img_num}.png"

    img_cv2 = cv2.imread(img_path)
    img_ski = skimage.io.imread(img_path)
    # Pré-traitement
    processed_rgb, processed_gray = preprocessing(img_cv2)

    img_RGB = img_cv2[:, :, ::-1]
    affiche_image("Image pré-traitée", img_RGB)

    # Méthode selectionnée (argument)
    start = time.time()
    resultat = launch_segmentation(seg_method, (img_cv2, processed_rgb, processed_gray), img_ski)
    print(f"Résultat obtenu en {time.time() - start} secondes")
    # Affichage
    affiche_image("Resultat du traitement", resultat, plt.cm.gray)

    # mise en relation labels et image d'origine
    im = skimage.color.label2rgb(resultat, image=img_RGB, bg_label=0)
    affiche_image("Labels", im)
    plt.show()

    # Moyennes et dataframe
    releve_grains(img_base=img_RGB, labels=resultat, save_to_file=True, out_csv_name=img_path)

    return


if __name__ == "__main__":
    main()
