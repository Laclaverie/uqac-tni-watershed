import imageio
import skimage.filters
import skimage.color
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from skimage.morphology import square
from WatershedScikit import watershedSegmentation
from dataclasses import dataclass


### Fonctions ###
def affiche_image(titre, image):
    plt.figure()
    plt.imshow(image)
    plt.title(titre)

### Structures de données ###
@dataclass
class Grain:
    """Classe pour garder les attributs d'un grain (couleur moy, aire)"""
    avg_color: []
    area_covered: float
    def __init__(self):
        self.avg_color = [0,0,0] # [r, g, b]
        self.area_covered = 0.0  # l'aire qu'il occupe sur l'image (si besoin)
    def __str__(self):
        return f"color: [ r={self.avg_color[0]}, g={self.avg_color[1]}, b={self.avg_color[2]}] \t aire : {self.area_covered}"

### Execution du programme ###

# Lecture image
img = imageio.imread("Images/Echantillion1Mod2_301.png")

affiche_image('Image originale', img)

# conversion en niveau de gris
gray = skimage.color.rgb2gray(img)
laplace = skimage.filters.laplace(gray)

# affiche_image("Laplace", laplace)
# affiche_image("NDG", gray)

# Rehaussement des contours en soustrayant le laplacien de l'image a l'image d'origine
gray = gray - laplace
# affiche_image("Contours réhaussés", gray)

# Binarisation de l'image avec algorithme triangle
thresh = skimage.filters.threshold_triangle(gray)
binary = gray > thresh
# affiche_image("Threshold", binary)

#Application d'une érosion permettant de distinguer les grains
binary = morphology.erosion(binary, square(3))

# segmentation avec la ligne de partage des eaux
labels, distance = watershedSegmentation(binary)


# Affichage des resultats
fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(binary, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()

# mise en relation labels et image d'origine
im = skimage.color.label2rgb(labels, image=img, bg_label=0)
affiche_image("Labels", im)

#Repertorier les couleurs moyennes des grains
nb_grains = np.amax(labels)
grains=[] #Contiendra les données de grains (dataclass)

for i in range(nb_grains):
    label_i = (labels == i) #matrice de vrai/faux
    grain = Grain()
    #pour chaque couleur du grain
    for j in range(3):
        col_array = label_i * img[:, :, j] #multiplier par le masque de vrai/faux
        nb_pixels = np.count_nonzero(label_i) #diviseur
        grain.avg_color[j] = int( np.sum(col_array) / nb_pixels)   #Moyenne (int) (0-255)
        grain.area_covered = nb_pixels/labels.size #Supplémentaire: trouver l'aire couverte
    #Ajout de la donnee
    grains.append(grain)

#Affichage des données sur les grains sous forme de DataFrame
df=pd.DataFrame()
compteur = 1
for grain in grains:
    a = {'Moyenne de B': [grain.avg_color[2]], 'Moyenne de G': [grain.avg_color[1]], 'Moyenne de R': [grain.avg_color[0]]}
    transit = pd.DataFrame(a, index=[f"Grain isolé {compteur}"])
    df=pd.concat([df, transit])
    compteur+=1
print(df)
plt.show()


