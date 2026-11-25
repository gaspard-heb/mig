
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from pathlib import Path
from cycler import cycler
from pybaselines import Baseline


# Parametres SNIP
MAX_HALF_WINDOW = 40
SMOOTH_HALF_WINDOW = 10


# Longueur d'onde Cu Kalpha1
LAMBDA_ANGSTROM = 1.5406



# Palette de couleurs
plt.rcParams['axes.prop_cycle'] = cycler(
    color=[
        "firebrick", "purple", "darkmagenta",
        "darkviolet", "darkorchid", "indigo",
        "midnightblue", "navy", "darkblue",
        "mediumblue", "blue", "royalblue",
        "dodgerblue", "blueviolet", "mediumpurple"
    ]
)


def nom_fichier(numero_fichier):
    return f"Plaque_CMSX4_R1_30x10-0000{numero_fichier}.dat"


def LectureCHI_fichier(numero_fichier):
    # lecture du fichier retourne [x],[y]
    fichier = nom_fichier(numero_fichier)

    x = []
    y = []

    with open(fichier, 'r') as f:

        for ligne in f:

            ligne = ligne.strip()

            if not ligne or ligne.startswith('#'):
                continue

            valeurs = ligne.split()

            if len(valeurs) >= 2:

                try:
                    x.append(float(valeurs[0]))
                    y.append(float(valeurs[1]))

                except ValueError:
                    continue

    if len(x) == 0:
        raise ValueError("Aucune donnée valide dans le fichier.")

    return np.array(x), np.array(y)


def preparer_donnees(angle, intensite):
    # gere les potentielles donées incohérentes

    # Suppression des valeurs non finies
    masque = np.isfinite(angle) & np.isfinite(intensite)

    angle = angle[masque]
    intensite = intensite[masque]

    # Tri selon les angles
    ordre = np.argsort(angle)

    angle = angle[ordre]
    intensite = intensite[ordre]

    # Regroupement des angles dupliqués
    angles_uniques, indices = np.unique(
        angle, return_inverse=True
    )

    sommes = np.bincount(indices, weights=intensite)
    nombres = np.bincount(indices)

    intensites_uniques = sommes / nombres

    if len(angles_uniques) < 10:
        raise ValueError("Nombre de points insuffisant.")

    return angles_uniques, intensites_uniques


def estimation_fond(angle, intensite):

    # SNIP travaille avec une fenêtre exprimée
    # en nombre de points.

    signal_positif = np.maximum(intensite, 0)
    baseline = Baseline(x_data=angle)

    # methode snip
    fond_calcule, parametres = baseline.snip(signal_positif,max_half_window=MAX_HALF_WINDOW,decreasing=True,smooth_half_window=SMOOTH_HALF_WINDOW)
    # Retour aux angles expérimentaux
    fond = np.interp(angle,angle,fond_calcule)

    return fond


# =====================================================
# 5. AFFICHAGE DU DIFRACTOGRAMME
# =====================================================

def afficher_resultats(angle,intensite,fond,intensite_corrigee):

    fig, axes = plt.subplots(
        2, 1,
        figsize=(13, 9),
        sharex=True
    )

    # Premier graphique : fond estimé

    axes[0].plot(
        angle,
        intensite,
        color='black',
        linewidth=0.8,
        label='Diffractogramme expérimental'
    )

    axes[0].plot(
        angle,
        fond,
        color='red',
        linewidth=2,
        label='Fond estimé (SNIP)'
    )

    axes[0].set_ylabel('Intensité (coups)')
    axes[0].set_title(
        'Estimation automatique du bruit de fond DRX'
    )

    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Deuxième graphique : signal corrigé

    axes[1].plot(
        angle,
        intensite_corrigee,
        color='blue',
        linewidth=0.8,
        label='Signal corrigé'
    )

    axes[1].axhline(
        0,
        color='red',
        linestyle='--',
        linewidth=0.8
    )

    axes[1].set_xlabel(r'Angle $\theta$ (degrés)')
    axes[1].set_ylabel('Intensité corrigée')
    axes[1].set_title(
        'Diffractogramme après soustraction du fond'
    )

    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# =====================================================
# 7. PROGRAMME PRINCIPAL
# =====================================================

def traitement_DRX(numero_fichier):

    print("\n================================")
    print("       TRAITEMENT DRX")
    print("================================")

    print("Fichier :", nom_fichier(numero_fichier))

    # Lecture
    angle, intensite = LectureCHI_fichier(
        numero_fichier
    )

    # Préparation
    angle, intensite = preparer_donnees(
        angle,
        intensite
    )

    print("Nombre de points :", len(angle))

    # Estimation automatique
    print("\nEstimation du bruit de fond...")

    fond = estimation_fond(
        angle,
        intensite
    )

    # Soustraction du fond
    intensite_corrigee = intensite - fond

    # Affichage
    afficher_resultats(
        angle,
        intensite,
        fond,
        intensite_corrigee
    )

    print("\nTraitement terminé avec succès.")


def signal_sans_bruit(numero_fichier):
    # Lecture du fichier
    angle, intensite = LectureCHI_fichier(
        numero_fichier
    )

    # Préparation des données
    angle, intensite = preparer_donnees(
        angle,
        intensite
    )

    # Estimation du fond
    fond = estimation_fond(
        angle,
        intensite
    )

    # Soustraction du fond
    intensite_corrigee = intensite - fond

    return angle, intensite_corrigee


# detection
def detection_pics(angle,intensite,prominence_relative=0.01,distance=4,afficher=True,afficher_un_par_un=True):
    """
    Detecte les maxima locaux du diffractogramme.
    jouer sur la prominence_relative pour detecter des pics de maniere plus ou moins sensible
    """

    angle = np.asarray(angle)
    intensite = np.asarray(intensite)

    seuil = prominence_relative * np.max(intensite)
    indices,_ = find_peaks(intensite,prominence=seuil,distance=distance)
    
    if afficher :
        plt.plot(angle,intensite)
        for l in indices :
            plt.scatter(angle[l],intensite[l])
        if afficher_un_par_un :
            plt.show()
        
    return angle[indices]




def calcul_gamma(angle_pic):
    #on calcul le parametre de maille de gamma à a partir de l'angle 2*theta du premier pic associé {1,1,1}
    theta = np.radians(angle_pic / 2)
    d = LAMBDA_ANGSTROM / (2*np.sin(theta))
    return np.sqrt(3)*d


def calcul_gamma_bis(angle_pic):
    #on calcul le parametre de maille de gamma à a partir de l'angle 2*theta du premier pic associé {1,1,1}
    theta = np.radians(angle_pic / 2)
    d = LAMBDA_ANGSTROM / (2*np.sin(theta))
    return np.sqrt(8)*d



def calcul_alpha(angle_pic):
    theta = np.radians(angle_pic / 2)
    d = LAMBDA_ANGSTROM / (2*np.sin(theta))
    return 2*d




param_gamma = []
param_gamma_prime = []
param_alpha_a = []
param_alpha_c = []



for i in range(1,9):
    ang1,int1 = signal_sans_bruit(i)
    tab = detection_pics(ang1,int1,prominence_relative=0.0005,afficher=False)
    prime = True
    for l in range(len(tab)):
        #remarque importante : on calcul gamma prime sur le pic {1,1,1} si on ne le repare pas on utlise le pic {2,0,0} d'ou un ecart possible 
        if abs(tab[l]-42.45)<0.1:
            param_gamma.append(int(calcul_gamma(tab[l])*1000))
        elif abs(tab[l]-47.5)<0.2:
            param_alpha_a.append(int(calcul_alpha(tab[l])*1000))
        elif abs(tab[l]-50.9)<0.2:
            param_alpha_c.append(int(calcul_alpha(tab[l])*1000))
        elif abs(tab[l]-42.3)<0.1:
            param_gamma_prime.append(int(calcul_gamma(tab[l])*1000))
            prime = False
            print(prime,i)
        elif abs(tab[l]-72.05)<0.1 and prime:
            param_gamma_prime.append(int(calcul_gamma_bis(tab[l])*1000))
            print(i)
        









#main calcul des parmères de mailles
plt.show()
print(param_gamma)
print(param_alpha_a)
print(param_alpha_c)
print(param_gamma_prime)