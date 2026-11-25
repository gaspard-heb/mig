import numpy as np
import matplotlib.pyplot as plt
import lmfit as lf
from pathlib import Path
from cycler import cycler
from pybaselines import Baseline
from scipy.signal import find_peaks


# Parametres SNIP
MAX_HALF_WINDOW = 40
SMOOTH_HALF_WINDOW = 10


# Longueur d'onde Cu Kalpha1
LAMBDA_ANGSTROM = 1.5406

# Energie correspondante en keV
ENERGIE_KEV = 8.047


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
    return f"BRX_plaques/Plaque_CMSX4_R1_30x10-0000{numero_fichier}.dat"


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
    # On vérifie donc l'espacement angulaire.

    pas = np.diff(angle)

    pas_moyen = np.median(pas)

    variation = np.max(
        np.abs(pas - pas_moyen)
    ) / pas_moyen

    signal_positif = np.maximum(
        intensite, 0
    )

    baseline = Baseline(
        x_data=angle
    )

    # methode snip
    fond_calcule, parametres = baseline.snip(
        signal_positif,
        max_half_window=MAX_HALF_WINDOW,
        decreasing=True,
        smooth_half_window=SMOOTH_HALF_WINDOW
    )

    # Retour aux angles expérimentaux
    fond = np.interp(
        angle,
        angle,
        fond_calcule
    )

    return fond


# =====================================================
# 5. AFFICHAGE DU DIFRACTOGRAMME
# =====================================================

def afficher_resultats(
    angle,
    intensite,
    fond,
    intensite_corrigee
):

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

# =====================================================
# 8. PICS
# =====================================================

def detection_pics(angle,intensite,prominence_relative=0.0001,distance=4,afficher=True):
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
        plt.show()
    return angle[indices]


# =====================================================
# 9. APPROXIMATION PSEUDO-VOIGT
# =====================================================

for i in range(1,9) :
    brx = np.array(signal_sans_bruit(i))
    premiers_pics = brx[:,(brx[0] <= 43) & (brx[0] >= 41.5)]


    peak1 = lf.models.PseudoVoigtModel(prefix='p1_')
    peak2 = lf.models.PseudoVoigtModel(prefix='p2_')
    peak3 = lf.models.PseudoVoigtModel(prefix='p3_')
    model = peak1 + peak2 + peak3

    params = model.make_params()
    centers = (detection_pics(premiers_pics[0],premiers_pics[1],afficher=False))[:3]
    params['p1_center'].set(value=centers[0])
    params['p2_center'].set(value=centers[1]-0.2)
    params['p3_center'].set(value=centers[2])

    result = model.fit(premiers_pics[1], params, x=premiers_pics[0])

    approximation = result.best_fit

    plt.plot(premiers_pics[0],premiers_pics[1])
    plt.plot(premiers_pics[0],approximation)
    plt.show()
    c1 = result.params['p1_center'].value
    c2 = result.params['p2_center'].value
    c3 = result.params['p3_center'].value
    print(f"intégrale alpha {i} : {result.params['p1_amplitude'].value} \n")
    print(f"intégrale gamma prime {i} : {result.params['p2_amplitude'].value} \n")
    print(f"intégrale gamma {i} : {result.params['p3_amplitude'].value} \n")
    print("\n")

