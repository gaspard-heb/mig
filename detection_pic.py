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

# Longueurs d'onde Cu Kalpha1 et Kalpha2
LAMBDA_ANGSTROM = 1.54056
LAMBDA_KALPHA2 = 1.54439

# Rapport des intensites Kalpha2/Kalpha1
RATIO_KALPHA = 0.5

# Energie correspondante en keV
ENERGIE_KEV = 8.047

# Parametres de maille (nm)
aGamma = 0.3683
aGammaPrime = 0.3701
aAlpha = 0.3821
cAlpha = 0.3591

# Parametres d'ajustement des pics
SigmaIni = 0.05
SigmaMin = 0.005
SigmaMax = 0.3
FractionIni = 0.5
DeltaCenter = 0.3
FenetreFit = 0.7
MaxEvaluations = 600


# =====================================================
# POSITIONS INITIALES DES PICS KALPHA1 ET INDICES HKL
# =====================================================

# Gamma - Ni - Cubique
LGamma = [
    [42.477,49.453,72.534,87.840,92.854,113.561,131.466,138.561,None],
    [(1,1,1),(2,0,0),(2,2,0),(3,1,1),(2,2,2),(4,0,0),(3,3,1),(4,2,0),(4,2,2)]
]

# Gamma prime - Ni3Al - Cubique
LGammaPrime = [
    [24.025,34.236,42.260,49.197,55.471,61.302,72.126,77.274,77.274,82.319,87.305,92.270,97.253,102.291,112.715,124.015,130.244,137.112,None],
    [(1,0,0),(1,1,0),(1,1,1),(2,0,0),(2,1,0),(2,1,1),(2,2,0),(2,2,1),(3,0,0),(3,1,0),(3,1,1),(2,2,2),(3,2,0),(3,2,1),(4,0,0),(3,3,0),(3,3,1),(4,2,0),(4,2,2)]
]

# Alpha - NiPt - Tetragonale
LAlpha = [
    [24.773,33.129,41.805,47.555,50.809,54.348,62.008,69.526,72.134,75.063,79.209,80.108,84.537,89.471,91.050,98.819,100.421,107.485,113.108,117.581,118.188,118.583],
    [(0,0,1),(1,1,0),(1,1,1),(2,0,0),(0,0,2),(2,0,1),(1,1,2),(2,2,0),(2,0,2),(2,2,1),(3,1,0),(0,0,3),(3,1,1),(1,1,3),(2,2,2),(2,0,3),(3,1,2),(4,0,0),(4,0,1),(3,3,0),(0,0,4),(2,2,3)]
]


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
                    xi = float(valeurs[0])
                    yi = float(valeurs[1])
                    x.append(xi)
                    y.append(yi)
                except ValueError:
                    continue
    if len(x) == 0:
        raise ValueError("Aucune donnÃ©e valide dans le fichier.")
    return np.array(x), np.array(y)


def preparer_donnees(angle, intensite):
    # gere les potentielles donÃ©es incohÃ©rentes

    # Suppression des valeurs non finies
    masque = np.isfinite(angle) & np.isfinite(intensite)
    angle = angle[masque]
    intensite = intensite[masque]

    # Tri selon les angles
    ordre = np.argsort(angle)
    angle = angle[ordre]
    intensite = intensite[ordre]

    # Regroupement des angles dupliquÃ©s
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

    # SNIP travaille avec une fenÃªtre exprimÃ©e
    # en nombre de points.
    # On vÃ©rifie donc l'espacement angulaire.

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

    # Retour aux angles expÃ©rimentaux
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

    # Premier graphique : fond estimÃ©

    axes[0].plot(
        angle,
        intensite,
        color='black',
        linewidth=0.8,
        label='Diffractogramme expÃ©rimental'
    )

    axes[0].plot(
        angle,
        fond,
        color='red',
        linewidth=2,
        label='Fond estimÃ© (SNIP)'
    )

    axes[0].set_ylabel('IntensitÃ© (coups)')
    axes[0].set_title(
        'Estimation automatique du bruit de fond DRX'
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # DeuxiÃ¨me graphique : signal corrigÃ©

    axes[1].plot(
        angle,
        intensite_corrigee,
        color='blue',
        linewidth=0.8,
        label='Signal corrigÃ©'
    )

    axes[1].axhline(
        0,
        color='red',
        linestyle='--',
        linewidth=0.8
    )

    axes[1].set_xlabel(r'Angle $2\theta$ (degrÃ©s)')
    axes[1].set_ylabel('IntensitÃ© corrigÃ©e')
    axes[1].set_title(
        'Diffractogramme aprÃ¨s soustraction du fond'
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

    # PrÃ©paration
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
    print("\nTraitement terminÃ© avec succÃ¨s.")


def signal_sans_bruit(numero_fichier):
    # Lecture du fichier
    angle, intensite = LectureCHI_fichier(
        numero_fichier
    )

    # PrÃ©paration des donnÃ©es
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


# Calcul de la position theorique d'une reflexion
# a et c en nm, longueur d'onde en Angstrom
def position_theorique(hkl,a,c):
    h,k,l = hkl
    d = 10/np.sqrt((h**2+k**2)/a**2+l**2/c**2)
    argument = LAMBDA_ANGSTROM/(2*d)
    if argument>1:
        return None
    return 360/np.pi*np.arcsin(argument)


# Calcul de la position Kalpha2 connaissant Kalpha1
def position_Kalpha2(center):
    argument = LAMBDA_KALPHA2/LAMBDA_ANGSTROM*np.sin(center*np.pi/360)
    if argument>1:
        return None
    return 360/np.pi*np.arcsin(argument)


# Construction de la liste des reflexions a ajuster
def liste_reflexions(angle):
    reflexions = []
    for L,phase,couleur,label in [
        (LAlpha,'alpha','blue',r'$\alpha$'),
        (LGammaPrime,'gamma_prime','green',r"$\gamma'$"),
        (LGamma,'gamma','goldenrod',r'$\gamma$')
    ]:
        if len(L[0])!=len(L[1]):
            raise ValueError(f"Listes de tailles differentes : {phase}")
        for center,hkl in zip(L[0],L[1]):
            # Exclusion des reflexions impossibles ou hors du diffractogramme
            if center is None:
                continue
            if center<angle[0] or center>angle[-1]:
                continue
            if position_Kalpha2(center) is None:
                continue

            # Regroupement des reflexions de meme phase ayant le meme angle
            existe = False
            for r in reflexions:
                if r[2]==phase and abs(r[0]-center)<1e-6:
                    r[1].append(hkl)
                    existe = True
                    break
            if not existe:
                reflexions.append([center,[hkl],phase,couleur,label])

    return sorted(reflexions,key=lambda r:r[0])


# Regroupement des reflexions voisines
def groupes_reflexions(reflexions):
    groupes = []
    for r in reflexions:
        center = r[0]
        debut = center-FenetreFit
        fin = position_Kalpha2(center)+FenetreFit

        # Conservation de la fenetre initiale pour les pics 111
        if (1,1,1) in r[1]:
            debut = min(debut,41)
            fin = max(fin,43.5)

        if len(groupes)==0 or debut>groupes[-1][1]:
            groupes.append([debut,fin,[r]])
        else:
            groupes[-1][0] = min(groupes[-1][0],debut)
            groupes[-1][1] = max(groupes[-1][1],fin)
            groupes[-1][2].append(r)

    return groupes


# =====================================================
# 9. APPROXIMATION PSEUDO-VOIGT
# =====================================================

def fit_groupe(angle,intensite,reflexions):

    # Construction du modele
    model = None
    for j in range(len(reflexions)):
        p = f'p{j+1}_'
        peak_Ka1 = lf.models.PseudoVoigtModel(prefix=p)
        peak_Ka2 = lf.models.PseudoVoigtModel(prefix=p+'Ka2_')
        doublet = peak_Ka1+peak_Ka2
        if model is None:
            model = doublet
        else:
            model = model+doublet

    params = model.make_params()

    # Limite de validite de la loi de Bragg pour Kalpha2
    CenterMaxBragg = 360/np.pi*np.arcsin(LAMBDA_ANGSTROM/LAMBDA_KALPHA2)-1e-6

    for j,r in enumerate(reflexions):
        p = f'p{j+1}_'
        center,hkls,phase,couleur,label = r

        # Initialisation et bornes des positions Kalpha1
        center_min = max(center-DeltaCenter,angle[0])
        center_max = min(center+DeltaCenter,angle[-1],CenterMaxBragg)

        # Conservation des bornes du fit initial pour les pics 111
        if (1,1,1) in hkls:
            if phase=='alpha':
                center_min,center_max = 41.5,42.05
            elif phase=='gamma_prime':
                center_min,center_max = 41.95,42.45
            elif phase=='gamma':
                center_min,center_max = 42.2,42.85
            center_min = max(center_min,angle[0])
            center_max = min(center_max,angle[-1])

        params[p+'center'].set(value=np.clip(center,center_min,center_max),
                               min=center_min,max=center_max)

        # Initialisation des amplitudes et des largeurs Kalpha1
        height = np.interp(center,angle,intensite)
        amplitude = max(height,1)*2.5*SigmaIni
        params[p+'amplitude'].set(value=amplitude,min=0)
        params[p+'sigma'].set(value=SigmaIni,min=SigmaMin,max=SigmaMax)
        params[p+'fraction'].set(value=FractionIni,min=0,max=1)

        # Contraintes physiques entre Kalpha1 et Kalpha2
        params[p+'Ka2_center'].set(expr=f'2*180/pi*arcsin({LAMBDA_KALPHA2/LAMBDA_ANGSTROM}*sin({p}center*pi/360))')
        params[p+'Ka2_amplitude'].set(expr=f'{RATIO_KALPHA}*{p}amplitude')
        params[p+'Ka2_sigma'].set(expr=f'{p}sigma')
        params[p+'Ka2_fraction'].set(expr=f'{p}fraction')

    # Ajustement par Levenberg-Marquardt
    result = model.fit(intensite,params,x=angle,method='leastsq',
                       max_nfev=MaxEvaluations,calc_covar=False)

    return result


# Intensites integrees des reflexions 111
gamma, gamma_prime, alpha = np.zeros(8),np.zeros(8),np.zeros(8)

# Premiere figure : plaques 1 a 4
fig1, axes1 = plt.subplots(2,2,figsize=(15,9),sharex=True,sharey=True)
axes1 = axes1.flatten()

# Deuxieme figure : plaques 5 a 8
fig2, axes2 = plt.subplots(2,2,figsize=(15,9),sharex=True,sharey=True)
axes2 = axes2.flatten()

axes = np.concatenate((axes1,axes2))

# Stockage des resultats de chaque plaque
Resultats = []

for i in range(1,9):

    angle,intensite = signal_sans_bruit(i)

    # Liste des reflexions attendues dans le diffractogramme
    reflexions = liste_reflexions(angle)
    groupes = groupes_reflexions(reflexions)

    print(f"\n========== Plaque {i} ==========")
    print(f"Nombre de reflexions : {len(reflexions)}")
    print(f"Nombre de groupes : {len(groupes)}")

    approximation = np.zeros_like(intensite)
    contributions = []
    resultats_plaque = []

    for debut,fin,reflexions_groupe in groupes:

        # Selection de la region angulaire a ajuster
        masque = (angle>=debut)&(angle<=fin)
        angle_local = angle[masque]
        intensite_locale = intensite[masque]

        if len(angle_local)<=4*len(reflexions_groupe):
            print(f"Groupe {debut:.2f}-{fin:.2f} : nombre de points insuffisant")
            continue

        print(f"Fit {debut:.2f}-{fin:.2f} deg : {len(reflexions_groupe)} reflexions",flush=True)

        # Ajustement local
        result = fit_groupe(angle_local,intensite_locale,reflexions_groupe)

        print(f"Convergence : {result.success}, evaluations : {result.nfev}")

        # Evaluation des contributions sur tout le diffractogramme
        components = result.eval_components(x=angle)

        for j,r in enumerate(reflexions_groupe):
            p = f'p{j+1}_'
            center,hkls,phase,couleur,label = r

            contribution_Ka1 = components[p]
            contribution_Ka2 = components[p+'Ka2_']

            approximation += contribution_Ka1+contribution_Ka2

            # Stockage des contributions individuelles
            contributions.append([contribution_Ka1,contribution_Ka2,couleur,label])

            # Parametres ajustes
            center_fit = result.params[p+'center'].value
            amplitude_fit = result.params[p+'amplitude'].value+result.params[p+'Ka2_amplitude'].value
            sigma_fit = result.params[p+'sigma'].value
            fraction_fit = result.params[p+'fraction'].value

            resultats_plaque.append([phase,hkls,center_fit,amplitude_fit,sigma_fit,fraction_fit])

            # Intensites integrees des reflexions 111
            if (1,1,1) in hkls:
                if phase=='alpha':
                    alpha[i-1] = amplitude_fit
                elif phase=='gamma_prime':
                    gamma_prime[i-1] = amplitude_fit
                elif phase=='gamma':
                    gamma[i-1] = amplitude_fit

    Resultats.append(resultats_plaque)

    # Affichage du diffractogramme et du fit total
    ax = axes[i-1]
    ax.plot(angle,intensite,color='black',linewidth=1,label='Exp.')
    ax.plot(angle,approximation,color='red',linewidth=1,label='Fit')

    # Affichage des contributions individuelles
    labels_affiches = []

    for contribution_Ka1,contribution_Ka2,couleur,label in contributions:
        if label not in labels_affiches:
            ax.plot(angle,contribution_Ka1,color=couleur,linewidth=0.8,label=label)
            labels_affiches.append(label)
        else:
            ax.plot(angle,contribution_Ka1,color=couleur,linewidth=0.8)
        ax.plot(angle,contribution_Ka2,color=couleur,linewidth=0.8,linestyle='--')

    ax.set_title(f'Plaque {i}')
    ax.set_xlabel(r'$2\theta$ (Â°)')
    ax.set_ylabel('IntensitÃ© (coups)')
    ax.set_xlim(angle[0],angle[-1])
    ax.legend()

fig1.tight_layout()
fig2.tight_layout()


# =====================================================
# 10. INTENSITES RELATIVES DES REFLEXIONS 111
# =====================================================

categories = np.array([f"{i}" for i in range(1,9)])

total = alpha+gamma+gamma_prime
p_alpha = np.divide(alpha,total,out=np.full_like(alpha,np.nan),where=total>0)
p_gamma_prime = np.divide(gamma_prime,total,out=np.full_like(gamma_prime,np.nan),where=total>0)
p_gamma = np.divide(gamma,total,out=np.full_like(gamma,np.nan),where=total>0)

plt.show()