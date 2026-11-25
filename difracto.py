#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from pymicro.crystal.lattice import Lattice, HklPlane
import matplotlib.cm as cm
import matplotlib.colors as colors
import re
from matplotlib.widgets import Slider
from math import radians, degrees, sin, asin
# define our crystal structures using pymicro
from pymicro.crystal.lattice import Lattice, HklPlane


# Palette de couleurs personnalisée
plt.rcParams['axes.prop_cycle'] = cycler(color=["firebrick", "purple", "darkmagenta", "darkviolet", "darkorchid", "indigo","midnightblue", "navy", "darkblue", "mediumblue", "blue", "royalblue","dodgerblue", "blueviolet", "mediumpurple", "orchid", "mediumorchid","mediumvioletred", "crimson", "red", "indianred"])



def LectureCHI_fichier(fichier):
    x = []
    y = []
    with open(fichier, 'r') as f:
        for ligne in f:
            if ligne.strip() and not ligne.startswith('#'):
                valeurs = ligne.strip().split()
                if len(valeurs) >= 2:
                    try:
                        x.append(float(valeurs[0]))
                        y.append(float(valeurs[1]))
                    except ValueError:
                        continue
    return [x, y]

def lire_folder(num):
    X_Plot= LectureCHI_fichier("Plaque_CMSX4_R1_30x10-0000"+str(num)+".dat")
    my_dpi=180
    plt.figure(figsize=(1600/my_dpi, 800/my_dpi), dpi=my_dpi)
    
    plt.plot(X_Plot[0],X_Plot[1])

lire_folder(2)

# Paramètres de longueur d'onde

lambda_kev = 8.047 #cuivre K alpha1 


#Ni gamma
Ni = Lattice.cubic(0.3683)  # unit is nm
miller_indices_Ni = [(1,1,1),(2,0,0),(2,2,0),(3,1,1),(2,2,2),(4,0,0),(3,3,1),(4,2,0),(4,2,2)]

#Ni gamma_prime
Ni_prime = Lattice.cubic(0.3701)
miller_indices_Ni_prime = [(1,0,0),(1,1,0),(1,1,1),(2,0,0),(2,2,0),(3,1,1),(2,2,2),(4,0,0),(3,3,1),(4,2,0),(4,2,2)]

#Ni beta
Ni_beta = Lattice.cubic(0.2891)
miller_indices_Ni_beta = [(1,1,0),(2,0,0),(2,1,1),(2,2,0),(3,1,0),(4,0,0),(3,2,1),(4,1,1),(4,2,0),(2,2,2),(3,3,0)]

#NiPtAl
NiPtAl = Lattice.tetragonal(0.3822, 0.3585)

miller_indices_NiPtAl = [
    (1,0,0), (0,0,1), (1,1,0),
    (1,0,1), (1,1,1), (2,0,0),
    (0,0,2), (2,1,0), (2,0,1),
    (2,1,1), (2,2,0), (2,0,2),
    (3,0,0), (0,0,3), (3,1,0),
    (3,0,1), (3,1,1), (2,2,2)
]


def ListeDist(Cristallo,LHKL):
    """
    renvoie liste [angle de bragg,coordonéees famille plan]
    """
    LDist=[[],[]]
    for h,k,l in LHKL:
        hkl_plane=hkl_plane = HklPlane(h, k, l, lattice=Cristallo)
        Dth = 2*np.degrees(hkl_plane.bragg_angle(lambda_kev)) #2theta = angle de Bragg
        LDist[0].append(Dth)
        LDist[1].append([h,k,l])
    return LDist
        
LGamma=ListeDist(Ni,miller_indices_Ni)
LGamma_prime=ListeDist(Ni_prime,miller_indices_Ni_prime)
LBeta = ListeDist(Ni_beta,miller_indices_Ni_beta)
Lpt = ListeDist(NiPtAl,miller_indices_NiPtAl)


my_dpi=180
    
# plt.xlim([46,49])
plt.ylabel("Intensity (a.u.)")
plt.xlabel("2theta (°)  (corrected in lambda)")
# plt.xlim([22.5,82])


# ##################################################################Plan Interéticulaire 

for i in range(len(LGamma[0])):
    pos = LGamma[0][i]
    hkl = LGamma[1][i]
    label = f"[{hkl[0]}{hkl[1]}{hkl[2]}]"
    # Tracer une ligne verticale fine
    plt.axvline(x=pos, color="green", linestyle="--", linewidth=1)
    # Afficher le texte légèrement au-dessus
    plt.text(pos, 1.02, label, rotation=90,ha='center', va='bottom',color="green")



for i in range(len(LGamma_prime[0])):
    pos = LGamma_prime[0][i]
    hkl = LGamma_prime[1][i]
    label = f"[{hkl[0]}{hkl[1]}{hkl[2]}]"
    # Tracer une ligne verticale fine
    plt.axvline(x=pos, color="red", linestyle="--", linewidth=1)
    # Afficher le texte légèrement au-dessus
    plt.text(pos, 1.02, label, rotation=90,ha='center', va='bottom',color="red")
"""
for i in range(len(LBeta[0])):
    pos = LBeta[0][i]
    hkl = LBeta[1][i]
    label = f"[{hkl[0]}{hkl[1]}{hkl[2]}]"
    # Tracer une ligne verticale fine
    plt.axvline(x=pos, color="blue", linestyle="--", linewidth=1)
    # Afficher le texte légèrement au-dessus
    plt.text(pos, 1.02, label, rotation=90,ha='center', va='bottom',color="blue", transform=ax1.get_xaxis_transform())

for i in range(len(Lpt[0])):
    pos = Lpt[0][i]
    hkl = Lpt[1][i]
    label = f"[{hkl[0]}{hkl[1]}{hkl[2]}]"
    # Tracer une ligne verticale fine
    plt.axvline(x=pos, color="orange", linestyle="--", linewidth=1)
    # Afficher le texte légèrement au-dessus
    plt.text(pos, 1.02, label, rotation=90,ha='center', va='bottom',color="orange", transform=ax1.get_xaxis_transform())
  """ 
################################# Slider ##################################################
# plt.subplots_adjust(bottom=0.25)  # espace pour le slider
# ValeurModifiable = 0.3617#*0.95
# ValeurModifiable2 = 0.5175#*0.95
# PhaseFitable = Lattice.cubic(ValeurModifiable)  # unit is nm
# # PhaseFitable = Lattice.hexagonal(ValeurModifiable,ValeurModifiable2)  # unit is nm
# # PhaseFitable = Lattice.tetragonal(ValeurModifiable,ValeurModifiable2)  # unit is nm
# miller_indices_PhaseFitable = miller_indices_Ni3Al
# LPhaseFitable=ListeDist(PhaseFitable,miller_indices_PhaseFitable)
# bars = [ax1.axvline(x=pos, color="black", linestyle="--", linewidth=1) for pos in LPhaseFitable[0]]
# texts = [ax1.text(pos, 1.02, f"[{hkl[0]}{hkl[1]}{hkl[2]}]", rotation=90, ha='center', va='bottom',color="black", transform=ax1.get_xaxis_transform()) for pos, hkl in zip( LPhaseFitable[0], LPhaseFitable[1])]
# ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
# slider = Slider(ax_slider, 'a₀ (nm)', ValeurModifiable-0.3, ValeurModifiable+0.5, valinit=ValeurModifiable, valstep=0.0001)
# # ax2_slider = plt.axes([0.2, 0.08, 0.6, 0.03])
# # slider2 = Slider(ax2_slider, 'c₀ (nm)',  ValeurModifiable2-0.2, ValeurModifiable2+0.2, valinit=ValeurModifiable2, valstep=0.0001, facecolor='blue'  )
# def update(val):
#     a0 = slider.val
#     # c0 = slider2.val
#     PhaseFitable = Lattice.cubic(a0)
#     # PhaseFitable = Lattice.hexagonal(a0,c0)  # unit is nm
#     # PhaseFitable = Lattice.tetragonal(a0,c0)  # unit is nm
#     LPhaseFitable=ListeDist(PhaseFitable,miller_indices_PhaseFitable)
#     for bar, text, pos in zip(bars, texts, LPhaseFitable[0]):
#         bar.set_xdata([pos])
#         text.set_x(pos)
#     fig.canvas.draw_idle()
# slider.on_changed(update)
# # slider2.on_changed(update)
################################## Slider ##################################################

    
plt.tight_layout()
plt.show()

