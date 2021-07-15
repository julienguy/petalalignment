#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from PAC import compute_target_bmr_guide_spikes_coords_mm,Transfo,array2str

def compute_target_bmr_guide_spikes_coords_mm_bis(petal) :
    """
    Computes the target bmr guide spikes coordinates in CS5
    Arg:
       petal number (int)


    Returns:
       2D np.array with coordinates in mm
       1st axis is axis index (x,y,z)
       2nd axis is ball number

    Same as compute_target_bmr_guide_spikes_coords_mm but using the metrology results
    from DESI-4949 (finding offset of 0.1 mm between the two)
    """

    metrology=np.array([[375.2493,208.1763,182.5254,444.5140,188.7656,182.1765,520.1267,71.4734,181.8783,425.8791,51.5346,182.1460],
                        [375.5460,208.4098,182.1637,444.8191,189.0376,181.7222,520.4942,71.7869,181.4122,426.2611,51.7953,181.8488],
                        [375.5610,208.3894,183.0745,444.8321,189.0095,182.6271,520.4954,71.7522,182.3027,426.2603,51.7688,182.7417],
                        [375.5708,208.4177,182.3806,444.8428,189.0370,181.9354,520.5027,71.7802,181.6267,426.2679,51.7986,182.0683]])

    # data rows: Petal02R,Run1  Petal03R,Run1  Petal03R,Run2  Petal03R,Run3
    # Average of 3 metrology runs on Petal03R
    # Van: "As for the coordinate deviations from the metrology data: I've reviewed my records and it appears that we excluded the metrology run on petal 02R from the averages. 02R had some significant problems which warranted its exclusion. The short story is that we chose to use the averages from the runs on petal 03R, and I'd be inclined to stick with that decision in the absence of a new reason to change it."

    xyz = np.mean(metrology[1:],axis=0).reshape((4,3)).T

    angle = 2*np.pi/10.*(3-petal)
    ca    = np.cos(angle)
    sa    = np.sin(angle)
    rot = np.array([[ca,sa,0],[-sa,ca,0],[0,0,1]])
    xyz = rot.dot(xyz)

    # From email from Van (2021/07/15)
    # Plane A does not have a fixed Z coordinate in CS5; it changes as the petal translates along the insertion axis. It's Z location in CS5 will vary from petal to petal due to shimming, however it is nominally -108 millimeters minus the thickness of the shims for that petal (the shims increase the magnitude of the negative displacement).

    z_CS5_plane_A = -108. # mm

    xyz[2] += z_CS5_plane_A

    # coordinates
    plt.figure("CS5")
    plt.plot(xyz[0],xyz[1],"x",color="C1",alpha=0.5)
    for b in range(4) :
        plt.text(xyz[0,b]+5,xyz[1,b]+5,bmr_labels[b],color="C0")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()


    return xyz



inch2mm = 25.4

petal = 0
bmr_labels=["B1","B2","B3","B4"]
target_bmr_CS5_mm       = compute_target_bmr_guide_spikes_coords_mm(petal)
other_target_bmr_CS5_mm = compute_target_bmr_guide_spikes_coords_mm_bis(petal)

# align z for now
other_target_bmr_CS5_mm[2] += np.mean(target_bmr_CS5_mm[2]-other_target_bmr_CS5_mm[2])
# other_target_bmr_CS5_mm[2] = target_bmr_CS5_mm[2] # same z

corr=Transfo()
corr.fit(other_target_bmr_CS5_mm,target_bmr_CS5_mm)
ca = np.sum(corr.targetvec*corr.startvec)/np.sqrt(np.sum(corr.targetvec**2)*np.sum(corr.startvec**2))
angle_deg = np.arccos(ca)*180/np.pi
print("angle = {:.6f} deg".format(angle_deg))
print("petal =",petal)
delta = (other_target_bmr_CS5_mm-target_bmr_CS5_mm)
print("delta (DESI-4949 - current) [dx,dy,dz] in CS5 (mm)")
print("----------------------------------------------------")
for b in range(4) :
    print("   Ball {} = {}".format(bmr_labels[b],array2str(delta[:,b])))
print("    Mean   = ",array2str(np.mean(delta,axis=1)))

plt.show()
