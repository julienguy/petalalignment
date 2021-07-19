#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import yaml

from transfo import Transfo3D,Transfo2D

from PAC import array2str,str2array,\
    compute_target_bmr_light_weight_red_leg_coords_mm,\
    compute_target_bmr_heavy_weight_red_leg_coords_mm,\
    get_red_leg_mount_holes_in_CS5_mm,CS5_to_PMA_inch

inch2mm = 25.4

def main() :

    pma_adjust_gs      = Transfo3D.read("transfo-petal6-gs.yaml")
    pma_adjust_red_leg = Transfo3D.read("transfo-petal6-light-red-leg.yaml")

    # solidworks measured holes for light red leg config in CS5
    xyz4=np.zeros((3,3))
    xyz4[:,0]=[-323.19643922,340.10791362,-1759.2820881]
    xyz4[:,1]=[-270.14129446,310.56739657,-1759.27470220]
    xyz4[:,2]=[-219.92149856,364.62873842,-1759.1524087]
    xyz4 /= inch2mm

    # solidworks measured holes for guide spikes config in CS5
    xyz5=np.zeros((3,3))
    xyz5[:,0]=[-324.247,338.653,-1759.147]
    xyz5[:,1]=[-271.194,309.110,-1759.127]
    xyz5[:,2]=[-220.971,363.168,-1758.877]
    xyz5 /= inch2mm

    if 1 :
        # convert coords to PMA, then apply PMA corr
        xyz4 = CS5_to_PMA_inch(xyz4)
        xyz5 = CS5_to_PMA_inch(xyz5)
    if 1 :
        # don't apply the rotation along x and y
        # because too noisy
        pma_adjust_red_leg.ax=0.
        pma_adjust_red_leg.ay=0.
        pma_adjust_gs.ax=0.
        pma_adjust_gs.ay=0.
    if 1 :
        xyz4 = pma_adjust_red_leg.apply(xyz4)
        xyz5 = pma_adjust_gs.apply(xyz5)

    delta = xyz5-xyz4
    delta_mm = delta*inch2mm
    print("delta (PMA, mm):")
    print(delta_mm)





if __name__ == '__main__':
    main()
