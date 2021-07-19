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

    # Read inputs
    #################################################
    if len(sys.argv)>1 :
        ifilename = sys.argv[1]
    else :
        print("""
Please add as an argument a filename:
test-petal6-light-red-leg.yaml or test-petal6-heavy-red-leg.yaml
        """)
        sys.exit(12)

    if not os.path.isfile(ifilename) :
        print("cannot find or open",ifilename)
        sys.exit(1)

    print("Input filename:",ifilename)

    with open(ifilename) as ifile :
        inputs=yaml.safe_load(ifile)

    petal = int(inputs["petal"])
    print("Petal:",petal)
    print("BMR: '{}'".format(inputs['bmr_type']))


    if inputs["bmr_type"]== "light_weight_red_leg" :
        target_bmr_CS5_mm = compute_target_bmr_light_weight_red_leg_coords_mm(petal)
    elif inputs["bmr_type"]== "heavy_weight_red_leg" :
        target_bmr_CS5_mm = compute_target_bmr_heavy_weight_red_leg_coords_mm(petal)
    else :
        print('error {} not in ["guide_spikes","light_weight_red_leg","heavy_weight_red_leg"]'.format(inputs["bmr_type"]))
        sys.exit(2)
    target_bmr_CS5_inch = target_bmr_CS5_mm/inch2mm

    # Input BMR Locations (in CS5, will change input method later)
    #################################################

    if inputs["bmr_type"] == "heavy_weight_red_leg" :
        bmr_labels=["B1","B2","B3","B4","B5"]
    else :
        bmr_labels=["B1","B2","B3","B4"]

    number_of_balls=len(bmr_labels)
    measured_bmr_CS5_inch = np.zeros((3,number_of_balls))

    print("Input BMR coordinates (inch):")
    for index,bmr_label in enumerate(bmr_labels) :
        if bmr_label in inputs :
            measured_bmr_CS5_inch[:,index] = str2array(inputs[bmr_label])
            print("{} {}".format(bmr_label,inputs[bmr_label]))

    xyz1=target_bmr_CS5_inch

    # 'measured=solidworks' bmr
    xyz2=measured_bmr_CS5_inch

    # target holes
    xyz3=get_red_leg_mount_holes_in_CS5_mm(petal)/inch2mm

    # 'measured=solidworks' holes
    xyz4=np.zeros((3,3))
    xyz4[:,0]=[-323.19643922,340.10791362,-1759.2820881]
    xyz4[:,1]=[-270.14129446,310.56739657,-1759.27470220]
    xyz4[:,2]=[-219.92149856,364.62873842,-1759.1524087]
    xyz4 /= inch2mm

    if 1 :
        # convert coords to PMA, then apply PMA corr
        # after that the axes should be aligned ... ?
        xyz1 = CS5_to_PMA_inch(xyz1) # target bmr
        xyz2 = CS5_to_PMA_inch(xyz2) # 'measured=solidworks' bmr
        xyz3 = CS5_to_PMA_inch(xyz3) # target holes
        xyz4 = CS5_to_PMA_inch(xyz4) # 'measured=solidworks' holes

    if 1 :
        # apply transfo that has been fit to align
        # the PMA (this aligns the z axes)
        t=Transfo3D.read("transfo-petal6-light-red-leg.yaml")
        print(t)
        xyz1 = t.apply(xyz1)
        xyz2 = t.apply(xyz2)
        xyz3 = t.apply(xyz3)
        xyz4 = t.apply(xyz4)

    # now should be aligned
    # discard z offsets for both pairs
    xyz2[2] += np.mean(xyz1[2]-xyz2[2])
    xyz4[2] += np.mean(xyz3[2]-xyz4[2])

    # HACK
    # xyz2[2] = xyz1[2]
    # xyz4[2] = xyz3[2]

    cs5_invert_mm = Transfo3D()
    rms=cs5_invert_mm.fit(xyz1,xyz2)
    print("Inverse transfo fit rms =",rms,"inch")
    print("Inverse transfo fit mm  =",rms*inch2mm,"mm")
    print(cs5_invert_mm)
    xyz3b = cs5_invert_mm.apply(xyz3)
    delta = xyz4-xyz3b
    print("delta (inch) hole 1 =",delta[:,0])
    print("delta (inch) hole 2 =",delta[:,1])
    print("delta (inch) hole 3 =",delta[:,2])
    print("delta (mm) hole 1 =",delta[:,0]*inch2mm)
    print("delta (mm) hole 2 =",delta[:,1]*inch2mm)
    print("delta (mm) hole 3 =",delta[:,2]*inch2mm)
    max_dist_mm=np.max(np.sqrt(delta[0,:]**2+delta[1,:]**2)*inch2mm)
    print("max dist (mm) =",max_dist_mm)
if __name__ == '__main__':
    main()
