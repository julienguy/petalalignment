#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import yaml

from transfo import Transfo3D as Transfo
from PAC import array2str,str2array
from PAC import compute_target_bmr_light_weight_red_leg_coords_mm,get_red_leg_mount_holes_in_CS5_mm

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
    valid_bmr = np.repeat(True,number_of_balls)
    for index,bmr_label in enumerate(bmr_labels) :
        if bmr_label in inputs :
            measured_bmr_CS5_inch[:,index] = str2array(inputs[bmr_label])
            print("{} {}".format(bmr_label,inputs[bmr_label]))

    # align target bmr in z ?
    # target_bmr_CS5_inch[2] = np.mean(target_bmr_CS5_inch[2])

    xyz1=target_bmr_CS5_inch[:,valid_bmr]*inch2mm+0.
    xyz2=measured_bmr_CS5_inch[:,valid_bmr]*inch2mm+0.
    print("target   BMR mm z=",xyz1[2])
    print("measured BMR mm z=",xyz2[2])
    print("measured-target BMR dx (mm)=",np.mean(xyz2[0]-xyz1[0]))
    print("measured-target BMR dy (mm)=",np.mean(xyz2[1]-xyz1[1]))
    xyz2[2] += np.mean(xyz1[2]-xyz2[2])

    # just ignore the offsets in z?
    #xyz2[2] = xyz1[2]

    cs5_invert_mm = Transfo()
    rms=cs5_invert_mm.fit(xyz1,xyz2)
    print("Inverse transfo fit rms=",rms,"mm")
    print(cs5_invert_mm)

    xyz = get_red_leg_mount_holes_in_CS5_mm(petal)
    xyz2 = cs5_invert_mm.apply(xyz)
    for b in range(xyz.shape[1]) :
        print("before fit of bmr, hole",b,"xyz CS5 mm =",array2str(xyz[:,b]))
    for b in range(xyz2.shape[1]) :
        print("after fit of bmr, hole",b,"xyz CS5 mm =",array2str(xyz2[:,b]))

    # the light weight solidworks mount hole xyz coords
    xyz3=np.zeros((3,3))
    xyz3[:,0]=[-323.19643922,340.10791362,-1759.2820881]
    xyz3[:,1]=[-270.14129446,310.56739657,-1759.27470220]
    xyz3[:,2]=[-219.92149856,364.62873842,-1759.1524087]
    for b in range(xyz3.shape[1]) :
        print("solidworks, hole",b,"xyz CS5 mm =",array2str(xyz3[:,b]))
    for b in range(xyz3.shape[1]) :
        print("delta, hole",b,"xyz CS5 mm =",array2str(xyz2[:,b]-xyz3[:,b]))

    # the GS xyz coords
    xyz4=np.zeros((3,3))
    xyz4[:,0]=[-324.24768586,338.65356890,-1759.14749184]
    xyz4[:,1]=[-271.19422020,309.11004225,-1759.12740842]
    xyz4[:,2]=[-220.97156843,363.16829061,-1758.87726430]

    print("dx(light-GS)={:.4f} mm".format(np.mean(xyz3[0]-xyz4[0])))
    print("dy(light-GS)={:.4f} mm".format(np.mean(xyz3[1]-xyz4[1])))
    print("dz(light-GS)={:.4f} mm".format(np.mean(xyz3[2]-xyz4[2])))

    t = Transfo()
    rms=t.fit(xyz,xyz3)
    xyz5=t.apply(xyz)
    print("t rms=",rms)
    print("t dx=",np.mean(xyz5[0]-xyz3[0]))
    print("t dy=",np.mean(xyz5[1]-xyz3[1]))
    plt.figure("quiver")
    plt.quiver(xyz3[0],xyz3[1],xyz5[0]-xyz3[0],xyz5[1]-xyz3[1])
    #plt.quiver(xyz3[0],xyz3[1],xyz2[0]-xyz3[0],xyz2[1]-xyz3[1])


    plt.show()

if __name__ == '__main__':
    main()
