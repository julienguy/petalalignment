#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import yaml

from transfo import Transfo3D as Transfo
from transfo import Transfo3D,Transfo2D,Rot3D,Rot2D
#from transfo import Transfo2D as Transfo

inch2mm = 25.4
######################################################################
plot  = True
debug = False
######################################################################

def CS5_to_PMA_inch(xyz,carriage_z) :
    """
    Transform coordinates from CS5 to PMA
    Input :
     xyz np.array 1D or 2D, first dimension is 3 (for x,y,z), coordinates in inches
    Output :
     array of same shape as xyz, in inches

    """
    # Angle between CS5 and PMA CS
    CS5toPMA_Clocking = -134.75 * np.pi / 180

    # Arrays to convert CS5 coords to PMA CS coords
    CS5toPMA = np.array([[np.cos(CS5toPMA_Clocking), -np.sin(CS5toPMA_Clocking), 0],
                         [np.sin(CS5toPMA_Clocking), np.cos(CS5toPMA_Clocking), 0],
                         [0, 0, 1]])

    CS5toPMA_translate = np.array([0,0,-carriage_z])

    res = np.dot(CS5toPMA, xyz)
    if len(res.shape)==1 :
        return res + CS5toPMA_translate
    else :
        return res + CS5toPMA_translate[:,None]

def GS1_to_Petal_inch(xyz) :
    """
    Transform coordinates from GS1 (Guide Spike 1) to Petal.
    It's a pure translation
    """

    # coordinate of GS1 in petal CS
    GS1_Petal = np.array([-429 * np.sin(np.radians(9)),
                              -429 * np.cos(np.radians(9)),
                              -57.4351 * inch2mm]) / inch2mm # inches
    if len(xyz.shape)==1 :
        return xyz + GS1_Petal
    else :
        return xyz + GS1_Petal[:,None]

def Petal_to_CS5(xyz,petal) :
    """
    Transform coordinates from Petal to CS5
    It's a pure rotation
    """
    # Calculate the target locations based on the nominal petal position in the FPA
    petalTransform = np.array([[np.cos(np.radians(petal * 36)), -np.sin(np.radians(petal * 36)), 0],
                                       [np.sin(np.radians(petal * 36)), np.cos(np.radians(petal * 36)), 0],
                                       [0, 0, 1]])
    return petalTransform.dot(xyz)

def parsearray(inputtext):
    '''
    Takes an input string, parses it as a tab delimited sheet of floats, and
    outputs it to a Numpy array.
    '''

    last = 0
    i = 0
    x = 0
    c = 1
    r = 1
    length = len(inputtext)
    out = np.array([])

    while (i != -1 or x != -1):
        i = inputtext.find('\t',last)
        x = inputtext.find('\n',last)

        notendoftext = last != length

        if (x < i or i == -1) and x != -1 and notendoftext:
            out = np.append(out, float(inputtext[last:x]))
            last = x + 1
            if x < length - 1:
                r = r + 1
        elif notendoftext:
            out = np.append(out, float(inputtext[last:i]))
            last = i + 1
            if r < 2:
                c = c + 1

    out = np.reshape(out,[r,c])

    return out

def pastearray():
    '''
    Grabs the contents of the clipboard and trys to convert it to a Numpy array
    of floats. Only works in Windows. Only use with numbers.
    '''
    try :
        import win32clipboard
        win32clipboard.OpenClipboard()
        out = parsearray(win32clipboard.GetClipboardData())
        win32clipboard.CloseClipboard()
    except ModuleNotFoundError as e :
        print("no windows, use shell prompt")
        user_input = input2("input:")
        out = parsearray(user_input)

    print("read from input =",out)

    return out

def array2str(x) :
    line="["
    for i,ix in enumerate(x) :
        if i>0 :
            line += ","
        line += " {:.3f}".format(ix)
    line += " ]"
    return line


######################################################################

def get_target_bmr_guide_spikes_in_GS1_inch() :
    # originally from Van's python script
    # BMR (Ball Mount Refectors) Target Locations
    # when mounted on petal (for petal insertion)
    # in Guide Spike 1 CS, from metrology:
    target_guide_spikes_bmr_GS1CS_inch = np.zeros((3,4))
    target_guide_spikes_bmr_GS1CS_inch[:,0]=[5.8764, 0.0842, 1.5300]
    target_guide_spikes_bmr_GS1CS_inch[:,1]=[4.3081, -2.2738, 1.5125]
    target_guide_spikes_bmr_GS1CS_inch[:,2]=[-1.0028, -3.6804, 1.5001]
    target_guide_spikes_bmr_GS1CS_inch[:,3]=[-0.6047, 0.0911, 1.5174]
    return target_guide_spikes_bmr_GS1CS_inch

def compute_target_bmr_guide_spikes_coords_mm(petal,plot=False) :
    """
    Computes the target bmr guide spikes coordinates in CS5
    Arg:
       petal number (int)


    Returns:
       2D np.array with coordinates in mm
       1st axis is axis index (x,y,z)
       2nd axis is ball number
    """
    bmr_GS1CS_inch = get_target_bmr_guide_spikes_in_GS1_inch()

    ## transform
    bmr_Petal_inch = GS1_to_Petal_inch(bmr_GS1CS_inch) # rotation and translation
    bmr_CS5_inch   = Petal_to_CS5(bmr_Petal_inch,petal) # pure rotation

    ## convert inch to mm
    bmr_CS5_mm = bmr_CS5_inch*inch2mm

    if plot :
        import matplotlib.pyplot as plt
        plt.figure("CS5")
        plt.plot(bmr_CS5_mm[0],bmr_CS5_mm[1],"o",markersize=12,label="bmr")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid()
        plt.xlabel("X_CS5 (mm)")
        plt.ylabel("Y_CS5 (mm)")

        if 0 :
            plt.figure("GS1")
            bmr_GS1CS_mm = bmr_GS1CS_inch * inch2mm
            plt.plot(bmr_GS1CS_mm[0],bmr_GS1CS_mm[1],"o",markersize=12,label="bmr")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid()
            plt.xlabel("X_GS1 (mm)")
            plt.ylabel("Y_GS1 (mm)")

    # For agreement with orginal code from Van:
    # z_pma = 59.935 inch = 1522.349 mm, tuned to get
    # carriage_z = -115.8456 inch
    # needed to get the correct relative position
    # of the bmr and the upper struts.
    # this is about 10 inches in front of the red leg
    # bmr which is coherent.

    # More recently Pat measured on the SolidWorks model that
    # "the average Z value for the four balls is 695.7427 mm closer
    # to the telescope than the base of PMA strut (#6)"
    # "This is the PMA vertical strut that is closest to the telescope
    # and on the left when facing the telescope"
    # It is 'US6' with z_pma = 35.397 inch
    # So the BMR z_pma = 35.397*25.4 + 695.7427 = 1594.8265 mm = 62.788 inch

    z_pma_mm = 1594.8265

    return bmr_CS5_mm , z_pma_mm

######################################################################

def get_red_leg_mount_holes_in_CS5_mm(petal) :
    """
    red_leg_mount_holes_in_CS5_mm
    Arg:
       petal number (int)


    Returns:
       2D np.array with coordinates in mm
       1st axis is axis index (x,y,z)
       2nd axis is hole number (1,2,3)
    """
    # from CS5LegHoles.xlsx
    # row: hole1,hole2,hole3
    # column: petal 0,1,2,3,4,5,6,7,8,9
    holes_coords_CS5_mm = np.zeros((10,3*3))
    holes_coords_CS5_mm = np.array([[59.9195,-465.5913,-265.4000,34.4005,-410.5622,-265.4000,-37.9811,-424.7652,-265.4000],
                                 [322.1436,-341.4515,-265.4000,269.1530,-311.9317,-265.4000,218.9434,-365.9670,-265.4000],
                                 [461.3198,-86.8888,-265.4000,401.0982,-94.1539,-265.4000,392.2389,-167.3818,-265.4000],
                                 [424.2875,200.8625,-265.4000,379.8375,159.5875,-265.4000,415.7125,95.1375,-265.4000],
                                 [225.1918,411.8911,-265.4000,213.4918,352.3719,-265.4000,280.3981,321.3175,-265.4000],
                                 [-59.9195,465.5913,-265.4000,-34.4005,410.5622,-265.4000,37.9811,424.7652,-265.4000],
                                 [-322.1436,341.4515,-265.4000,-269.1530,311.9317,-265.4000,-218.9434,365.9670,-265.4000],
                                 [-461.3198,86.8888,-265.4000,-401.0982,94.1539,-265.4000,-392.2389,167.3818,-265.4000],
                                 [-424.2875,-200.8625,-265.4000,-379.8375,-159.5875,-265.4000,-415.7125,-95.1375,-265.4000],
                                 [-225.1918,-411.8911,-265.4000,-213.4918,-352.3719,-265.4000,-280.3981,-321.3175,-265.4000]])


    if plot :
        import matplotlib.pyplot as plt
        plt.figure("CS5")
        for p in range(10) :
            holes = holes_coords_CS5_mm[p].reshape(3,3).T

            holes[2] -= (55.-2.49)*inch2mm # offset in z_CS5 HACK HERE

            if p == petal :
                alpha = 1
            else :
                continue
                alpha = 0.1
            plt.plot(holes[0,0],holes[1,0],"o",color="b",alpha=alpha)
            plt.plot(holes[0,1],holes[1,1],"o",color="g",alpha=alpha)
            plt.plot(holes[0,2],holes[1,2],"o",color="r",alpha=alpha)
            plt.gca().set_aspect('equal', adjustable='box')

    xyz = holes_coords_CS5_mm[petal].reshape(3,3).T
    if debug :
        for p in range(xyz.shape[1]) :
            print("DEBUG Red Leg Hole {} xyz (mm) = {}".format(p+1,xyz[:,p]))

    return xyz


def get_red_leg_mount_holes_in_6206_mm() :
    # from DESI-6206
    # with permutation of holes to get same references as in Pat's spreadsheet CS5LegHoles.xlsx
    red_leg_holes_6206_mm = np.array([[26.7884204,-3.6420895,0],[-14.5357283,40.8529444,0],[-79.0081898,4.964665,0]]).T
    if False and plot :
        import matplotlib.pyplot as plt
        plt.figure("red_leg_mount_holes_6206")
        plt.plot(red_leg_holes_6206_mm[0,0],red_leg_holes_6206_mm[1,0],"o",color="b",label="hole1")
        plt.plot(red_leg_holes_6206_mm[0,1],red_leg_holes_6206_mm[1,1],"o",color="g",label="hole2")
        plt.plot(red_leg_holes_6206_mm[0,2],red_leg_holes_6206_mm[1,2],"o",color="r",label="hole3")
        plt.gca().set_aspect('equal', adjustable='box')

    return red_leg_holes_6206_mm

def compute_target_bmr_light_weight_red_leg_coords_mm(petal,plot=False) :

    red_leg_mount_holes_CS5_mm  = get_red_leg_mount_holes_in_CS5_mm(petal)
    red_leg_mount_holes_6206_mm = get_red_leg_mount_holes_in_6206_mm()
    # set same z
    # red_leg_mount_holes_6206_mm[2] += np.mean(red_leg_mount_holes_CS5_mm[2]-red_leg_mount_holes_6206_mm[2])
    C6206_to_CS5 = Transfo()
    rms = C6206_to_CS5.fit(red_leg_mount_holes_6206_mm,red_leg_mount_holes_CS5_mm)
    if rms > 1. : # mm
        print("ERROR rms(6206->CS5) ={:.3f} mmm".format(rms))
    elif debug :
        print("DEBUG rms(6206->CS5) ={:.3f} mmm".format(rms))

    # BMR (Ball Mount Refectors) Locations
    # from DESI-6207 'leg Laser Target Mount metrology'
    # Use Sphere label 1 2 3 4 as described in DESI-6207
    # Use column 'actual' (instead of 'nominal')
    # each entry in x,y,z
    # pin labeled 'A' has coordinates x=0,y=0
    # pin labeled 'B' has coordinates x=-50.021,y=0
    # z=0 is defined by surface of plate opposite side of the balls
    # dimensions in mm
    bmr_6207_mm = np.zeros((3,4))
    bmr_6207_mm[:,0]=[-46.3274581,59.1762362,-25.8866415]
    bmr_6207_mm[:,1]=[88.606006,23.190805,-25.8669889]
    bmr_6207_mm[:,2]=[128.6572029,-36.8625971,-25.8868659]
    bmr_6207_mm[:,3]=[-36.3462882,-36.7938889,-25.8457813]

    xyz_6206 = np.array([ [0.,0.,0.] , [-50.,0.,0.] ]).T
    xyz_6207 = np.array([ [0.,0.,0.] , [50.,0.,0.] ]).T
    C6207_to_C6206 = Transfo()
    rms=C6207_to_C6206.fit(xyz_6207,xyz_6206)
    if debug : print("DEBUG rms(6207->6206) ={:.3f} mmm".format(rms))

    bmr_6206_mm = C6207_to_C6206.apply(bmr_6207_mm)
    bmr_CS5_mm  = C6206_to_CS5.apply(bmr_6206_mm)

    if plot :
        import matplotlib.pyplot as plt
        plt.figure("CS5")
        plt.plot(red_leg_mount_holes_CS5_mm[0,0],red_leg_mount_holes_CS5_mm[1,0],"o",color="b",label="hole1")
        plt.plot(red_leg_mount_holes_CS5_mm[0,1],red_leg_mount_holes_CS5_mm[1,1],"o",color="g",label="hole2")
        plt.plot(red_leg_mount_holes_CS5_mm[0,2],red_leg_mount_holes_CS5_mm[1,2],"o",color="r",label="hole3")
        plt.plot(bmr_CS5_mm[0],bmr_CS5_mm[1],"+",markersize=12,label="bmr")
        #for b in range(bmr_CS5_mm.shape[1]) :
        #    plt.text(bmr_CS5_mm[0,b],bmr_CS5_mm[1,b],str(b+1))
        xyz_CS5 = C6206_to_CS5.apply(xyz_6206)
        plt.plot(xyz_CS5[0,0],xyz_CS5[1,0],"X",color="gray",label="A")
        plt.plot(xyz_CS5[0,1],xyz_CS5[1,1],".",color="gray")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid()
        plt.xlabel("X_CS5 (mm)")
        plt.ylabel("Y_CS5 (mm)")

        if 0 :
            plt.figure("C6206")
            plt.plot(red_leg_mount_holes_6206_mm[0,0],red_leg_mount_holes_6206_mm[1,0],"o",color="b",label="hole1")
            plt.plot(red_leg_mount_holes_6206_mm[0,1],red_leg_mount_holes_6206_mm[1,1],"o",color="g",label="hole2")
            plt.plot(red_leg_mount_holes_6206_mm[0,2],red_leg_mount_holes_6206_mm[1,2],"o",color="r",label="hole3")

            plt.plot(bmr_6206_mm[0],bmr_6206_mm[1],"o",markersize=12,label="bmr",color="C0")
            for b in range(bmr_6206_mm.shape[1]) :
                plt.text(bmr_6206_mm[0,b],bmr_6206_mm[1,b],str(b+1),color="C0")
            plt.plot(xyz_6206[0,0],xyz_6206[1,0],"X",color="gray",label="A")
            plt.plot(xyz_6206[0,1],xyz_6206[1,1],".",color="gray")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid()
            plt.xlabel("X_DESI-6206 (mm)")
            plt.ylabel("Y_DESI-6206 (mm)")

    z_pma_inch = 35.397+15.04 # z of strut + z of difference between strut and ball
    z_pma_mm   = z_pma_inch*inch2mm
    return bmr_CS5_mm , z_pma_mm

def compute_target_bmr_heavy_weight_red_leg_coords_mm(petal,plot=False) :

    red_leg_mount_holes_CS5_mm  = get_red_leg_mount_holes_in_CS5_mm(petal)
    red_leg_mount_holes_6206_mm = get_red_leg_mount_holes_in_6206_mm()
    #red_leg_mount_holes_6206_mm[2] += np.mean(red_leg_mount_holes_CS5_mm[2]-red_leg_mount_holes_6206_mm[2])
    C6206_to_CS5 = Transfo()
    rms =  C6206_to_CS5.fit(red_leg_mount_holes_6206_mm,red_leg_mount_holes_CS5_mm)
    rms = C6206_to_CS5.fit(red_leg_mount_holes_6206_mm,red_leg_mount_holes_CS5_mm)
    if rms > 1. : # mm
        print("ERROR rms(6206->CS5) = {:.3f} mmm".format(rms))
    elif debug :
        print("DEBUG rms(6206->CS5) = {:.3f} mmm".format(rms))

    # BMR (Ball Mount Refectors) Locations
    # from DESI-6211 'FPP Mass Dummy Endplate metrology'
    # Use Sphere label 1 2 3 4 5 as described in DESI-6211
    # Use column 'actual' (instead of 'nominal')
    # each entry in x,y,z
    # pin labeled 'A' has coordinates x=0,y=0
    # pin labeled 'B' has coordinates x=0,y=+49.9844149
    bmr_6211_mm = np.zeros((3,5))
    bmr_6211_mm[:,0]=[-49.1142666,-72.9231945,-25.5298635]
    bmr_6211_mm[:,1]=[54.3977431,-106.5602541,-25.5191309]
    bmr_6211_mm[:,2]=[118.6746571,-18.4486229,-25.552428]
    bmr_6211_mm[:,3]=[54.37223,69.8032562,-25.6009506]
    bmr_6211_mm[:,4]=[-49.2316389,35.6339609,-25.5679048]

    xyz_6206 = np.array([ [0.,0.,0.] , [-50.,0.,0.] ]).T
    xyz_6211 = np.array([ [0.,50.,0.] , [0.,0.,0.] ]).T
    C6211_to_C6206 = Transfo()
    rms = C6211_to_C6206.fit(xyz_6211,xyz_6206)
    if debug: print("DEBUG rms(6211->6206) = {:.3f} mmm".format(rms))

    bmr_6206_mm = C6211_to_C6206.apply(bmr_6211_mm)
    bmr_CS5_mm  = C6206_to_CS5.apply(bmr_6206_mm)

    if plot :
        import matplotlib.pyplot as plt
        plt.figure("CS5")
        plt.plot(red_leg_mount_holes_CS5_mm[0,0],red_leg_mount_holes_CS5_mm[1,0],"o",color="b",label="hole1")
        plt.plot(red_leg_mount_holes_CS5_mm[0,1],red_leg_mount_holes_CS5_mm[1,1],"o",color="g",label="hole2")
        plt.plot(red_leg_mount_holes_CS5_mm[0,2],red_leg_mount_holes_CS5_mm[1,2],"o",color="r",label="hole3")
        plt.plot(bmr_CS5_mm[0],bmr_CS5_mm[1],"X",markersize=12,label="bmr")
        xyz_CS5 = C6206_to_CS5.apply(xyz_6206)
        plt.plot(xyz_CS5[0,0],xyz_CS5[1,0],"X",color="gray",label="A")
        plt.plot(xyz_CS5[0,1],xyz_CS5[1,1],".",color="gray")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid()
        plt.xlabel("X_CS5 (mm)")
        plt.ylabel("Y_CS5 (mm)")

        if 0 :
            plt.figure("C6206")
            plt.plot(red_leg_mount_holes_6206_mm[0,0],red_leg_mount_holes_6206_mm[1,0],"o",color="b",label="hole1")
            plt.plot(red_leg_mount_holes_6206_mm[0,1],red_leg_mount_holes_6206_mm[1,1],"o",color="g",label="hole2")
            plt.plot(red_leg_mount_holes_6206_mm[0,2],red_leg_mount_holes_6206_mm[1,2],"o",color="r",label="hole3")

            plt.plot(bmr_6206_mm[0],bmr_6206_mm[1],"o",markersize=12,label="bmr")
            plt.plot(xyz_6206[0,0],xyz_6206[1,0],"X",color="gray",label="A")
            plt.plot(xyz_6206[0,1],xyz_6206[1,1],".",color="gray")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid()
            plt.xlabel("X_DESI-6206 (mm)")
            plt.ylabel("Y_DESI-6206 (mm)")

    z_pma_inch = 35.397+15.04 # z of strut + z of difference between strut and ball
    z_pma_mm   = z_pma_inch*inch2mm

    return bmr_CS5_mm , z_pma_mm


def adjust_rail_axis(petal,carriage_z,bmr_xyz_cs5_inch,rail_xyz_cs5_inch) :

    # series of points along the rails
    bmr_xyz = CS5_to_PMA_inch(bmr_xyz_cs5_inch,carriage_z)
    pivot_xyz = np.mean(bmr_xyz,axis=1)
    rail_xyz = CS5_to_PMA_inch(rail_xyz_cs5_inch,carriage_z)

    dxyz=rail_xyz-pivot_xyz[:,None]
    x=dxyz[0]
    y=dxyz[1]
    z=dxyz[2]
    cx=np.polyfit(z,x,1)
    dxdz=cx[0]
    cy=np.polyfit(z,y,1)
    dydz=cy[0]
    #print("dxdz=",dxdz)
    #print("dydz=",dydz)
    dxyz_target = np.array([[0.,0.,0.],[0.,0.,1.]]).T
    dxyz_meas   = dxyz_target.copy()
    dxyz_meas[0] += dxdz*dxyz_target[2]
    dxyz_meas[1] += dydz*dxyz_target[2]

    transfo=Rot2D()
    transfo.center = np.zeros(3) # center is zero
    transfo.fit(dxyz_meas,dxyz_target)
    transfo.center = pivot_xyz # center is mean of bmr
    print("Fitted transfo for rail alignment:")
    print(transfo)
    transformed_rail_xyz = transfo.apply(rail_xyz)
    print("DEBUG before rail fit, dx (mm) =",array2str((rail_xyz[0]-np.mean(rail_xyz[0]))*inch2mm))
    print("DEBUG before rail fit, dy (mm) =",array2str((rail_xyz[1]-np.mean(rail_xyz[1]))*inch2mm))
    print("DEBUG after  rail fit, dx (mm) =",array2str((transformed_rail_xyz[0]-np.mean(transformed_rail_xyz[0]))*inch2mm))
    print("DEBUG after  rail fit, dy (mm) =",array2str((transformed_rail_xyz[1]-np.mean(transformed_rail_xyz[1]))*inch2mm))

    # check change of coords for bmrs
    transformed_bmr_xyz = transfo.apply(bmr_xyz)
    dxyz = transformed_bmr_xyz - bmr_xyz
    dist_mm = np.sqrt(dxyz[0]**2+dxyz[1]**2)*inch2mm
    print("DEBUG after rail fit, dr(bmr) (mm) =",array2str(dist_mm))
    return transfo


def get_lower_struts_cs5_inch():
    strut_base_xyz_cs5_inch = np.zeros((3,6),dtype=float)
    strut_plateform_xyz_cs5_inch = np.zeros((3,6),dtype=float)

    strut_base_xyz_cs5_inch[:,0]=[ 37.734, 78.241, 8.322 ]
    strut_base_xyz_cs5_inch[:,1]=[ 78.567, 37.050, 8.322 ]
    strut_base_xyz_cs5_inch[:,2]=[ 58.151, 57.645, -106.679 ]
    strut_base_xyz_cs5_inch[:,3]=[ 22.999, 63.633, -44.673 ]
    strut_base_xyz_cs5_inch[:,4]=[ 63.831, 22.442, -44.673 ]
    strut_base_xyz_cs5_inch[:,5]=[ 24.786, 62.589, -109.214 ]
    strut_plateform_xyz_cs5_inch[:,0]=[ 23.369, 64.001, 5.785 ]
    strut_plateform_xyz_cs5_inch[:,1]=[ 64.202, 22.810, 5.785 ]
    strut_plateform_xyz_cs5_inch[:,2]=[ 43.786, 43.405, -109.215 ]
    strut_plateform_xyz_cs5_inch[:,3]=[ 24.790, 65.409, -64.902 ]
    strut_plateform_xyz_cs5_inch[:,4]=[ 65.623, 24.218, -64.902 ]
    strut_plateform_xyz_cs5_inch[:,5]=[ 39.129, 48.102, -109.215 ]

    return strut_base_xyz_cs5_inch , strut_plateform_xyz_cs5_inch

def compute_leg_center_of_rotation_cs5_mm(petal,plot=True) :
    # from drawing provided by Bobby
    # (file 'PMA Servicing Setup rot axes dims.PDF')
    # here are the 'light' bmr coordinates
    # in a coordinate system centered on the leg axis of rotation,
    # that we call it CCS here,
    bmr_xyz_ccs_mm = np.zeros((3,4))
    bmr_xyz_ccs_mm[:,0]=[9.49,86.76,0]
    bmr_xyz_ccs_mm[:,1]=[-1.77,-52.50,0]
    bmr_xyz_ccs_mm[:,2]=[41.37,-110.28,0]
    bmr_xyz_ccs_mm[:,3]=[96.61,45.20,0]

    # get coordinates in cs5
    bmr_xyz_cs5_mm,junk = compute_target_bmr_light_weight_red_leg_coords_mm(petal)
    bmr_xyz_ccs_mm[2] = np.mean(bmr_xyz_cs5_mm[2])

    # fit transfo
    transfo = Transfo2D()
    rms=transfo.fit(bmr_xyz_ccs_mm,bmr_xyz_cs5_mm)
    #print("DEBUG rms rotation center coord fit (mm) = ",rms)

    # apply transfo to the center (which is 0,0,0)
    center=np.zeros(3)
    center=transfo.apply(center)

    if plot :
        plt.figure("CS5")
        plt.plot(center[0],center[1],"+",color="purple")


    return center

def str2array(vals) :
    res = [float(v) for v in vals.split(",")]
    return np.array(res)

def main() :

    # Read inputs
    #################################################
    if len(sys.argv)>1 :
        ifilename = sys.argv[1]
    else :
        print("""
Please add as an argument a filename.
I will use a default file for now as a code test.
        """)
        ifilename = "test-petal0.yaml"

    if not os.path.isfile(ifilename) :
        print("cannot find or open",ifilename)
        sys.exit(1)

    print("Input filename:",ifilename)

    with open(ifilename) as ifile :
        inputs=yaml.safe_load(ifile)

    petal = int(inputs["petal"])
    print("Petal:",petal)
    print("BMR: '{}'".format(inputs['bmr_type']))

    if "plot" in inputs :
        val = int(inputs["plot"])
        assert (val in [0,1])
        plot = (val==1)
    else :
        plot = True


    # Compute target bmr coords in CS5 from metrology
    #################################################
    if inputs["bmr_type"] == "guide_spikes" or inputs["bmr_type"] == "bracket" :
        target_bmr_CS5_mm , target_bmr_z_pma_mm  = compute_target_bmr_guide_spikes_coords_mm(petal,plot=plot)
    elif inputs["bmr_type"]== "light_weight_red_leg" :
        target_bmr_CS5_mm , target_bmr_z_pma_mm = compute_target_bmr_light_weight_red_leg_coords_mm(petal,plot=plot)
    elif inputs["bmr_type"]== "heavy_weight_red_leg" :
        target_bmr_CS5_mm , target_bmr_z_pma_mm = compute_target_bmr_heavy_weight_red_leg_coords_mm(petal,plot=plot)
    else :
        print('error {} not in ["guide_spikes","bracket","light_weight_red_leg","heavy_weight_red_leg"]'.format(inputs["bmr_type"]))
        sys.exit(2)
    target_bmr_CS5_inch = target_bmr_CS5_mm/inch2mm

    print("Targets locations CS5 inch")
    for b in range(target_bmr_CS5_inch.shape[1]) :
        print("{} {}".format(b,array2str(target_bmr_CS5_inch[:,b])))
    print("Mean target CS5 z (mm)=",np.mean(target_bmr_CS5_inch[2]*inch2mm))
    #################################################





    # Input BMR Locations (in CS5, will change input method later)
    #################################################

    if inputs["bmr_type"] == "heavy_weight_red_leg" :
        bmr_labels=["B1","B2","B3","B4","B5"]
    else :
        bmr_labels=["B1","B2","B3","B4"]

    number_of_balls=len(bmr_labels)
    measured_bmr_CS5_inch = np.zeros((3,number_of_balls))



    # Check if config with bracket
    #################################################
    if inputs["bmr_type"] == "bracket" :
        print("Read calibration data first")

        calib_gs_bmr_labels=["CALIB_GS_B1","CALIB_GS_B2","CALIB_GS_B3","CALIB_GS_B4"]
        calib_gs_bmr_CS5_inch=np.zeros((3,4))
        for index,bmr_label in enumerate( calib_gs_bmr_labels) :
            if not bmr_label in inputs:
                print("error, need coordinates for CALIB_GS_B1 CALIB_GS_B2 CALIB_GS_B3 CALIB_GS_B4")
                sys.exit(1)
            calib_gs_bmr_CS5_inch[:,index] = str2array(inputs[bmr_label])
            print("{} {}".format(bmr_label,inputs[bmr_label]))

        calib_bracket_bmr_labels=["CALIB_BRACKET_B1","CALIB_BRACKET_B2","CALIB_BRACKET_B3","CALIB_BRACKET_B4"]
        calib_bracket_bmr_CS5_inch=np.zeros((3,4))
        for index,bmr_label in enumerate( calib_bracket_bmr_labels) :
            if not bmr_label in inputs:
                print("error, need coordinates for CALIB_BRACKET_B1 CALIB_BRACKET_B2 CALIB_BRACKET_B3 CALIB_BRACKET_B4")
                sys.exit(1)
            calib_bracket_bmr_CS5_inch[:,index] = str2array(inputs[bmr_label])
            print("{} {}".format(bmr_label,inputs[bmr_label]))

    print("Input BMR coordinates (inch):")
    valid_bmr = np.repeat(False,number_of_balls)
    for index,bmr_label in enumerate(bmr_labels) :
        if bmr_label in inputs :
            measured_bmr_CS5_inch[:,index] = str2array(inputs[bmr_label])
            valid_bmr[index]=True
            print("{} {}".format(bmr_label,inputs[bmr_label]))
        else :
            print("WARNING: no data for ball '{}'".format(bmr_label))
            valid_bmr[index]=False

    if inputs["bmr_type"] == "bracket" :
        print("Fit transform between calib and current bracket bmr data")
        transfo = Transfo2D()
        rms = transfo.fit(calib_bracket_bmr_CS5_inch[:,valid_bmr],measured_bmr_CS5_inch[:,valid_bmr],test_mirror=False)
        print(transfo)
        print("rms of fit residuals = {:.4f} mm".format(rms*inch2mm))
        print("Apply transform to calib guide spikes bmr data")
        measured_bmr_CS5_inch = transfo.apply(calib_gs_bmr_CS5_inch)
        print("Now we consider we have the GS data")




    # Input points along leg rail (in CS5)
    #################################################
    rail_xyz_cs5_inch = None
    for p in range(1,100) :
        key="RAIL{}".format(p)
        if key in inputs :
            if rail_xyz_cs5_inch is None :
                rail_xyz_cs5_inch = []
            rail_xyz_cs5_inch.append(str2array(inputs[key]))
    if rail_xyz_cs5_inch is not None :
        rail_xyz_cs5_inch = np.array(rail_xyz_cs5_inch).T
        print("Read the coordinates of {} points along the leg rail:".format(rail_xyz_cs5_inch.shape[1]))
        print(rail_xyz_cs5_inch.T)
        if rail_xyz_cs5_inch.shape[1]<2 :
            print("ERROR: need at least two points to measure the rail axis")
            print("I will not try this")
            rail_xyz_cs5_inch=None

    # Forced deltas
    #################################################
    forced_delta_x_cs5_mm = None
    forced_delta_x_cs5_mm = None
    forced_delta_x_pma_mm = None
    forced_delta_x_pma_mm = None
    if "FORCED_DX_CS5_MM" in inputs and not "FORCED_DY_CS5_MM" in inputs :
        print("ERROR: in configuration yaml file, need both FORCED_DX_CS5_MM and FORCED_DY_CS5_MM or none.")
        sys.exit(12)
    if "FORCED_DY_CS5_MM" in inputs and not "FORCED_DX_CS5_MM" in inputs :
        print("ERROR: in configuration yaml file, need both FORCED_DX_CS5_MM and FORCED_DY_CS5_MM or none.")
        sys.exit(12)
    if "FORCED_DX_CS5_MM" in inputs and "FORCED_DY_CS5_MM" in inputs :
        forced_delta_x_cs5_mm = float(inputs["FORCED_DX_CS5_MM"])
        forced_delta_y_cs5_mm = float(inputs["FORCED_DY_CS5_MM"])
    if "FORCED_DX_PMA_MM" in inputs and not "FORCED_DY_PMA_MM" in inputs :
        print("ERROR: in configuration yaml file, need both FORCED_DX_PMA_MM and FORCED_DY_PMA_MM or none.")
        sys.exit(12)
    if "FORCED_DY_PMA_MM" in inputs and not "FORCED_DX_PMA_MM" in inputs :
        print("ERROR: in configuration yaml file, need both FORCED_DX_PMA_MM and FORCED_DY_PMA_MM or none.")
        sys.exit(12)

    if "FORCED_DX_PMA_MM" in inputs and "FORCED_DY_PMA_MM" in inputs :
        if "FORCED_DX_CS5_MM" in inputs or "FORCED_DY_CS5_MM" in inputs :
            print("ERROR: in configuration file, cannot have both FORCED_DX_CS5_MM and FORCED_DX_PMA_MM")
            sys.exit(12)
        forced_delta_x_pma_mm = float(inputs["FORCED_DX_PMA_MM"])
        forced_delta_y_pma_mm = float(inputs["FORCED_DY_PMA_MM"])

    # Input PMA translation axis misalignment params
    #################################################
    if not "correct_pma_misalignement" in inputs :
        print("WARNING no keyword 'correct_pma_misalignement' found")
        print("I assume you don't want to correct from the PMA translation axis misalignment.")
        correct_pma_misalignement = False
    else :
        val = int(inputs["correct_pma_misalignement"])
        assert (val in [0,1])
        correct_pma_misalignement = (val==1)

    if correct_pma_misalignement :
        # measurements of changes of PMA translation axis alignment
        # will measure twice the same fixed point in the PMA
        # once with carriage engaged as close as possible to focal plane
        # once with carriage in std retracted rest position
        measured_pma_partially_engaged_bmr_CS5_inch = str2array(inputs["partially_engaged_pma_coords_inch"])
        measured_pma_retracted_bmr_CS5_inch = str2array(inputs["retracted_pma_coords_inch"])
        # z_CS5 of fixed point when PMA fully engaged
        pma_fully_engaged_bmr_z_CS5_inch = float(inputs["fully_engaged_pma_z_coord_inch"])
        print("Input PMA translation axis misalignment (inch):")
        print("  coords when partially engaged:",array2str(measured_pma_partially_engaged_bmr_CS5_inch))
        print("  coords when retracted        :",array2str(measured_pma_retracted_bmr_CS5_inch))
        print("  z coord when fully engaged   : {:.3f}".format(pma_fully_engaged_bmr_z_CS5_inch))
    ######################################################################

    if not "outfile" in inputs :
        outfile=None
        print("WARNING no keyword 'outfile' found")
    else :
        outfile=inputs["outfile"]

    correct_pma_arm_rotation = False
    if not "correct_pma_arm_rotation" in inputs :
        print("WARNING no keyword 'correct_pma_arm_rotation' found")
        print("I assume you don't want to correct from the PMA arm rotation.")
    else :
        val = int(inputs["correct_pma_arm_rotation"])
        assert (val in [0,1])
        correct_pma_arm_rotation = (val==1)
        if correct_pma_arm_rotation :
            print("Will correct for the PMA arm rotation")

    fixed_pma_leg_rotation_phi = None
    if "fixed_pma_leg_rotation_phi" in inputs :
        fixed_pma_leg_rotation_phi = float(inputs["fixed_pma_leg_rotation_phi"])
        print("Will FIX the PMA leg rotation angle PHI to {:+.3f} deg".format(fixed_pma_leg_rotation_phi))

    if fixed_pma_leg_rotation_phi :
        correct_pma_leg_rotation = True
    else :
        correct_pma_leg_rotation = False
        if not "correct_pma_leg_rotation" in inputs :
            print("WARNING no keyword 'correct_pma_leg_rotation' found")
            print("I assume you don't want to correct from the PMA leg rotation.")
        else :
            val = int(inputs["correct_pma_leg_rotation"])
            assert (val in [0,1])
            correct_pma_leg_rotation = (val==1)
            if correct_pma_leg_rotation :
                print("Will correct for the PMA leg rotation")



    # Compute PMA translation correction
    ##############################################################
    if correct_pma_misalignement :

        # First extrapolate PMA measurement to fully engage location
        measured_pma_fully_engaged_bmr_CS5_inch = measured_pma_partially_engaged_bmr_CS5_inch + (measured_pma_partially_engaged_bmr_CS5_inch-measured_pma_retracted_bmr_CS5_inch) * (pma_fully_engaged_bmr_z_CS5_inch-measured_pma_partially_engaged_bmr_CS5_inch[2])/(measured_pma_partially_engaged_bmr_CS5_inch[2]-measured_pma_retracted_bmr_CS5_inch[2])

        # Correction to apply to the input coordinates when retracted
        # so the final coords after engaging the carriage are correct.
        correction_when_retracting_pma_inch = measured_pma_retracted_bmr_CS5_inch - measured_pma_fully_engaged_bmr_CS5_inch

        dtrans = np.sqrt(correction_when_retracting_pma_inch[0]**2+correction_when_retracting_pma_inch[1]**2)
        angle_deg  = np.arctan(dtrans/correction_when_retracting_pma_inch[2])*180./np.pi
        print("=================================================")
        print("Apply correction for PMA misalignment of {:.3f} degrees of dx,dy= {:+.3f},{:+.3f} inch".format(angle_deg,
                                                                                                                   correction_when_retracting_pma_inch[0],correction_when_retracting_pma_inch[1]))
        if len(target_bmr_CS5_inch.shape) == 1 :
            target_bmr_CS5_inch[0:2] += correction_when_retracting_pma_inch[0:2] # change only x and y
        else :
            target_bmr_CS5_inch[0:2] += correction_when_retracting_pma_inch[0:2][:,None]

    if not "correct_lower_struts_length" in inputs :
        print("WARNING no keyword 'correct_lower_struts_length' found")
        print("I assume you don't want to correct from the lower (sled) struts length.")
        correct_lower_struts_length = False
    else :
        val = int(inputs["correct_lower_struts_length"])
        assert (val in [0,1])
        correct_lower_struts_length = (val==1)
        if correct_lower_struts_length :
            print("Will correct for the lower (sled) struts length")

    if not "lower_struts_xy_translation" in inputs :
        lower_struts_xy_translation = False
    else :
        val = int(inputs["lower_struts_xy_translation"])
        assert (val in [0,1])
        lower_struts_xy_translation = (val==1)
        if lower_struts_xy_translation :
            print("Will use lower (sled) struts to adjust xy translation")

    ##############################################################

    moves = dict()
    moves_description = dict()

    # test
    #a = 5.*np.pi/180
    #ca = np.cos(a)
    #sa = np.sin(a)
    #rot= np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    #xyz = np.array([0,1,0])
    #print(rot.dot(xyz))
    #measured_bmr_CS5_inch = rot.dot(measured_bmr_CS5_inch)

    # first measure a residual rotation angle for the PMA arm
    x1 = target_bmr_CS5_inch[0,valid_bmr]
    y1 = target_bmr_CS5_inch[1,valid_bmr]
    r1 = np.sqrt(x1**2+y1**2)
    x2 = measured_bmr_CS5_inch[0,valid_bmr]
    y2 = measured_bmr_CS5_inch[1,valid_bmr]
    r2 = np.sqrt(x2**2+y2**2)
    angles_deg = np.arcsin((-x2*y1+x1*y2)/(r1*r2)) *180/np.pi
    mean_angle_deg = np.mean(angles_deg)
    rms = np.std(angles_deg)
    err = rms/np.sqrt(angles_deg.size)

    print("=================================================")
    print("Arm rotation angle correction to apply = {:+.2f} +- {:.2f} deg".format(-mean_angle_deg,err))
    print("(Positive is clockwise when looking at the telescope from the PMA)")


    moves["THETA"]=0.
    moves_description["THETA"]=["PMA arm rotation angle, degrees"]

    if correct_pma_arm_rotation :
        print("APPLY ARM ROTATION delta THETA = {:+.2f}".format(-mean_angle_deg))
        moves["THETA"] = -mean_angle_deg # correction is opposite of effect
        ca=np.cos(-mean_angle_deg*np.pi/180)
        sa=np.sin(-mean_angle_deg*np.pi/180)
        rot= np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
        measured_bmr_CS5_inch = rot.dot(measured_bmr_CS5_inch)

        # debug test
        #x2 = measured_bmr_CS5_inch[0,valid_bmr]
        #y2 = measured_bmr_CS5_inch[1,valid_bmr]
        #r2 = np.sqrt(x2**2+y2**2)
        #angles_deg = np.arcsin((-x2*y1+x1*y2)/(r1*r2)) *180/np.pi
        #mean_angle_deg = np.mean(angles_deg)
        #print("Is this zero?",mean_angle_deg)
    print("=================================================")

    # measure local rotation of red leg (I don't know yet center of rotation")
    x1 = target_bmr_CS5_inch[0,valid_bmr]
    y1 = target_bmr_CS5_inch[1,valid_bmr]
    x2 = measured_bmr_CS5_inch[0,valid_bmr]
    y2 = measured_bmr_CS5_inch[1,valid_bmr]

    xyz_center = compute_leg_center_of_rotation_cs5_mm(petal)/inch2mm
    xc  = xyz_center[0]
    yc  = xyz_center[1]
    print("xc=",xc,"yc=",yc)

    dx1=x1-xc
    dy1=y1-yc
    dx2=x2-xc
    dy2=y2-yc

    dr1=np.sqrt(dx1**2+dy1**2)
    dr2=np.sqrt(dx2**2+dy2**2)

    angles_deg = np.arcsin((-dx2*dy1+dx1*dy2)/(dr1*dr2)) *180/np.pi
    mean_leg_angle_deg = np.mean(angles_deg)
    rms = np.std(angles_deg)
    err = rms/np.sqrt(angles_deg.size)

    if fixed_pma_leg_rotation_phi is not None :
        print("Measured a rotation of {:+.2f} BUT will apply a rotation of {:+.2f} deg".format( -mean_leg_angle_deg,fixed_pma_leg_rotation_phi))
        delta_phi = fixed_pma_leg_rotation_phi
        print("(fixed_pma_leg_rotation_phi)")
    else :
        delta_phi = -mean_leg_angle_deg  # correction is opposite of effect
        print("Measured leg rotation angle correction to apply = {:+.2f} deg".format(delta_phi))

    print("(Positive is clockwise when looking at the telescope from the PMA)")

    moves["PHI"]=0.
    moves_description["PHI"]=["PMA new leg rotation angle, degrees"]

    if correct_pma_leg_rotation or (fixed_pma_leg_rotation_phi is not None):
        print("APPLY LEG ROTATION delta PHI = {:+.2f}".format(delta_phi))
        moves["PHI"] = delta_phi
        measured_bmr_CS5_inch[0] -= xc
        measured_bmr_CS5_inch[1] -= yc
        ca=np.cos(delta_phi*np.pi/180)
        sa=np.sin(delta_phi*np.pi/180)
        rot= np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
        measured_bmr_CS5_inch = rot.dot(measured_bmr_CS5_inch)
        measured_bmr_CS5_inch[0] += xc
        measured_bmr_CS5_inch[1] += yc

    print("=================================================")
##############################################################

    carriage_z = -60 # dummy value before adjustment
    for loop in range(2) :
        target_bmr_PMA_inch   = CS5_to_PMA_inch(target_bmr_CS5_inch,carriage_z = carriage_z)
        measured_bmr_PMA_inch = CS5_to_PMA_inch(measured_bmr_CS5_inch,carriage_z = carriage_z)

        measured_z_pma_inch = np.mean(measured_bmr_PMA_inch[2][valid_bmr])
        target_bmr_z_pma_inch = target_bmr_z_pma_mm / inch2mm

        delta_z = target_bmr_z_pma_inch-measured_z_pma_inch
        #print("for carriage_z = {:+.3f} delta_z = {:+.3f}".format(carriage_z,delta_z))
        carriage_z -= delta_z


    target_bmr_PMA_inch   = CS5_to_PMA_inch(target_bmr_CS5_inch,carriage_z = carriage_z)


    if forced_delta_x_cs5_mm is not None :
        print ("!! WARNING: replace BMR data by direct offset of coordinates in CS5 dx_mm={:.2f} dy_mm={:.2f} !!".format(forced_delta_x_cs5_mm,forced_delta_y_cs5_mm))
        measured_bmr_CS5_inch = target_bmr_CS5_inch.copy()
        measured_bmr_CS5_inch[0] += forced_delta_x_cs5_mm/inch2mm
        measured_bmr_CS5_inch[1] += forced_delta_y_cs5_mm/inch2mm

    measured_bmr_PMA_inch = CS5_to_PMA_inch(measured_bmr_CS5_inch,carriage_z = carriage_z)

    if forced_delta_x_pma_mm is not None :
        print ("!! WARNING: replace BMR data by direct offset of coordinates in PMA dx_mm={:.2f} dy_mm={:.2f} !!".format(forced_delta_x_pma_mm,forced_delta_y_pma_mm))
        measured_bmr_PMA_inch = target_bmr_PMA_inch.copy()
        measured_bmr_PMA_inch[0] += forced_delta_x_pma_mm/inch2mm
        measured_bmr_PMA_inch[1] += forced_delta_y_pma_mm/inch2mm

    print("Fitted carriage z            = {:+0.2f} inch".format(carriage_z))
    print("mean z target_bmr (pma CS)   = {:+0.2f} inch".format(np.mean(target_bmr_PMA_inch[2])))
    print("mean z measured_bmr (pma CS) = {:+0.2f} inch".format(np.mean(measured_bmr_PMA_inch[2])))

    if True : # Ignore difference in z
        print("Ignore average difference in z between measurements and targets")
        measured_bmr_PMA_inch[2,valid_bmr] += ( np.mean(target_bmr_PMA_inch[2,valid_bmr]) - np.mean(measured_bmr_PMA_inch[2,valid_bmr]) )

    ##############################################################
    ## sled lower struts
    lower_strut_base_cs5_inch , lower_strut_plateform_cs5_inch = get_lower_struts_cs5_inch()
    lower_struts_labels = ["LS1","LS2","LS3","LS4","LS5","LS6"]
    lower_struts_base_xyz      = CS5_to_PMA_inch(lower_strut_base_cs5_inch,carriage_z=carriage_z)
    initial_lower_struts_plateform_xyz = CS5_to_PMA_inch(lower_strut_plateform_cs5_inch,carriage_z=carriage_z)


    ## pma upper struts
    upper_struts_labels = ["US1","US2","US3","US4","US5","US6"]

    # Reference coordinates of the base ends of the PMA struts in the PMA CS
    upper_struts_base_xyz = np.array([[-26.875, -26.875, 26.875, 26.875, 15.475, 31.25],
                                      [-43.811, -42.561, -43.811, -42.561, -42.561, -43.811],
                                      [ -6.76, 8.49, -6.76, 8.49, 35.397, 35.397]])

    # Reference coordinates of the plateform ends of the PMA struts in the PMA CS
    initial_upper_struts_plateform_xyz = np.array([[-26.875, -26.875, 26.875, 26.875, 26.475, 31.25],
                                                   [-32.811, -42.561, -32.811, -42.561, -42.561, -32.811],
                                                   [ -6.76, -2.51, -6.76, -2.51, 35.397, 35.397]])

    print("=================================================")


    delta_inch_CS5 = (measured_bmr_CS5_inch-target_bmr_CS5_inch)
    dr_inch_CS5 = np.sqrt(delta_inch_CS5[0]**2+delta_inch_CS5[1]**2)
    print("BMR offsets CS5 (sqrt(dx2+dy2), inch) =",array2str(dr_inch_CS5[valid_bmr]))
    #print("BMR mean offset dx (CS5) = {:+.3f} inch".format(np.mean(delta_inch_CS5[0][valid_bmr])))
    #print("BMR mean offset dy (CS5)  = {:+.3f} inch".format(np.mean(delta_inch_CS5[1][valid_bmr])))

    print("BMR offsets (sqrt(dx2+dy2), mm)   =",array2str(dr_inch_CS5[valid_bmr]*inch2mm))
    print("BMR mean offset dx (CS5)  = {:+.3f} mm".format(np.mean(delta_inch_CS5[0][valid_bmr])*inch2mm))
    print("BMR mean offset dy (CS5)  = {:+.3f} mm".format(np.mean(delta_inch_CS5[1][valid_bmr])*inch2mm))

    delta_inch_PMA = (measured_bmr_PMA_inch-target_bmr_PMA_inch)
    #dr_inch_PMA = np.sqrt(delta_inch_PMA[0]**2+delta_inch_PMA[1]**2)
    #print("BMR offsets PMA (sqrt(dx2+dy2), inch) =",array2str(dr_inch_PMA[valid_bmr]))
    print("BMR mean offset dx (PMA)  = {:+.3f} mm (horizontal, positive towards the laser tracker)".format(np.mean(delta_inch_PMA[0][valid_bmr])*inch2mm))
    print("BMR mean offset dy (PMA)  = {:+.3f} mm (vertical, positive upward)".format(np.mean(delta_inch_PMA[1][valid_bmr])*inch2mm))
    #print("BMR mean offset dx (PMA) = {:+.3f} inch".format(np.mean(delta_inch[0][valid_bmr])))
    #print("BMR mean offset dy (PMA)  = {:+.3f} inch".format(np.mean(delta_inch[1][valid_bmr])))
    print("=================================================")

    for name in lower_struts_labels :
        moves[name]=0.
        moves_description[name]=["Lower (sled) strut #{}, change of length, mm".format(name[-1])]

    if correct_lower_struts_length :
        print("Sled struts adjustment")

        sled_adjust = Transfo2D()

        if not lower_struts_xy_translation  :
            print(" Fit pure vertical translation")
            sled_adjust.angle=0.
            sled_adjust.t[0]=0.
            sled_adjust.t[1]=np.mean(target_bmr_PMA_inch[1][valid_bmr]-measured_bmr_PMA_inch[1][valid_bmr])
        else :
            print(" Fit pure xy translation")
            sled_adjust.angle=0.
            sled_adjust.t[0]=np.mean(target_bmr_PMA_inch[0][valid_bmr]-measured_bmr_PMA_inch[0][valid_bmr])
            sled_adjust.t[1]=np.mean(target_bmr_PMA_inch[1][valid_bmr]-measured_bmr_PMA_inch[1][valid_bmr])

        #else  :
        #    print(" Fit xy translation + rotation about z axis")
        #    rms = sled_adjust.fit(measured_bmr_PMA_inch[:,valid_bmr],target_bmr_PMA_inch[:,valid_bmr],test_mirror=False)

        predicted_new_bmr_PMA_inch = sled_adjust.apply(measured_bmr_PMA_inch)
        dist2_inch = np.sum((predicted_new_bmr_PMA_inch - target_bmr_PMA_inch)**2,axis=0)
        rms_inch   = np.sqrt(np.mean(dist2_inch[valid_bmr]))
        rms_mm = rms_inch*inch2mm
        print("BMR fit rms = {:.3f} mm".format(rms_mm))
        print(sled_adjust)

        #print("Apply this transform to the lower struts")
        new_lower_struts_plateform_xyz = sled_adjust.apply(initial_lower_struts_plateform_xyz)

        initial_lower_struts_length_inch = np.sqrt(np.sum((initial_lower_struts_plateform_xyz - lower_struts_base_xyz)**2,axis=0))
        new_lower_struts_length_inch = np.sqrt(np.sum((new_lower_struts_plateform_xyz - lower_struts_base_xyz)**2,axis=0))
        lower_strut_deltas_inch = new_lower_struts_length_inch - initial_lower_struts_length_inch

        #print("Initial lower_struts length (inch) =",array2str(initial_lower_struts_length_inch))
        #print("New lower_struts length (inch)     =",array2str(new_lower_struts_length_inch))

        print("Initial lower_struts length (mm) =",array2str(initial_lower_struts_length_inch*inch2mm))
        print("New lower_struts length (mm)     =",array2str(new_lower_struts_length_inch*inch2mm))
        print("Lower_Struts deltas (mm):")
        for s,d in enumerate(lower_strut_deltas_inch) :
            print("  {} {:+.3f}".format(lower_struts_labels[s],d*inch2mm))
            moves[lower_struts_labels[s]]=d*inch2mm
        print("=================================================")
        print("Also apply this transform to the bmr")
        measured_bmr_PMA_inch = sled_adjust.apply(measured_bmr_PMA_inch)
        print("Remove average z offset")
        measured_bmr_PMA_inch[2] += np.mean(target_bmr_PMA_inch[2][valid_bmr]-measured_bmr_PMA_inch[2][valid_bmr])
        delta_inch = measured_bmr_PMA_inch[:,valid_bmr]-target_bmr_PMA_inch[:,valid_bmr]
        delta_mm   = delta_inch*inch2mm

        print("Corr. BMR mean offset dx (PMA)  = {:+.3f} mm".format(np.mean(delta_mm[0])))
        print("Corr. BMR mean offset dy (PMA)  = {:+.3f} mm".format(np.mean(delta_mm[1])))
        print("=================================================")


    for name in upper_struts_labels :
        moves[name]=0.
        moves_description[name]=["Upper (PMA) strut #{}, change of length, mm".format(name[-1])]

    print("PMA struts adjustment")
    pma_adjust = Transfo2D()
    if 0 :
        print("Force pure translation")
        pma_adjust.t[0] = np.mean(target_bmr_PMA_inch[0][valid_bmr]-measured_bmr_PMA_inch[0][valid_bmr])
        pma_adjust.t[1] = np.mean(target_bmr_PMA_inch[1][valid_bmr]-measured_bmr_PMA_inch[1][valid_bmr])
        pma_adjust.angle = 0
    else :
        rms = pma_adjust.fit(measured_bmr_PMA_inch[:,valid_bmr],target_bmr_PMA_inch[:,valid_bmr],test_mirror=False)

    predicted_new_bmr_PMA_inch = pma_adjust.apply(measured_bmr_PMA_inch)
    dist2_inch = np.sum((predicted_new_bmr_PMA_inch - target_bmr_PMA_inch)**2,axis=0)
    rms_inch   = np.sqrt(np.mean(dist2_inch[valid_bmr]))
    rms_mm = rms_inch*inch2mm
    print("BMR fit rms = {:.3f} mm".format(rms_mm))
    print(pma_adjust)

    new_upper_struts_plateform_xyz = pma_adjust.apply(initial_upper_struts_plateform_xyz)
    initial_upper_struts_length_inch = np.sqrt(np.sum((initial_upper_struts_plateform_xyz - upper_struts_base_xyz)**2,axis=0))
    new_upper_struts_length_inch = np.sqrt(np.sum((new_upper_struts_plateform_xyz - upper_struts_base_xyz)**2,axis=0))
    #print("Initial upper_struts length (inch) =",array2str(initial_upper_struts_length_inch))
    #print("New upper_struts length (inch)     =",array2str(new_upper_struts_length_inch))
    upper_strut_deltas_inch = new_upper_struts_length_inch - initial_upper_struts_length_inch
    #print("Upper struts deltas (inch):")
    #for s,d in enumerate(upper_strut_deltas_inch) :
    #    print("  {} {:+.3f}".format(upper_struts_labels[s],d))

    print("Initial upper_struts length (mm) =",array2str(initial_upper_struts_length_inch*inch2mm))
    print("New upper_struts length (mm)     =",array2str(new_upper_struts_length_inch*inch2mm))

    print("Upper struts deltas (mm):")
    for s,d in enumerate(upper_strut_deltas_inch) :
        print("  {} {:+.3f}".format(upper_struts_labels[s],d*inch2mm))
        moves[upper_struts_labels[s]]=d*inch2mm

    predicted_new_bmr_PMA_inch = pma_adjust.apply(measured_bmr_PMA_inch)
    dist2_inch = np.sum((predicted_new_bmr_PMA_inch - target_bmr_PMA_inch)**2,axis=0)

    #print("BMR fit residuals (inch)     =",array2str(np.sqrt(dist2_inch[valid_bmr])))
    #print("BMR fit residuals (mm)       =",array2str(np.sqrt(dist2_inch[valid_bmr])*inch2mm))

    rms_inch   = np.sqrt(np.mean(dist2_inch[valid_bmr]))
    rms_mm = rms_inch*inch2mm
    if rms_mm > 1 :
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR fit rms = {:.3f} mm !!!".format(rms_mm))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("=================================================")

    # save moves as csv table
    try :
        if outfile is not None :
            # will survive an import error
            from astropy.table import Table

            movetable=Table()
            movetable["PETAL"]=[petal]
            for name in moves.keys() :
                movetable[name]=[moves[name]]
            #print("Move table")
            #print(movetable)
            movetable.write(outfile,overwrite=True)
            print("wrote move table in",outfile)
            #print("=================================================")
    except Exception as e :
        print(e)
        pass


    sys.stdout.flush()


    if rail_xyz_cs5_inch is not None :
        print("=================================================")
        print("Upper PMA upper rail alignment")

        bmr_xyz_cs5_inch  = target_bmr_CS5_inch

        transfo = adjust_rail_axis(petal,carriage_z,bmr_xyz_cs5_inch,rail_xyz_cs5_inch)

        modified_upper_struts_plateform_xyz = transfo.apply(initial_upper_struts_plateform_xyz)
        modified_upper_struts_length_inch = np.sqrt(np.sum((modified_upper_struts_plateform_xyz - upper_struts_base_xyz)**2,axis=0))
        upper_strut_deltas_inch = modified_upper_struts_length_inch - initial_upper_struts_length_inch
        print("Initial upper_struts length (mm) =",array2str(initial_upper_struts_length_inch*inch2mm))
        print("New upper_struts length (mm)     =",array2str(modified_upper_struts_length_inch*inch2mm))
        print("Upper struts deltas (FOR RAIL ALIGNMENT ONLY) (mm):")
        for s,d in enumerate(upper_strut_deltas_inch) :
            print("  {} {:+.3f}".format(upper_struts_labels[s],d*inch2mm))
        print("This change of struts was FOR RAIL ALIGNMENT ONLY.\nYou would need to add the two sets of deltas to do\nboth corrections at once\n(BMR coordinates and rail alignment).")
        print("=================================================")



    if plot :
        plt.figure("CS5")
        xyz    = measured_bmr_CS5_inch*inch2mm
        for b in range(measured_bmr_CS5_inch.shape[1]) :
            if not valid_bmr[b]:
                #print("ball {} has no valid data".format(bmr_labels[b]))
                continue
            plt.plot(xyz[0,b],xyz[1,b],"x",color="k",alpha=0.5)
            plt.text(xyz[0,b]+10,xyz[1,b]+10,bmr_labels[b],color="k")

        x_cs5=[]
        y_cs5=[]
        z_cs5=[]
        x_cs5.append(0)
        y_cs5.append(0)
        z_cs5.append(0)
        radius=410.
        theta=2*np.pi/10.*np.linspace(petal-0.5,petal+0.5,10)
        x_cs5.append(radius*np.sin(theta))
        y_cs5.append(-radius*np.cos(theta))
        z_cs5.append(0)
        x_cs5.append(0)
        y_cs5.append(0)
        z_cs5.append(0)
        plt.plot(np.hstack(x_cs5),np.hstack(y_cs5),color="gray")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel(r"$x_{CS5}$ (mm)")
        plt.ylabel(r"$y_{CS5}$ (mm)")
        plt.grid()

        if 1 :
            from mpl_toolkits import mplot3d
            plt.figure("3D")
            ax = plt.axes(projection='3d')

            def xyz2plot(xyz) :
                res=np.zeros(xyz.shape)
                res[0]=-xyz[0] # -x
                res[1]=xyz[2] # z
                res[2]=xyz[1] # y
                return res

            if 1 : # upper struts
                xyz1=xyz2plot(upper_struts_base_xyz)
                xyz2=xyz2plot(initial_upper_struts_plateform_xyz)
                for s in range(6) :

                    ax.plot3D([xyz1[0,s],xyz2[0,s]],
                              [xyz1[1,s],xyz2[1,s]],
                              [xyz1[2,s],xyz2[2,s]],
                          color="red")
                    ax.text3D(xyz1[0,s],xyz1[1,s],xyz1[2,s]," "+upper_struts_labels[s],color="red")

            if 1 : # lower struts
                xyz1=xyz2plot(lower_struts_base_xyz)
                xyz2=xyz2plot(initial_lower_struts_plateform_xyz)
                for s in range(6) :
                    ax.plot3D([xyz1[0,s],xyz2[0,s]],
                              [xyz1[1,s],xyz2[1,s]],
                              [xyz1[2,s],xyz2[2,s]],
                          color="brown")
                    ax.text3D(xyz1[0,s],xyz1[1,s],xyz1[2,s]," "+lower_struts_labels[s],color="brown")

            if 1 : # bmr
                xyz=xyz2plot(measured_bmr_PMA_inch)
                ax.plot3D(xyz[0,valid_bmr],xyz[1,valid_bmr],xyz[2,valid_bmr],"x",color="k",label="measured BMR")
                xyz=xyz2plot(target_bmr_PMA_inch)
                ax.scatter3D(xyz[0],xyz[1],xyz[2],color="C0",label="target BMR")
                for b in range(4) :
                    ax.text3D(xyz[0,b],xyz[1,b],xyz[2,b],bmr_labels[b],color="C0")

            if 1 : # focal plane
                t=np.linspace(0,2*np.pi,100)
                rad=410./inch2mm # inch
                x_cs5 = rad*np.cos(t)
                y_cs5 = rad*np.sin(t)
                z_cs5 = np.zeros(t.shape)
                xyz = xyz2plot(CS5_to_PMA_inch(np.array([x_cs5,y_cs5,z_cs5]),carriage_z = carriage_z))
                ax.plot3D(xyz[0],xyz[1],xyz[2],color="gray",alpha=0.5)

            if 1 : # CS5 coordinate system
                xyz = xyz2plot(CS5_to_PMA_inch(np.array([[0,0],[0,1.2*rad],[0,0]]),carriage_z = carriage_z))
                ax.plot3D(xyz[0],xyz[1],xyz[2],"--",color="gray",alpha=0.5)
                ax.text3D(xyz[0,1],xyz[1,1],xyz[2,1],r"$y_{CS5}$",color="gray")

                xyz = xyz2plot(CS5_to_PMA_inch(np.array([[0,1.2*rad],[0,0],[0,0]]),carriage_z = carriage_z))
                ax.plot3D(xyz[0],xyz[1],xyz[2],"--",color="gray",alpha=0.5)
                ax.text3D(xyz[0,1],xyz[1,1],xyz[2,1],r"$x_{CS5}$",color="gray")

            if 1 : # petal
                t=2*np.pi/10.*np.linspace(petal-0.5,petal+0.5,10)
                x_cs5 = np.hstack([0,rad*np.sin(t),0])
                y_cs5 = np.hstack([0,-rad*np.cos(t),0])
                z_cs5 = np.zeros(x_cs5.shape)
                xyz = xyz2plot(CS5_to_PMA_inch(np.array([x_cs5,y_cs5,z_cs5]),carriage_z = carriage_z))
                ax.plot3D(xyz[0],xyz[1],xyz[2],color="green",label="petal {}".format(petal))

            ax.set_xlabel(r'$-x_{PMA}$')
            ax.set_ylabel(r'$z_{PMA}$')
            ax.set_zlabel(r'$y_{PMA}$')
            ax.legend()

        sys.stdout.flush()
        plt.show()



if __name__ == '__main__':
    main()
