#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import yaml

from transfo import Transfo3D as Transfo

inch2mm = 25.4
######################################################################
plot  = True
debug = False
######################################################################

def CS5_to_PMA_inch(xyz) :
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

    CS5toPMA_translate = np.array([0, 0, 115.8456])

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

def compute_target_bmr_guide_spikes_coords_mm(petal) :
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
    return bmr_CS5_mm

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

def compute_target_bmr_light_weight_red_leg_coords_mm(petal) :

    red_leg_mount_holes_CS5_mm  = get_red_leg_mount_holes_in_CS5_mm(petal)
    red_leg_mount_holes_6206_mm = get_red_leg_mount_holes_in_6206_mm()
    # set same z
    red_leg_mount_holes_6206_mm[2] += np.mean(red_leg_mount_holes_CS5_mm[2]-red_leg_mount_holes_6206_mm[2])
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

    xyz_6206 = np.array([ [0,0,0] , [-50,0,0] ]).T
    xyz_6207 = np.array([ [0,0,0] , [50,0,0] ]).T
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
        plt.plot(bmr_CS5_mm[0],bmr_CS5_mm[1],"o",markersize=12,label="bmr")
        #for b in range(bmr_CS5_mm.shape[1]) :
        #    plt.text(bmr_CS5_mm[0,b],bmr_CS5_mm[1,b],str(b+1))
        xyz_CS5 = C6206_to_CS5.apply(xyz_6206)
        plt.plot(xyz_CS5[0,0],xyz_CS5[1,0],"X",color="gray",label="A")
        plt.plot(xyz_CS5[0,1],xyz_CS5[1,1],".",color="gray")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid()
        plt.xlabel("X_CS5 (mm)")
        plt.ylabel("Y_CS5 (mm)")

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

    return bmr_CS5_mm

def compute_target_bmr_heavy_weight_red_leg_coords_mm(petal) :

    red_leg_mount_holes_CS5_mm  = get_red_leg_mount_holes_in_CS5_mm(petal)
    red_leg_mount_holes_6206_mm = get_red_leg_mount_holes_in_6206_mm()
    red_leg_mount_holes_6206_mm[2] += np.mean(red_leg_mount_holes_CS5_mm[2]-red_leg_mount_holes_6206_mm[2])
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

    xyz_6206 = np.array([ [0,0,0] , [-50,0,0] ]).T
    xyz_6211 = np.array([ [0,50,0] , [0,0,0] ]).T
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
        plt.plot(bmr_CS5_mm[0],bmr_CS5_mm[1],"o",markersize=12,label="bmr")
        xyz_CS5 = C6206_to_CS5.apply(xyz_6206)
        plt.plot(xyz_CS5[0,0],xyz_CS5[1,0],"X",color="gray",label="A")
        plt.plot(xyz_CS5[0,1],xyz_CS5[1,1],".",color="gray")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid()
        plt.xlabel("X_CS5 (mm)")
        plt.ylabel("Y_CS5 (mm)")

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

    return bmr_CS5_mm


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


    # Compute target bmr coords in CS5 from metrology
    #################################################
    if inputs["bmr_type"] == "guide_spikes" :
        target_bmr_CS5_mm  = compute_target_bmr_guide_spikes_coords_mm(petal)
    elif inputs["bmr_type"]== "light_weight_red_leg" :
        target_bmr_CS5_mm = compute_target_bmr_light_weight_red_leg_coords_mm(petal)
    elif inputs["bmr_type"]== "heavy_weight_red_leg" :
        target_bmr_CS5_mm = compute_target_bmr_heavy_weight_red_leg_coords_mm(petal)
    else :
        print('error {} not in ["guide_spikes","light_weight_red_leg","heavy_weight_red_leg"]'.format(inputs["bmr_type"]))
        sys.exit(2)
    target_bmr_CS5_inch = target_bmr_CS5_mm/inch2mm

    #################################################


    # Input BMR Locations (in CS5, will change input method later)
    #################################################

    if inputs["bmr_type"] == "heavy_weight_red_leg" :
        bmr_labels=["B1","B2","B3","B4","B5"]
    else :
        bmr_labels=["B1","B2","B3","B4"]

    number_of_balls=len(bmr_labels)
    measured_bmr_CS5_inch = np.zeros((3,number_of_balls))

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
        print("Apply correction for PMA misalignment of {:.1f} degrees of dx,dy= {:+.3f},{:+.3f} inch".format(angle_deg,
                                                                                                                   correction_when_retracting_pma_inch[0],correction_when_retracting_pma_inch[1]))
        if len(target_bmr_CS5_inch.shape) == 1 :
            target_bmr_CS5_inch[0:2] += correction_when_retracting_pma_inch[0:2] # change only x and y
        else :
            target_bmr_CS5_inch[0:2] += correction_when_retracting_pma_inch[0:2][:,None]

    ##############################################################


    target_bmr_PMA_inch   = CS5_to_PMA_inch(target_bmr_CS5_inch)
    measured_bmr_PMA_inch = CS5_to_PMA_inch(measured_bmr_CS5_inch)


    # struts labels
    struts_labels = ["S1","S2","S3","S4","S5","S6"]

    # Reference coordinates of the base ends of the PMA struts in the PMA CS
    struts_base_coords_inch     = np.array([[-26.875, -26.875, 26.875, 26.875, 15.475, 31.25],
                                       [-43.811, -42.561, -43.811, -42.561, -42.561, -43.811],
                                       [ -6.76, 8.49, -6.76, 8.49, 35.397, 35.397]])

    # Reference coordinates of the platform ends of the PMA struts in the PMA CS
    initial_struts_platform_coords_inch = np.array([[-26.875, -26.875, 26.875, 26.875, 26.475, 31.25],
                                                    [-32.811, -42.561, -32.811, -42.561, -42.561, -32.811],
                                                    [ -6.76, -2.51, -6.76, -2.51, 35.397, 35.397]])

    initial_struts_length_inch = np.sqrt(np.sum((initial_struts_platform_coords_inch - struts_base_coords_inch)**2,axis=0))

    print("=================================================")

    if True :
        # Ignore difference in z
        measured_bmr_PMA_inch[2,valid_bmr] += ( np.mean(target_bmr_PMA_inch[2,valid_bmr]) - np.mean(measured_bmr_PMA_inch[2,valid_bmr]) )

    delta_inch_CS5 = (measured_bmr_CS5_inch-target_bmr_CS5_inch)
    dr_inch_CS5 = np.sqrt(delta_inch_CS5[0]**2+delta_inch_CS5[1]**2)
    #print("BMR offsets (sqrt(dx2+dy2), inch) =",array2str(dr_inch_CS5[valid_bmr]))
    #print("BMR mean offset dx (CS5) = {:+.3f} inch".format(np.mean(delta_inch_CS5[0][valid_bmr])))
    #print("BMR mean offset dy (CS5)  = {:+.3f} inch".format(np.mean(delta_inch_CS5[1][valid_bmr])))

    print("BMR offsets (sqrt(dx2+dy2), mm)   =",array2str(dr_inch_CS5[valid_bmr]*inch2mm))
    print("BMR mean offset dx (CS5)  = {:+.3f} mm".format(np.mean(delta_inch_CS5[0][valid_bmr])*inch2mm))
    print("BMR mean offset dy (CS5)  = {:+.3f} mm".format(np.mean(delta_inch_CS5[1][valid_bmr])*inch2mm))

    delta_inch_PMA = (measured_bmr_PMA_inch-target_bmr_PMA_inch)
    print("BMR mean offset dx (PMA)  = {:+.3f} mm".format(np.mean(delta_inch_PMA[0][valid_bmr])*inch2mm))
    print("BMR mean offset dy (PMA)  = {:+.3f} mm".format(np.mean(delta_inch_PMA[1][valid_bmr])*inch2mm))
    #print("BMR mean offset dx (PMA) = {:+.3f} inch".format(np.mean(delta_inch[0][valid_bmr])))
    #print("BMR mean offset dy (PMA)  = {:+.3f} inch".format(np.mean(delta_inch[1][valid_bmr])))
    print("=================================================")

    pma_adjust = Transfo()
    pma_adjust.fit(measured_bmr_PMA_inch[:,valid_bmr],target_bmr_PMA_inch[:,valid_bmr])


    new_struts_platform_coords_inch = pma_adjust.apply(initial_struts_platform_coords_inch)

    new_struts_length_inch = np.sqrt(np.sum((new_struts_platform_coords_inch - struts_base_coords_inch)**2,axis=0))
    print("Initial struts length (inch) =",array2str(initial_struts_length_inch))
    print("New struts length (inch)     =",array2str(new_struts_length_inch))
    strut_deltas_inch = new_struts_length_inch - initial_struts_length_inch
    print("Struts deltas (inch):")
    for s,d in enumerate(strut_deltas_inch) :
        print("  {} {:+.3f}".format(struts_labels[s],d))

    print("Struts deltas (mm):")
    for s,d in enumerate(strut_deltas_inch) :
        print("  {} {:+.3f}".format(struts_labels[s],d*inch2mm))

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
    else :
        print("(fit rms = {:.3f} mm)".format(rms_mm))
    print("=================================================")

    if plot :
        plt.figure("CS5")


        xyz    = measured_bmr_CS5_inch*inch2mm
        for b in range(measured_bmr_CS5_inch.shape[1]) :
            if not valid_bmr[b]:
                #print("ball {} has no valid data".format(bmr_labels[b]))
                continue
            plt.plot(xyz[0,b],xyz[1,b],"o",color="k",alpha=0.5)
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
        plt.xlabel("X_CS5 (mm)")
        plt.ylabel("Y_CS5 (mm)")
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

            # struts
            label='struts'
            xyz1=xyz2plot(struts_base_coords_inch)
            xyz2=xyz2plot(initial_struts_platform_coords_inch)
            for s in range(6) :
                ax.plot3D([xyz1[0,s],xyz2[0,s]],
                          [xyz1[1,s],xyz2[1,s]],
                          [xyz1[2,s],xyz2[2,s]],
                      color="red")
                ax.text3D(xyz1[0,s],xyz1[1,s],xyz1[2,s],struts_labels[s],color="red")
                label=None

            # bmr
            xyz=xyz2plot(measured_bmr_PMA_inch)
            ax.scatter3D(xyz[0,valid_bmr],xyz[1,valid_bmr],xyz[2,valid_bmr],color="green",label="measured BMR")
            xyz=xyz2plot(target_bmr_PMA_inch)
            ax.scatter3D(xyz[0],xyz[1],xyz[2],color="blue",label="target BMR")
            for b in range(4) :
                ax.text3D(xyz[0,b],xyz[1,b],xyz[2,b],bmr_labels[b],color="blue")

            # focal plane
            t=np.linspace(0,2*np.pi,100)
            rad=410./inch2mm # inch
            x_cs5 = rad*np.cos(t)
            y_cs5 = rad*np.sin(t)
            z_cs5 = np.zeros(t.shape)
            xyz = xyz2plot(CS5_to_PMA_inch(np.array([x_cs5,y_cs5,z_cs5])))
            ax.plot3D(xyz[0],xyz[1],xyz[2],color="gray")

            x_cs5=[]
            y_cs5=[]
            z_cs5=[]
            x_cs5.append(rad*np.sin(2*np.pi/10.*(petal-0.5)))
            y_cs5.append(-rad*np.cos(2*np.pi/10.*(petal-0.5)))
            z_cs5.append(0)
            x_cs5.append(0)
            y_cs5.append(0)
            z_cs5.append(0)
            x_cs5.append(rad*np.sin(2*np.pi/10.*(petal+0.5)))
            y_cs5.append(-rad*np.cos(2*np.pi/10.*(petal+0.5)))
            z_cs5.append(0)
            xyz = xyz2plot(CS5_to_PMA_inch(np.array([x_cs5,y_cs5,z_cs5])))
            ax.plot3D(xyz[0],xyz[1],xyz[2],color="gray")

            xyz = xyz2plot(CS5_to_PMA_inch(np.array([[0,0],[0,-rad],[0,0]])))
            ax.plot3D(xyz[0],xyz[1],xyz[2],"--",color="gray",label="-y_CS5")
            ax.set_xlabel('-x_PMA')
            ax.set_ylabel('z_PMA')
            ax.set_zlabel('y_PMA')
            ax.legend()

        plt.show()



if __name__ == '__main__':
    main()
