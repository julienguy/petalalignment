#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

######################################################################
## Inputs
######################################################################
# Petal Position (Per DESI-3596)
PetalPosition = 0

# Input BMR Locations (in CS5, will change input method later)
measured_bmrs_CS5 = np.zeros((3,4))
measured_bmrs_CS5[:,0] = [3.3247, -16.5683, -55.9022]
measured_bmrs_CS5[:,1] = [1.7563, -18.9263, -55.9158]
measured_bmrs_CS5[:,2] = [-3.5547, -20.3330, -55.9186]
measured_bmrs_CS5[:,3] = [-3.1565, -16.5614, -55.9040]

# Options
debug = False
plot  = True

######################################################################
## Known measurements / metrology
######################################################################
# BMR (Ball Mount Refectors) Target Locations
# when mounted on petal (for petal insertion)
# in Guide Spike 1 CS, from metrology:
target_bmrs_GS1CS = np.zeros((3,4))
target_bmrs_GS1CS[:,0]=[5.8764, 0.0842, 1.5300]
target_bmrs_GS1CS[:,1]=[4.3081, -2.2738, 1.5125]
target_bmrs_GS1CS[:,2]=[-1.0028, -3.6804, 1.5001]
target_bmrs_GS1CS[:,3]=[-0.6047, 0.0911, 1.5174]

# Reference coordinates of the base ends of the PMA struts in the PMA CS
struts_base_coords     = np.array([[-26.875, -26.875, 26.875, 26.875, 15.475, 31.25],
                                   [-43.811, -42.561, -43.811, -42.561, -42.561, -43.811],
                                   [ -6.76, 8.49, -6.76, 8.49, 35.397, 35.397]])

# Reference coordinates of the platform ends of the PMA struts in the PMA CS
struts_platform_coords = np.array([[-26.875, -26.875, 26.875, 26.875, 26.475, 31.25],
                                   [-32.811, -42.561, -32.811, -42.561, -42.561, -32.811],
                                   [ -6.76, -2.51, -6.76, -2.51, 35.397, 35.397]])
######################################################################


def rotmat3d(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if s == 0:
        rotation_matrix = np.eye(3)
    else:
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def CS5_to_PMA(xyz) :
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

def GS1_to_Petal(xyz) :
    """
    Transform coordinates from GS1 (Guide Spike 1) to Petal.
    It's a pure translation
    """

    # coordinate of GS1 in petal CS
    GS1_Petal = np.array([-429 * np.sin(np.radians(9)),
                              -429 * np.cos(np.radians(9)),
                              -57.4351 * 25.4]) / 25.4 # inches
    if len(xyz.shape)==1 :
        return xyz + GS1_Petal
    else :
        return xyz + GS1_Petal[:,None]

def Petal_to_CS5(xyz,PetalPosition) :
    """
    Transform coordinates from Petal to CS5
    It's a pure rotation
    """
    # Calculate the target locations based on the nominal petal position in the FPA
    PetalPositionTransform = np.array([[np.cos(np.radians(PetalPosition * 36)), -np.sin(np.radians(PetalPosition * 36)), 0],
                                       [np.sin(np.radians(PetalPosition * 36)), np.cos(np.radians(PetalPosition * 36)), 0],
                                       [0, 0, 1]])
    return PetalPositionTransform.dot(xyz)

def GS1_to_PMA(xyz,petal) :
    """
    Transform coordinates from GS1 (Guide Spike 1) to PMA
    """
    return CS5_to_PMA(Petal_to_CS5(GS1_to_Petal(xyz),petal))


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

def correction_error(mat, inputs, targets):
    '''
    Function to minimize when best fitting the transform matrix between input
    BMR locations and target BMR locations.

    mat: 1x6 Numpy array, with the first 3 values being a target rotation vector
    and the last 3 values a translation vector

    inputs: 3xN Numpy array containing measured BMR locations, where N is the
    number of BMRs

    targets: 3xN Numpy array containing target BMR locations
    '''

    startvec = np.array([0, 0, 1])
    targetvec = mat[0:3]
    targettrans = mat[3:6]

    bmrcount = inputs.shape[1]

    targettrans = np.tile(targettrans,(bmrcount, 1)).T

    return np.sum(pow(np.dot(rotmat3d(startvec, targetvec), inputs) + targettrans - targets, 2))

## transform
target_bmrs_PMA   = GS1_to_PMA(target_bmrs_GS1CS,PetalPosition)
measured_bmrs_PMA = CS5_to_PMA(measured_bmrs_CS5)

# Get and remove mean Z values of BMR coords:
meanz = np.mean(target_bmrs_PMA[2])
target_bmrs_PMA_zm = target_bmrs_PMA.copy()
target_bmrs_PMA_zm[2] -= meanz
# Get and remove mean Z values of BMR coords:
measured_bmrs_PMA_zm = measured_bmrs_PMA.copy()
measured_bmrs_PMA_zm[2] -= np.mean(measured_bmrs_PMA[2])

# Generate a best fit rotation matrix and translation vector
fits = opt.minimize(correction_error, [0, 0, 1, 0, 0, 0],
                               args = (measured_bmrs_PMA_zm, target_bmrs_PMA_zm))
fitmat = rotmat3d([0,0,1],fits.x[0:3])
trans = fits.x[3:6]

if debug :
    target_bmrs_CS5_oGS1 = Petal_to_CS5(target_bmrs_GS1CS,PetalPosition) # for debug only
    print("DEBUG target_bmrs_GS1CS[:,0]    =",target_bmrs_GS1CS[:,0])
    print("DEBUG target_bmrs_CS5_oGS1[:,0] =",target_bmrs_CS5_oGS1[:,0])
    print("DEBUG target_bmrs_CS5[:,0]      =",target_bmrs_CS5[:,0])
    print("DEBUG target_bmrs_PMA[:,0]        =",target_bmrs_PMA[:,0])
    print("DEBUG target_bmrs_PMA[:,3]        =",target_bmrs_PMA[:,3])
    print("DEBUG measured_bmrs_PMA[:,0]      =",measured_bmrs_PMA[:,0])
    print("DEBUG measured_bmrs_PMA[:,3]      =",measured_bmrs_PMA[:,3])
    print("DEBUG target_bmrs_PMA_zm[:,0]        =",target_bmrs_PMA_zm[:,0])
    print("DEBUG target_bmrs_PMA_zm[:,3]        =",target_bmrs_PMA_zm[:,3])
    print("DEBUG measured_bmrs_PMA_zm[:,0]        =",measured_bmrs_PMA_zm[:,0])
    print("DEBUG measured_bmrs_PMA_zm[:,3]        =",measured_bmrs_PMA_zm[:,3])
    print("DEBUG x=",fits.x)
    print("DEBUG trans=",trans)

# Put the PMA Platform and Bases in same CS
struts_base_coords_zm     = struts_base_coords.copy()
struts_platform_coords_zm = struts_platform_coords.copy()
struts_base_coords_zm[2]     -= meanz
struts_platform_coords_zm[2] -= meanz


# Generate Strut Lengths
strut_lengths = np.linalg.norm(struts_base_coords_zm - (np.dot(fitmat, struts_platform_coords_zm) + np.tile(trans, (6, 1)).T), axis = 0)

# Generate Strut Deltas
strut_deltas = strut_lengths - 11 #?

print("struts deltas =",strut_deltas)

if debug :
    print("residuals =",np.dot(fitmat, measured_bmrs_PMA_zm) + np.tile(trans, (4, 1)).T - target_bmrs_PMA_zm)

if plot :
    from mpl_toolkits import mplot3d
    fig = plt.figure("PMA")
    ax = plt.axes(projection='3d')


    def xyz2plot(xyz) :
        res=np.zeros(xyz.shape)
        res[0]=-xyz[0] # -x
        res[1]=xyz[2] # z
        res[2]=xyz[1] # y
        return res


    xyz_gs1=CS5_to_PMA(Petal_to_CS5(GS1_to_Petal(np.array([0,0,0])),PetalPosition))
    xyz=xyz2plot(xyz_gs1)
    ax.scatter3D(xyz[0],xyz[1],xyz[2],color="k",label="Guide Spike 1")

    #ax.scatter3D(xplotsign*gs1_PMA[xplot],gs1_PMA[yplot],gs1_PMA[zplot],color="blue",label="GS1")
    #ax.scatter3D(gs2_PMA[xplot],gs2_PMA[yplot],gs2_PMA[zplot],color="blue",label="GS2")

    label='struts'

    xyz1=xyz2plot(struts_base_coords)
    xyz2=xyz2plot(struts_platform_coords)
    for s in range(6) :
        ax.plot3D([xyz1[0,s],xyz2[0,s]],
                  [xyz1[1,s],xyz2[1,s]],
                  [xyz1[2,s],xyz2[2,s]],
                  color="red",label=label)
        label=None
    xyz=xyz2plot(measured_bmrs_PMA)
    ax.scatter3D(xyz[0],xyz[1],xyz[2],color="green",label="measured BMR")
    xyz=xyz2plot(target_bmrs_PMA)
    ax.scatter3D(xyz[0],xyz[1],xyz[2],color="blue",label="target BMR")

    t=np.linspace(0,2*np.pi,100)
    rad=410./25.4 # inch
    x_cs5 = rad*np.cos(t)
    y_cs5 = rad*np.sin(t)
    z_cs5 = np.zeros(t.shape)
    xyz = xyz2plot(CS5_to_PMA(np.array([x_cs5,y_cs5,z_cs5])))
    ax.plot3D(xyz[0],xyz[1],xyz[2],color="gray")

    x_cs5=[]
    y_cs5=[]
    z_cs5=[]
    x_cs5.append(rad*np.sin(2*np.pi/10.*(PetalPosition-0.5)))
    y_cs5.append(-rad*np.cos(2*np.pi/10.*(PetalPosition-0.5)))
    z_cs5.append(0)
    x_cs5.append(0)
    y_cs5.append(0)
    z_cs5.append(0)
    x_cs5.append(rad*np.sin(2*np.pi/10.*(PetalPosition+0.5)))
    y_cs5.append(-rad*np.cos(2*np.pi/10.*(PetalPosition+0.5)))
    z_cs5.append(0)
    xyz = xyz2plot(CS5_to_PMA(np.array([x_cs5,y_cs5,z_cs5])))
    ax.plot3D(xyz[0],xyz[1],xyz[2],color="gray")

    xyz = xyz2plot(CS5_to_PMA(np.array([[0,0],[0,-rad],[0,0]])))
    ax.plot3D(xyz[0],xyz[1],xyz[2],"--",color="gray",label="-y_CS5")

    #xyz_PMA = np.dot(CS5toPMA, xyz_cs5) + CS5toPMA_translate[:,None]
    #ax.plot3D(xplotsign*xyz_PMA[xplot],xyz_PMA[yplot],xyz_PMA[zplot],"--",color="gray",label="z_CS5=0")



    ax.set_xlabel('-x_PMA')
    ax.set_ylabel('z_PMA')
    ax.set_zlabel('y_PMA')
    ax.legend()
    plt.show()
