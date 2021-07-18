# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 19:34:48 2021

@author: Van
"""

import sys
import numpy as np
import pandas as pd
import math
import scipy.optimize as opt
import matplotlib.pyplot as plt
#import win32clipboard

## Inputs

# Petal Position (Per DESI-3596)
PetalPosition = 0

# Input BMR Locations (in CS5, will change input method later)
bmr1_in = np.array([3.3247, -16.5683, -55.9022])
bmr2_in = np.array([1.7563, -18.9263, -55.9158])
bmr3_in = np.array([-3.5547, -20.3330, -55.9186])
bmr4_in = np.array([-3.1565, -16.5614, -55.9040])

# Functions

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
    win32clipboard.OpenClipboard()
    out = parsearray(win32clipboard.GetClipboardData())
    win32clipboard.CloseClipboard()
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

## Known measurements

# BMR Target Locations (Guide Spike 1 CS, from metrology):
bmr1_GS1CS = np.array([5.8764, 0.0842, 1.5300])
bmr2_GS1CS = np.array([4.3081, -2.2738, 1.5125])
bmr3_GS1CS = np.array([-1.0028, -3.6804, 1.5001])
bmr4_GS1CS = np.array([-0.6047, 0.0911, 1.5174])

# GS2 location in GS1 CS (from model)
gs2_GS1CS = np.array([5.2843, 0, 0])

# Angle between CS5 and PMA CS
CS5toPMA_Clocking = -134.75 * np.pi / 180

# Arrays to convert CS5 coords to PMA CS coords
CS5toPMA = np.array([[np.cos(CS5toPMA_Clocking), -np.sin(CS5toPMA_Clocking), 0],
                         [np.sin(CS5toPMA_Clocking), np.cos(CS5toPMA_Clocking), 0],
                         [0, 0, 1]])

CS5toPMA_translate = np.array([0, 0, 115.8456])

CS5toPMA2D = CS5toPMA[0:2,0:2]

# Calculate the target locations based on the nominal petal position in the FPA
PetalPositionTransform = np.array([[np.cos(np.radians(PetalPosition * 36)), -np.sin(np.radians(PetalPosition * 36)), 0],
                                   [np.sin(np.radians(PetalPosition * 36)), np.cos(np.radians(PetalPosition * 36)), 0],
                                   [0, 0, 1]])

PetalPositionTransform2D = PetalPositionTransform[0:2,0:2]

# Reference coordinates of the base ends of the PMA struts in the PMA CS
base_ref_coords = np.array([[-26.875, -26.875, 26.875, 26.875, 15.475, 31.25],
                            [-43.811, -42.561, -43.811, -42.561, -42.561, -43.811],
                            [ -6.76, 8.49, -6.76, 8.49, 35.397, 35.397]])

# Reference coordinates of the platform ends of the PMA struts in the PMA CS
platform_ref_coords = np.array([[-26.875, -26.875, 26.875, 26.875, 26.475, 31.25],
                                [-32.811, -42.561, -32.811, -42.561, -42.561, -32.811],
                                [ -6.76, -2.51, -6.76, -2.51, 35.397, 35.397]])


## Calcs

# Primary Guide Spike Coordinates (CS5):
gs1_CS5 = np.array([-429 * np.sin(np.radians(9 - (PetalPosition * 36))),
                      -429 * np.cos(np.radians(9 - (PetalPosition * 36))),
                      -57.4351 * 25.4])

gs2_CS5 = np.array([-429 * np.sin(np.radians(-9 - (PetalPosition * 36))),
                      -429 * np.cos(np.radians(-9 - (PetalPosition * 36))),
                      -57.4351 * 25.4])

# Convert to inches:
gs1_CS5 = gs1_CS5 / 25.4
gs2_CS5 = gs2_CS5 / 25.4


# Convert to PMA CS
gs1_PMA = np.dot(CS5toPMA, gs1_CS5) + CS5toPMA_translate
gs2_PMA = np.dot(CS5toPMA, gs2_CS5) + CS5toPMA_translate

# Rotate so axes are parallel to CS5 with origin at GS1
bmr1_CS5_oGS1 = np.dot(PetalPositionTransform, bmr1_GS1CS)
bmr2_CS5_oGS1 = np.dot(PetalPositionTransform, bmr2_GS1CS)
bmr3_CS5_oGS1 = np.dot(PetalPositionTransform, bmr3_GS1CS)
bmr4_CS5_oGS1 = np.dot(PetalPositionTransform, bmr4_GS1CS)

gs2_CS5_oGS1 = np.dot(PetalPositionTransform, gs2_GS1CS)

# Rotate so axes are parallel to PMA CS with origin at GS1 (for debugging)
bmr1_PMA_oGS1 = np.dot(CS5toPMA, bmr1_CS5_oGS1)
bmr2_PMA_oGS1 = np.dot(CS5toPMA, bmr2_CS5_oGS1)
bmr3_PMA_oGS1 = np.dot(CS5toPMA, bmr3_CS5_oGS1)
bmr4_PMA_oGS1 = np.dot(CS5toPMA, bmr4_CS5_oGS1)

gs2_PMA_oGS1 = np.dot(CS5toPMA, gs2_CS5_oGS1)

# Translate into CS5
bmr1_CS5 = bmr1_CS5_oGS1 + gs1_CS5
bmr2_CS5 = bmr2_CS5_oGS1 + gs1_CS5
bmr3_CS5 = bmr3_CS5_oGS1 + gs1_CS5
bmr4_CS5 = bmr4_CS5_oGS1 + gs1_CS5

# Translate and Rotate to PMA CS
bmr1_PMA = np.dot(CS5toPMA, bmr1_CS5) + CS5toPMA_translate
bmr2_PMA = np.dot(CS5toPMA, bmr2_CS5) + CS5toPMA_translate
bmr3_PMA = np.dot(CS5toPMA, bmr3_CS5) + CS5toPMA_translate
bmr4_PMA = np.dot(CS5toPMA, bmr4_CS5) + CS5toPMA_translate


# Get mean Z value of BMR coords:
bmr_PMA_Zmean = np.mean([bmr1_PMA[2], bmr2_PMA[2], bmr3_PMA[2], bmr4_PMA[2]])
bmr_PMA_Zmean_Mat = np.array([0, 0, bmr_PMA_Zmean])

# Translate BMR coords to eliminate Z offset
bmr1_PMA_ZM = bmr1_PMA - bmr_PMA_Zmean_Mat
bmr2_PMA_ZM = bmr2_PMA - bmr_PMA_Zmean_Mat
bmr3_PMA_ZM = bmr3_PMA - bmr_PMA_Zmean_Mat
bmr4_PMA_ZM = bmr4_PMA - bmr_PMA_Zmean_Mat


# Translate and rotate input BMR locations into PMA CS
bmr1_in_PMA = np.dot(CS5toPMA, bmr1_in) + CS5toPMA_translate
bmr2_in_PMA = np.dot(CS5toPMA, bmr2_in) + CS5toPMA_translate
bmr3_in_PMA = np.dot(CS5toPMA, bmr3_in) + CS5toPMA_translate
bmr4_in_PMA = np.dot(CS5toPMA, bmr4_in) + CS5toPMA_translate

# Get mean Z value of input BMR coords:
bmr_in_PMA_Zmean = np.mean([bmr1_in_PMA[2], bmr2_in_PMA[2], bmr3_in_PMA[2], bmr4_in_PMA[2]])
bmr_in_PMA_Zmean_Mat = np.array([0, 0, bmr_in_PMA_Zmean])

# Translate input BMR coords to eliminate Z offset
bmr1_in_PMA_ZM = bmr1_in_PMA - bmr_in_PMA_Zmean_Mat
bmr2_in_PMA_ZM = bmr2_in_PMA - bmr_in_PMA_Zmean_Mat
bmr3_in_PMA_ZM = bmr3_in_PMA - bmr_in_PMA_Zmean_Mat
bmr4_in_PMA_ZM = bmr4_in_PMA - bmr_in_PMA_Zmean_Mat

# Collect BMR coordinates into complete matrices
bmr_Targets = np.array([bmr1_PMA_ZM, bmr2_PMA_ZM, bmr3_PMA_ZM, bmr4_PMA_ZM]).T
bmr_Inputs = np.array([bmr1_in_PMA_ZM, bmr2_in_PMA_ZM, bmr3_in_PMA_ZM, bmr4_in_PMA_ZM]).T

# Generate a best fit rotation matrix and translation vector
fits = opt.minimize(correction_error, [0, 0, 1, 0, 0, 0],
                               args = (bmr_Inputs, bmr_Targets))
fitmat = rotmat3d([0,0,1],fits.x[0:3])
trans = fits.x[3:6]

# Put the PMA Platform and Bases in same CS
base_ref_coords_zm = base_ref_coords - np.tile(bmr_PMA_Zmean_Mat, (6,1)).T
platform_ref_coords_zm = platform_ref_coords - np.tile(bmr_PMA_Zmean_Mat, (6,1)).T

# Generate Strut Lengths
strut_lengths = np.linalg.norm(base_ref_coords_zm - (np.dot(fitmat, platform_ref_coords_zm) + np.tile(trans, (6, 1)).T), axis = 0)

# Generate Strut Deltas
strut_deltas = strut_lengths - 11

print("strut_deltas =",strut_deltas)

# Debug printout
print(np.dot(fitmat, bmr_Inputs) + np.tile(trans, (4, 1)).T - bmr_Targets)

# Debug for Solidworks PMA Math Test
swdebugout = np.flip(np.linalg.norm(base_ref_coords_zm - (np.dot(fitmat, platform_ref_coords_zm) + np.tile(trans, (6, 1)).T), axis = 0)*25.4)
