#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import yaml
from mpl_toolkits import mplot3d

from transfo import Transfo3D,Transfo2D,OldTransfo3D


inch2mm = 25.4
def CS5_to_PMA_sled(xyz) :
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

    # this number matters for the lower struts
    # based on comparison of struts
    CS5toPMA_translate = np.array([0,54.15395102, 55.+4.215])

    res = np.dot(CS5toPMA, xyz)
    if len(res.shape)==1 :
        return res + CS5toPMA_translate
    else :
        return res + CS5toPMA_translate[:,None]

# measurement of axis of rotation of PMA in CS5

add_npoints=4
meas_axis_xyz_cs5=np.zeros((3,2+add_npoints))

# first measurement
#meas_axis_xyz_cs5[:,0]=[0.1688,-0.1481,-101.4190]  # far point
#meas_axis_xyz_cs5[:,1]=[0.1349,-0.1189,-71.9418]   # close point
# second measurement
meas_axis_xyz_cs5[:,0]=[0.0028,0.,-101.507]  # far point
meas_axis_xyz_cs5[:,1]=[-0.0054,0.0041,-71.9508]  # close point
# third measurement
#meas_axis_xyz_cs5[:,0]=[0.0112,0.0036,-101.5149]  # far point
#meas_axis_xyz_cs5[:,1]=[0.0174,-0.0007,-72.6068]  # close point

dist=np.sqrt(np.sum((meas_axis_xyz_cs5[:,0]-meas_axis_xyz_cs5[:,1])[0:2]**2))
print("drift from axis     = {:.5f} inch = {:.5f} mm".format(dist,dist*inch2mm))
print("mis-alignment angle = {:.4f} deg".format(dist/np.abs(meas_axis_xyz_cs5[2,1]-meas_axis_xyz_cs5[2,0])*180/np.pi))

# target (inc)
target_axis_xyz_cs5=meas_axis_xyz_cs5.copy()
target_axis_xyz_cs5[0,:]=0
target_axis_xyz_cs5[1,:]=0

if add_npoints>0 :
    # add second axis offset from first one
    offset=0.1
    for p in range(2,2+add_npoints) :
        print("point",p,p%2)
        meas_axis_xyz_cs5[:,p] = meas_axis_xyz_cs5[:,p%2]
        if p>4 : i=1
        else : i=0
        meas_axis_xyz_cs5[i,p] += offset
        target_axis_xyz_cs5[:,p] = target_axis_xyz_cs5[:,p%2]
        target_axis_xyz_cs5[i,p] += offset

print("delta axis cs5=",target_axis_xyz_cs5-meas_axis_xyz_cs5)

# target location of plate (mm)
target_plate_xyz_cs5=np.zeros((3,5))
target_plate_xyz_cs5[:,0]=[1447.77,192.35,-1688.2]
target_plate_xyz_cs5[:,1]=[1447.77,192.35,-2856.6]
target_plate_xyz_cs5[:,2]=[204.97,1446.04,-2856.6]
target_plate_xyz_cs5[:,3]=[204.97,1446.04,-1688.2]
target_plate_xyz_cs5[:,4]=target_plate_xyz_cs5[:,0]
target_plate_xyz_cs5 /= inch2mm # now inch

# sled adjustment code
## https://docs.google.com/spreadsheets/d/1ZiRUVq-BNczgHcma21Go_CwneOJIhhFXROyY054inN4/edit?ts=60ba8dad#gid=17235295

# PMA coordinates of sled struts
tmp=[[29,-29,0,29,-29,27],
     [-27.727,-27.727,-27.727,-6.978,-6.978,-7.512],
     [67.537,67.537,-47.464,14.542,14.542,-49.999],
     [29,-29,0,29,-29,6.614],
     [-7.5,-7.5,-7.5,-9.5,-9.5,-7.5],
     [65,65,-50,-5.687,-5.687,-50]]

tmp=np.array(tmp)
strut_base_xyz=np.zeros((3,6))
strut_plateform_xyz=np.zeros((3,6))
for s in range(6) :
    strut_base_xyz[0:3,s]=tmp[0:3,s]
    strut_plateform_xyz[0:3,s]=tmp[3:6,s]


strut_base_xyz_cs5 = np.zeros(strut_base_xyz.shape)
strut_plateform_xyz_cs5 = np.zeros(strut_base_xyz.shape)
# NEED TO UNDERSTAND THE Z AND Y OFFSETS OF THE PMA

# from solidworks:
strut_base_xyz_cs5[:,0]=[37.734,78.241,8.322]
strut_base_xyz_cs5[:,1]=[78.567,37.050,8.322]
strut_plateform_xyz_cs5[:,0]=[23.369,64.000,5.785]
strut_plateform_xyz_cs5[:,1]=[64.202,22.810,5.785]

strut_base_xyz_bis = CS5_to_PMA_sled(strut_base_xyz_cs5)
strut_plateform_xyz_bis = CS5_to_PMA_sled(strut_plateform_xyz_cs5)

delta_strut_pma_coords = strut_base_xyz_bis-strut_base_xyz
print("delta x struts PMA=",delta_strut_pma_coords[0,0:2],)
print("delta y struts PMA=",delta_strut_pma_coords[1,0:2])
print("delta z struts PMA=",delta_strut_pma_coords[2,0:2])
delta_strut_pma_coords = strut_plateform_xyz_bis-strut_plateform_xyz
print("delta x struts PMA=",delta_strut_pma_coords[0,0:2])
print("delta y struts PMA=",delta_strut_pma_coords[1,0:2])
print("delta z struts PMA=",delta_strut_pma_coords[2,0:2])

#length=np.sqrt(np.sum((strut_plateform_xyz-strut_base_xyz)**2,axis=0))
#print("length=",length)

target_plate_xyz = CS5_to_PMA_sled(target_plate_xyz_cs5)
#print("target_plate_xyz_pma=",target_plate_xyz)

shim_xyz_cs5=np.zeros((3,4))
shim_xyz_cs5[:,0]=[1540.1,2144.2,-1002.8]
shim_xyz_cs5[:,1]=[2174.3,1509.9,-1033.6]
shim_xyz_cs5[:,2]=[2436.81,1662.74,-3377.6]
shim_xyz_cs5[:,3]=[1695.3,2405.3,-3346.8]
shim_xyz_cs5 /= inch2mm # now in inch
shim_xyz_pma = CS5_to_PMA_sled(shim_xyz_cs5)



def xyz2plot(xyz) :
    res=np.zeros(xyz.shape)
    res[0]=xyz[0] # x
    res[1]=-xyz[2] # -z
    res[2]=xyz[1] # y
    return res

plt.figure("3D-CS5")
ax = plt.axes(projection='3d')
for s in range(6) :
    xyz1=xyz2plot(strut_base_xyz[:,s])
    xyz2=xyz2plot(strut_plateform_xyz[:,s])

    #ax.plot3D(xyz1[0],xyz1[1],xyz1[2],"o",color="b")
    ax.text3D(xyz1[0],xyz1[1],xyz1[2],"S{}".format(s+1),color="b")

    ax.plot3D([xyz1[0],xyz2[0]],
              [xyz1[1],xyz2[1]],
              [xyz1[2],xyz2[2]],"-",color="b")

xyz=xyz2plot(target_plate_xyz)
ax.plot3D(xyz[0],xyz[1],xyz[2],color="purple",label="target carriage plate")

side=140
xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=float).T
xyz-=np.array([0.5,0.5,0.5])[:,None]
xyz *= side
xyz[2]-=60
xyz[1]-=10
ax.plot3D(xyz[0],xyz[1],xyz[2],".",color="gray")

ax.set_xlabel('x_pma')
ax.set_ylabel('-z_pma')
ax.set_zlabel('y_pma')


# now compute corrections

# in PMA coordinates z=CS5 y=up x=towards up
meas_plate_xyz = target_plate_xyz.copy()
meas_plate_xyz[1] += 1. # one inch above

plate_y = np.mean(meas_plate_xyz[1])
meas_xyz=np.zeros((3,3))
meas_xyz[:,0]=[0,0.213,0]
meas_xyz[:,1]=[0,0.4,-30.5]
meas_xyz[:,2]=[-34.,0.612,-31]
meas_xyz[0] += 15.
meas_xyz[0] += 3.

if 0 :
    print("FIT TRANSFO USING MEASUREMENTS ON PLATE")
    target_xyz=meas_xyz.copy()
    target_xyz[1]=0.
    transfo=Transfo3D()
    rms=transfo.fit(meas_xyz,target_xyz)
    print("plate meas transfo rms=",rms)

if 1 :
    print("FIT TRANSFO USING MEASUREMENT OF AXIS IN CS5")
    meas_axis_xyz_pma   = CS5_to_PMA_sled(meas_axis_xyz_cs5)
    target_axis_xyz_pma = CS5_to_PMA_sled(target_axis_xyz_cs5)
    print("delta axis=",(meas_axis_xyz_pma-target_axis_xyz_pma))

    transfo=Transfo3D()
    rms=transfo.fit(meas_axis_xyz_pma,target_axis_xyz_pma)
    print("meas axis transfo rms=",rms)

if 0 :
    print("FIT TRANSFO USING MEASUREMENT OF HEIGHT of PLATE")
    transfo=Transfo3D()
    rms=transfo.fit(meas_plate_xyz,target_plate_xyz)
    print("meas plate transfo rms=",rms)

xyz=xyz2plot(meas_plate_xyz)
ax.plot3D(xyz[0],xyz[1],xyz[2],color="green",label="meas carriage plate")

meas_xyz_plot = meas_xyz.copy()
meas_xyz_plot[1] += np.mean(meas_plate_xyz[1])-np.mean(meas_xyz_plot[1])
xyz=xyz2plot(meas_xyz_plot)
ax.plot3D(xyz[0],xyz[1],xyz[2],"+",color="green",label="measurements plate")
ax.plot3D([xyz[0,2]],[xyz[1,2]],[xyz[2,2]],"+",color="red",label="measurements plate")

# apply to strut
new_strut_plateform_xyz = transfo.apply(strut_plateform_xyz)
length     = np.sqrt(np.sum((strut_plateform_xyz-strut_base_xyz)**2,axis=0))
new_length = np.sqrt(np.sum((new_strut_plateform_xyz-strut_base_xyz)**2,axis=0))
print("init length=",length)
print("new  length=",new_length)

delta_length = new_length - length
for s in range(len(delta_length)) :
    print("strut {} delta_length = {:+.4f} inch".format(s+1,delta_length[s]))


real_measured_length=np.array([21.613,21.621,21.613,21.663,21.660,21.617])

target_measured_length=real_measured_length+delta_length
for s in range(len(delta_length)) :
    print("strut {} new length = {:+.4f} inch".format(s+1,target_measured_length[s]))

xyz=xyz2plot(shim_xyz_pma)
ax.plot3D(xyz[0],xyz[1],xyz[2],"X",color="brown",label="shim")
dy=shim_xyz_pma[1,2]-shim_xyz_pma[1,1]
dz=shim_xyz_pma[2,2]-shim_xyz_pma[2,1]
angle=np.arctan(dy/dz)
print("angle of PMA to ground = ",angle*180/np.pi)

ca=np.cos(angle)
sa=np.sin(angle)
vertical = np.array([0,ca,sa])
shim_xyz_base_pma = shim_xyz_pma - 10*vertical[:,None]

for s in range(4) :
    xyz1=xyz2plot(shim_xyz_pma[:,s])
    xyz2=xyz2plot(shim_xyz_base_pma[:,s])
    ax.plot3D([xyz1[0],xyz2[0]],
              [xyz1[1],xyz2[1]],
              [xyz1[2],xyz2[2]],"-",color="brown")
    ax.text3D(xyz1[0],xyz1[1],xyz1[2],"shim {}".format(s+1),color="brown")

shim_xyz_pma_corr = transfo.apply(shim_xyz_pma)

length     = np.sqrt(np.sum((shim_xyz_pma-shim_xyz_base_pma)**2,axis=0))
new_length = np.sqrt(np.sum((shim_xyz_pma_corr-shim_xyz_base_pma)**2,axis=0))
#print("init length=",length)
#print("new  length=",length)

delta_length = new_length - length
if 1 :
    print("Shim relative variation")
    delta_length -= np.mean(delta_length)
for s in range(len(delta_length)) :
    print("shim {} delta_length = {:+.4f} inch".format(s+1,delta_length[s]))


plt.show()
sys.exit(0)

# 0.078
# 0.083
