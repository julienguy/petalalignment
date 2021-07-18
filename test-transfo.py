#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from transfo import Transfo3D,OldTransfo3D


# test assuming mm

npts=20
xyz=np.zeros((3,npts))
xyz[0]=np.random.uniform(size=npts)*1000. # typical size is 1 m
xyz[1]=np.random.uniform(size=npts)*1000.
xyz[2]=np.random.uniform(size=npts)*1000.
#xyz[2]=2000.

dxyz=np.array([10,30,40])


print("test 1")

a=30.#deg
ca=np.cos(a/180*np.pi)
sa=np.sin(a/180*np.pi)
rot=np.array([[ca,sa,0],[-sa,ca,0],[0,0,1]])

xyz2 = rot.dot(xyz+dxyz[:,None])

t=Transfo3D()
rms=t.fit(xyz,xyz2)
print(t)
print("Transfo rms:",rms)

print("test 2")

a=-20.#deg
ca=np.cos(a/180*np.pi)
sa=np.sin(a/180*np.pi)
rot=np.array([[1,0,0],[0,ca,sa],[0,-sa,ca]])

xyz2 = rot.dot(xyz+dxyz[:,None])

t=Transfo3D()
rms=t.fit(xyz,xyz2)
print(t)
print("Transfo rms:",rms)

print("test 3")

a=10.#deg
ca=np.cos(a/180*np.pi)
sa=np.sin(a/180*np.pi)
rot=np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])

xyz2 = rot.dot(xyz+dxyz[:,None])

t=Transfo3D()
rms=t.fit(xyz,xyz2)
print(t)
print("Transfo rms:",rms)

print("test 4")

a=10.#deg
ca=np.cos(a/180*np.pi)
sa=np.sin(a/180*np.pi)
rot=rot.dot([[ca,0,sa],[0,1,0],[-sa,0,ca]])

xyz2 = rot.dot(xyz+dxyz[:,None])

xyz2[0] += 100.
xyz2[1] += 300.
xyz2[2] += 400.

t=Transfo3D()
rms=t.fit(xyz,xyz2)
print(t)
print("Transfo rms:",rms)

print("test 5")

a=30.#deg
ca=np.cos(a/180*np.pi)
sa=np.sin(a/180*np.pi)
rot=np.array([[ca,sa,0],[-sa,ca,0],[0,0,1]])

xyz2 = rot.dot(xyz+dxyz[:,None])

xyz2[0] = -xyz2[0] # change one sign

t=Transfo3D()
rms=t.fit(xyz,xyz2)
print(t)
print("Transfo rms:",rms)
