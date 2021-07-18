#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from transfo import Transfo3D,OldTransfo3D

npts=4
xyz=np.zeros((3,npts))
#np.random.seed(12)
xyz[0]=np.random.uniform(size=npts)
xyz[1]=np.random.uniform(size=npts)
xyz[2]=12.

a=30.#deg
ca=np.cos(a/180*np.pi)
sa=np.sin(a/180*np.pi)

rot=np.array([[ca,sa,0],[-sa,ca,0],[0,0,1]])
xyz2=rot.dot(xyz)
xyz2[0] += 1.
xyz2[1] += 3.
xyz2[2] += 4.

t=OldTransfo3D()
rms=t.fit(xyz,xyz2)
print("old 3D:",rms)

t=Transfo3D()
rms=t.fit(xyz,xyz2)
print("3D:",rms)
