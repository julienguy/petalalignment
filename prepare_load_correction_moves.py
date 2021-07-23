#!/usr/bin/env python

import numpy as np
from astropy.table import Table

from PAC import inch2mm,compute_target_bmr_heavy_weight_red_leg_coords_mm

table=Table.read("PetalShiftsLoadNoLoad-20210722.csv")
x=list(table["ActualX"])
y=list(table["ActualY"])
z=list(table["ActualZ"])


for petal in range(10) :

    # inches , cs5
    bmr_xyz_nl = np.zeros((3,5))
    bmr_xyz_fl = np.zeros((3,5))

    for b in range(5) :
        i=np.where(table["Name"]=="P{:d}_FL_H{:d}".format(petal,b+1))[0][0]
        #print(petal,b+1,i)
        bmr_xyz_fl[:,b] = [x[i],y[i],z[i]]
        i=np.where(table["Name"]=="P{:d}_NL_H{:d}".format(petal,b+1))[0][0]
        #print(petal,b+1,i)
        bmr_xyz_nl[:,b] = [x[i],y[i],z[i]]

    # get the target locations
    bmr_xyz_targets_mm , junk = compute_target_bmr_heavy_weight_red_leg_coords_mm(petal=petal,plot=False)
    bmr_xyz_targets = bmr_xyz_targets_mm/inch2mm

    # set measurement as effect of load
    bmr_xyz_meas = bmr_xyz_targets + bmr_xyz_fl-bmr_xyz_nl

    filename = f"petal{petal}-heavy-red-leg-20210722-FL2NL.yaml"
    file = open(filename,"w")
    file.write(f"petal: {petal}\n")
    file.write('bmr_type: "heavy_weight_red_leg"\n')
    file.write('correct_pma_misalignement: 0\n')
    file.write('correct_pma_arm_rotation: 0\n')
    file.write('correct_pma_leg_rotation: 1\n')
    file.write('correct_lower_struts_length: 1\n')
    for b in range(5) :
        file.write('B{:d}: {:f},{:f},{:f}\n'.format(b+1,
                                                 bmr_xyz_meas[0,b],
                                                 bmr_xyz_meas[1,b],
                                                 bmr_xyz_meas[2,b]))
    file.write(f'outfile: petal{petal}-load-correction-moves.csv\n')
    file.write('plot: 0\n')
    file.close()
    #print("wrote",filename)
    logfilename=f"log-petal{petal}-heavy-red-leg-20210722-FL2NL.txt"
    cmd="./PAC.py "+filename+" > "+logfilename
    print(cmd)

    # grep -e "Leg rotation angle correction to apply"  -e LS -e US log-petal9-heavy-red-leg-20210722-FL2NL.txt
