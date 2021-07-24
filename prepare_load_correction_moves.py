#!/usr/bin/env python

import numpy as np
from astropy.table import Table

from PAC import inch2mm,compute_target_bmr_heavy_weight_red_leg_coords_mm

table=Table.read("PetalLoadNoLoad3times.csv")
x=list(table["ActualX"])
y=list(table["ActualY"])
z=list(table["ActualZ"])


for petal in range(10) :

    # inches , cs5
    bmr_xyz_nls = np.zeros((3,3,5))
    bmr_xyz_fls = np.zeros((3,3,5))

    for meas in range(3) :

        for b in range(5) :
            i=np.where(table["Name"]=="P{:d}_FL_{:d}_723_H{:d}".format(petal,meas+1,b+1))[0][0]
            #print(petal,b+1,i)
            bmr_xyz_fls[meas,:,b] = [x[i],y[i],z[i]]
            i=np.where(table["Name"]=="P{:d}_NL_{:d}_723_H{:d}".format(petal,meas+1,b+1))[0][0]
            #print(petal,b+1,i)
            bmr_xyz_nls[meas,:,b] = [x[i],y[i],z[i]]


    bmr_xyz_fl=np.mean(bmr_xyz_fls,axis=0)
    bmr_xyz_nl=np.mean(bmr_xyz_nls,axis=0)
    bmr_xyz_err=np.std(bmr_xyz_fls-bmr_xyz_nls,axis=0)/np.sqrt(3.-1.)

    bmr_sag_dx=np.mean(bmr_xyz_fls[:,0,:],axis=-1)-np.mean(bmr_xyz_nls[:,0,:],axis=-1)
    bmr_sag_dy=np.mean(bmr_xyz_fls[:,1,:],axis=-1)-np.mean(bmr_xyz_nls[:,1,:],axis=-1)
    bmr_sag_dr=np.sqrt(bmr_sag_dx**2+bmr_sag_dy**2)
    bmr_sag_dr_mm=bmr_sag_dr*inch2mm
    #print("petal",petal,"dx(mm) =",bmr_sag_dx*inch2mm)
    #print("petal",petal,"dy(mm) =",bmr_sag_dy*inch2mm)
    #print("petal",petal,"dr(mm) =",bmr_sag_dr*inch2mm)

    mean_bmr_sag_dr_mm = np.mean(bmr_sag_dr_mm)
    err_bmr_sag_dr_mm = np.std(bmr_sag_dr_mm)/np.sqrt(2.)
    print("petal {} dr = {:+.3f} +- {:.3f} mm ({:.1f}%)".format(petal,mean_bmr_sag_dr_mm,err_bmr_sag_dr_mm,100*err_bmr_sag_dr_mm/mean_bmr_sag_dr_mm))

    # get the target locations
    bmr_xyz_targets_mm , junk = compute_target_bmr_heavy_weight_red_leg_coords_mm(petal=petal,plot=False)
    bmr_xyz_targets = bmr_xyz_targets_mm/inch2mm

    # set measurement as effect of load
    bmr_xyz_meas = bmr_xyz_targets + bmr_xyz_fl-bmr_xyz_nl

    filename = f"petal{petal}-heavy-red-leg-20210723-FL2NL3.yaml"
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
        file.write('BERR{:d}: {:f},{:f},{:f}\n'.format(b+1,
                                                       bmr_xyz_err[0,b],
                                                       bmr_xyz_err[1,b],
                                                       bmr_xyz_err[2,b]))
    file.write(f'outfile: petal{petal}-load-correction-moves.csv\n')
    file.write('plot: 0\n')
    file.close()
    #print("wrote",filename)
    logfilename=f"log-petal{petal}-heavy-red-leg-20210723-FL2NL3.txt"
    cmd="./PAC.py "+filename+" > "+logfilename
    print(cmd)

    # grep -e "Leg rotation angle correction to apply"  -e LS -e US log-petal9-heavy-red-leg-20210722-FL2NL.txt
