# Petal alignment python script

## Purpose

Measure corrections to the PMA arm and PMA leg rotation angles (theta and phi), and
determine corrections to the length of the 6 lower sled struts and the 6 upper PMA struts based on the laser tracker measured coordinates
of an array of reflective balls (Ball Mount Reflectors in the code), for different configurations.

3 BMR setups are supported

* BMR plate on petal mounted on the 2 Guide spikes for petal insertion
* Light weight BMR plate mounted on the tip of the red leg for petal removal
* Heavy weight BMR plate mounted on the tip of the red leg for petal removal

The code allows to choose which of the 14 degrees of freedom to adjust.

## Documents
### Metrology
 * DESI-4754 FPD Laser Track Nest Metrology

      (FPD=Focal Plane aDapter, the large conic cylinder that holds the petal ring, this defines CS5 for the alignment)

 * DESI-4949 FPP Laser Target Mount Metrology (FPP=Focal Plane Petal)
 * DESI-6206 Red Leg tip mounting holes metrology
 * DESI-6207-B Leg Laser Target Mount Metrology
 * DESI-6211-B FPP Mass Dummy Endplate Metrology
### Other
 * DESI-5905 Precision Alignment and Integration of DESI's Focal Plane using a Laser Tracker
 * DESI-3224 addendum: Procedure for transverse card swap campaign

## Usage

Single python script to simplify installation and usage on windows machine.

```python PAC.py example-config.yaml```

All inputs are provided in a yaml file. For instance, in `example-config.yaml`:
```
# Petal Position (Per DESI-3596)

petal: 5

# Which BMR (Ball Mount Refectors) mount plate configuration is used.
# 3 choices :
#  "guide_spikes" BMR plate mounted on the petal guide spikes for petal insertion procedure
#  "light_weight_red_leg" # BMR plate mounted at the tip of the red
#  "heavy_weight_red_leg" # BMR plate mounted at the tip of the red with a weight matching that of a petal

bmr_type: "light_weight_red_leg"

# PMA translation axis misalignment parameters
# Are we going to correct for PMA axis misalignment ( 0 or 1 )

correct_pma_misalignement: 1

# this is the coordinates of the center of rotation
# of the PMA arm, using the nest glued on the red
# counter weight of the arm

partially_engaged_pma_coords_inch: 0.0174,-0.0007,-72.6068
retracted_pma_coords_inch: 0.0112,0.0036,-101.5149
fully_engaged_pma_z_coord_inch: -50 # UNKNOWN FOR NOW

# correct for the PMA arm rotation (THETA angle)
# (used to go from one petal to the other)
# 0=no or 1=yes

correct_pma_arm_rotation: 0

# correct for the leg rotation (PHI angle)
# 0=no or 1=yes

correct_pma_leg_rotation: 1

# use the lower struts to correct for
# the vertical offset
# 0=no or 1=yes

correct_lower_struts_length: 1

# uncomment this to save the moves in an output csv table file

# outfile: example-moves.csv

# plot results (0=no or 1=yes)

plot: 0

# measured BMR coordinates
# name: X , Y , Z
# in CS5 and inches
# (if you miss one ball, just comment out the entry)

B1: -3.7424 , 16.1845 , -64.0084
B2: 1.7538 , 16.0473 , -64.0017
B3: 3.9335 , 17.8724 , -63.9971
B4: -2.3009 , 19.6984 , -64.0030

# Uncomment the following ONLY to impose
# a shift of the PMA based on the
# visual inspection of the linear bearing
# alignment to the rail.
# THIS OVERWRITES THE ABOVE BMR COORDINATES
# Use either CS5 or PMA coordinates
# (not both at the same time)
# offset units in mm
#FORCED_DX_CS5_MM: 0.
#FORCED_DY_CS5_MM: 1.
# x pma is horizontal, positive towards the laser tracker
# y pma is vertical, positive upward
#FORCED_DX_PMA_MM: 0.
#FORCED_DY_PMA_MM: 1.

# Optional, rail measurements
# X,Y,Z of points along the rail in CS5 and inches 
# (one row per point)
RAIL1: 0.1588,19.6727,-85.0901
RAIL2: 0.1570,19.6663,-80.9223
RAIL3: 0.1600,19.6575,-75.4784
RAIL4: 0.1660,19.6484,-70.1929
RAIL5: 0.1751,19.6391,-65.3240
RAIL6: 0.1799,19.6318,-60.9431
```
