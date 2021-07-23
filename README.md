# Petal alignment python script

## Purpose

Determine the length of the 6 PMA struts based on the laser tracker measured coordinates
of an array of reflective balls (Ball Mount Reflectors in the code), for different configurations.

3 BMR setups are supported

* BMR plate on petal mounted on the 2 Guide spikes for petal insertion
* Light weight BMR plate mounted on the tip of the red leg for petal removal
* Heavy weight BMR plate mounted on the tip of the red leg for petal removal


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

All inputs are provided in a yaml file. For instance, in `test/test-petal0.yaml`:
```
# Petal Position (Per DESI-3596)

petal: 0

# Which BMR (Ball Mount Refectors) mount plate configuration is used.
# 3 choices :
#  "guide_spikes" BMR plate mounted on the petal guide spikes for petal insertion procedure
#  "light_weight_red_leg" # BMR plate mounted at the tip of the red
#  "heavy_weight_red_leg" # BMR plate mounted at the tip of the red with a weight matching that of a petal

bmr_type: "guide_spikes"

# PMA translation axis misalignment parameters
# Are we going to correct for PMA axis misalignment ( 0 or 1 )
correct_pma_misalignement: 1

# Coordinates in CS5 of a fixed point on the PMA when the PMA is (partially) engaged
# (i.e. close to the FPA)
# Coordinates X,Y,Z (inch)
partially_engaged_pma_coords_inch: 0,0,-10

# Coordinates in CS5 of the same fixed point on the PMA when the PMA is retracted
# (i.e. far from the FPA)
# Coordinates X,Y,Z (inch)
retracted_pma_coords_inch: 0,0,-150

# Z coordinate in CS5 of the same fixed point on the PMA when the PMA is full engaged
# This has to be pre-computed once a fixed point is chosen
fully_engaged_pma_z_coord_inch: -5

# BMR (balls) Measurements from laser tracker in CS5 in inches
# X , Y , Z

B1: 3.3247, -16.5683, -55.9022
B2: 1.7563, -18.9263, -55.9158
B3: -3.5547, -20.3330, -55.9186
B4: -3.1565, -16.5614, -55.9040
```
