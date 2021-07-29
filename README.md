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

All inputs are provided in a yaml file. See for instance https://github.com/julienguy/petalalignment/blob/master/example-config.yaml .
