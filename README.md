# Petal alignment python script

## Purpose

Determine the length of the 6 PMA struts based on the laser tracker measured coordinates
of a array of reflective balls (Ball Mount Reflectors in the code).

3 BMR setups are supported

* BMR plate on petal mounted on the 2 Guide spikes for petal insertion
* Light weight BMR plate mounted on the tip of the red leg for petal removal
* Heavy weight BMR plate mounted on the tip of the red leg for petal removal


## Documents

 * DESI-3224 addendum: Procedure for transverse card swap campaign
 * DESI-4949 FPP Laser Target Mount Metrology
 * DESI-6206 Red Leg tip mounting holes metrology
 * DESI-6207-B Leg Laser Target Mount Metrology
 * DESI-6211-B FPP Mass Dummy Endplate Metrology

## Usage

Single python script to simplify installation and usage on windows machine.

```python PAC.py```

For now, the inputs have to be written directly in the script
(again, this is to run both on a linux and windows machines)
  * petal: petal being aligned. This is the petal location from 0 to 9.
  * type of BMR "guide_spikes" "light_weight_red_leg" or "heavy_weight_red_leg"
  * measured BMR coordinates in CS5 


