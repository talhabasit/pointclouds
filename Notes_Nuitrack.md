# Pointcloud Tests

## Observations

1. Floor points in PCDs might pose a problem in post processing.
2. Background plane removal with pcdplanefit() worked.
3. saving PCDs and joints works but still low fps
4. Time sync is a problem no direct correspondance might have to interpolate between PCDs
5. Timestamp difference to be tested i.e the timestamps from Nuitrack and python are different
6.  

## Brain Dump

1. Pickle nuitrack data

2. pickle name is nuitrack timestamp

3. Pickling didnt work

4. pickling arbitrary objects doesnt work

## Nuitrack live pointcloud Visu

1. 2D and 3D visulaizations are complete 
2. The 3D is taking ~40ms per frame whereas the 2D one is taking ~7-8ms per Frame
