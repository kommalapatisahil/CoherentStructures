# CoherentStructures
Bayesian Optimization to identfy coherent structures in Turbulent boundary layers. 

Here is the related abstract that was sent to APS-DFD 2020 [Abstract](https://meetings.aps.org/Meeting/DFD20/Session/P18.21) and my [talk](https://youtu.be/sseXCqn1wEY). 

![](https://github.com/kommalapatisahil/CoherentStructures/blob/master/dd.PNG)

Contents of the repository. 
## Notebooks.

SMR_W14_automation_heightV11.ipynb - Fully auomated framework that takes the PIV data as an input and produces the cumulative vortex properties convecting as a group in the captured flow. Includes analysis on the results and error estimates. The algorithm gives an ~80% efficiency, in terms of finding a converged global solution. 


## Modules

BoxPlot.py - contains functions to analyze the performance of our implementation against traditional (Truncated Newton's) minimization. 

PIVutils.py - contains utility functions to handle the PIV data. 

PODutils.py - contains utility functions to execute POD and other methods to analyse the PIV data. 

grafteaux.py - evaluate the T2 function defined by Laurent Graftieaux et al. (2001) to identify vortex cores. (and other supporting functions.)

automateG.py - contains function to automate the graftieaux (G) based identification. 

prom2d.py - finds the prominence regions of a 2d signal.


