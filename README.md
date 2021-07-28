# Automated Identification of turbulent CoherentStructures in boundary layer flow datasets. 

This repository contains code to replicate the results from our (Williams turbulence laboratory, UW) research on automated identification of turbulent coherent structures. Particularly, we focus on Hairpin vortex structures (signatures) which are the elemental units that contribute towards the generation and sustainence of wall bounded turbulence. 

A bayesian inference framework using Markov chain Monte carlo is developed to achieve the above. 

Here is the related [abstract](https://meetings.aps.org/Meeting/DFD20/Session/P18.21) that was sent to APS-DFD 2020 and my [talk](https://youtu.be/sseXCqn1wEY). 


Here is an overview of the approach.

It begins with the wall normal - stream wise velocity components of the PIV frame. The magnitude of these components is visualized below. 

![](https://github.com/kommalapatisahil/CoherentStructures/files/pl1.PNG)

Utilizing the velocity components, an objective vortex identification field, T2, is applied on the frame, to identify potential vortex candidates. These candidates appear as blobs/ clusters in the plot below. 

![](https://github.com/kommalapatisahil/CoherentStructures/files/pl2.PNG)

The contours of uniform velocity are brought back (in black). It should be observed that the vortex candidates appear at the intersection of different uniform velocity contours.

![](https://github.com/kommalapatisahil/CoherentStructures/files/pl3.PNG)

A two dimensional peak prominence threshold is utilized to identify potential vortex candidates. These candidates are shown in red in the plot below. 

![](https://github.com/kommalapatisahil/CoherentStructures/files/pl4.PNG)


At this stage an MCMC based matching procedure is executed to extract the properties of all vortices present in this frame. The output of matching for each vortex candidate (shown in red bounding boxes) is a corner plot that contains 5 PDFs as shown below. 

![](https://github.com/kommalapatisahil/CoherentStructures/files/mcmc1.PNG)

The peak of each PDF corresponds to the likely value of the vortex property that would optimize the matching procedure. This proceduce is repeated for all other vortices in all the available frames. 

The cummulative results from identification of vortices in 750 non-time resolved PIV measurements of boundary layer flow are visualized using Seaborn's violin plots. For example, here is the KDE plot for the convective velocity of all vortices identified at three different heights using MCMC and Truncated Newtons' minimization. (MIN) 

![](https://github.com/kommalapatisahil/CoherentStructures/files/piv5.PNG)

Here is a brief overview of the vortex matching procedure. 

![](https://github.com/kommalapatisahil/CoherentStructures/files/overview1.PNG)


And, here is an overview on how to handle divergence in identification. 

![](https://github.com/kommalapatisahil/CoherentStructures/files/own22.PNG)

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


