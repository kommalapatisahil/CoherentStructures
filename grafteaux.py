#------------------------------------------------------------------------------------#
#-----------*Grafteaux*--------------------------------------------------------------#
#------------------------------------------------------------------------------------#

#def grafteauxT1(X, Y, U, V, hop):

import numpy as np
import math
eps = np.finfo(float).eps


def T1_big_mat(X, Y, U, V, hop):
    
    def stencil(i, j, S, hop):
        stemp =  S[i-hop:i+hop+1, j-hop:j+hop+1]
        #if stemp.shape != (2*hop+1, 3): print('stencil issue.',stemp.shape, S.shape)
        return stemp

    def Tau1(c1, X, Y, Usim, Vsim):
        TM = thetaM(c1, X, Y, Usim, Vsim)
        to_radians = TM/180*np.pi
        #print(TM, '\n TM')

        return np.mean(np.sin(to_radians))

    def thetaM(c1, X, Y, Usim, Vsim): 
        #if X.shape != Usim.shape :
        #    print('Somethings wrong!')
        #    print(X.shape, 'X')
        #    print(Y.shape, 'Y')
        #    print(Usim.shape, 'U')
        #    print(Vsim.shape, 'V')
        PM = PM_mat(c1, X, Y)
        #print(PM, '\n PM')
        UM = UM_mat(Usim, Vsim)

        #ThetaM = np.zeros(PM.shape)
        ThetaM = 180-PM + UM 
        return ThetaM

    def UM_mat(Usim1, Vsim1):
        '''
        make sure to remove the /np.pi*180 while evaluating 
        '''
        UM = np.zeros(Usim1.shape)

        for i in range(len(Usim1)) : 
            for j in range(len(Usim1[i])): 
                pt1 = (Usim1[i,j],0)
                pt2 = (0, Vsim1[i,j])
                UM[i,j] = a2(pt1, pt2)

        return UM

    def PM_mat(c1, X, Y):
        '''
        c1 - the location of the center (P)
        X, Y - the mesh grids of x, y around point P
        returns the angles PM as a mat with dim of X
        '''
        center = [X[c1[0], c1[1]], Y[c1[0], c1[1]]]
        #print(center)
        angles = np.zeros(X.shape)
        for i in range(len(X)):
                for j in range(len(X[i])):
                    p2 = [X[i,j], Y[i,j]]
                    angles[i,j] = ang(center, p2)
        return angles 
    
    def a2(p1, p2):
        '''
        modified form of ang
        now p1 and p2 have vector components
        '''
        #return(np.arctan(p2[1]/(p1[0] + eps))/np.pi*180)
        #print(p1, p2, "supposed to be points")
        return(math.atan2(p2[1], (p1[0] + eps)) /np.pi*180)

    def ang(p1, p2):
        '''
        pi = (xi, yi)
        returns angles between the pts p1 and p2
        '''    
        eps = np.finfo(float).eps
        theta = (p2[1]-p1[1])/(p2[0] - p1[0] + eps)
        return math.atan2((p2[1]-p1[1]), (p2[0] - p1[0] + eps))/np.pi*180


    
    T1_big = np.zeros(X.shape)
    for i in range(hop, len(T1_big)-hop):
        for j in range(hop, len(T1_big[i])-hop): 
                
                X_temp = stencil(i, j, X, hop)
                Y_temp = stencil(i, j, Y, hop)
                U_temp = stencil(i, j, U, hop)
                V_temp = stencil(i, j, V, hop)
                
                #print(X_temp, '\n x', Y_temp, '\n y', U_temp, '\n u', V_temp, '\n v')
                
                cent = (hop, hop)
                t1_temp = Tau1(cent, X_temp, Y_temp, U_temp, V_temp)
                T1_big[i,j] = t1_temp 
                
    return T1_big 

#basically T2 but UM is modified. 

def T2_big_mat(X, Y, U, V, hop):
    
    def stencil(i, j, S, hop):
        stemp =  S[i-hop:i+hop+1, j-hop:j+hop+1]
        #if stemp.shape != (2*hop+1, 3): print('stencil issue.',stemp.shape, S.shape)
        return stemp

    def Tau1(c1, X, Y, Usim, Vsim):
        TM = thetaM(c1, X, Y, Usim, Vsim)
        to_radians = TM/180*np.pi
        #print(TM, '\n TM')

        return np.mean(np.sin(to_radians))

    def thetaM(c1, X, Y, Usim, Vsim): 
        
        PM = PM_mat(c1, X, Y)
        UM = UM_mat2(Usim, Vsim)

        ThetaM = 180-PM + UM 
        return ThetaM
    
    #**--Important change for T2--**#

    def UM_mat2(Usim1, Vsim1):
        '''
        The only function which changes for T2 in Grafteaux 
        make sure to remove the /np.pi*180 while evaluating 
        '''
        UM = np.zeros(Usim1.shape)
        
        umean = np.mean(Usim1)
        vmean = np.mean(Vsim1)
        
        for i in range(len(Usim1)) : 
            for j in range(len(Usim1[i])): 
                pt1 = (Usim1[i,j]-umean,0)
                pt2 = (0, Vsim1[i,j]-vmean)
                UM[i,j] = ang2(pt1, pt2)

        return UM

    def PM_mat(c1, X, Y):
        '''
        c1 - the location of the center (P)
        X, Y - the mesh grids of x, y around point P
        returns the angles PM as a mat with dim of X
        '''
        center = [X[c1[0], c1[1]], Y[c1[0], c1[1]]]
        #print(center)
        angles = np.zeros(X.shape)
        for i in range(len(X)):
                for j in range(len(X[i])):
                    p2 = [X[i,j], Y[i,j]]
                    angles[i,j] = ang(center, p2)
        return angles 
    
    def ang2(p1, p2):
        '''
        modified form of ang
        now p1 and p2 have vector components
        '''
        #return(np.arctan(p2[1]/(p1[0] + eps))/np.pi*180)
        #print(p1, p2)
        return(math.atan2(p2[1], (p1[0] + eps)) /np.pi*180)

    def ang(p1, p2):
        '''
        pi = (xi, yi)
        returns angles between the pts p1 and p2
        '''    
        eps = np.finfo(float).eps
        theta = (p2[1]-p1[1])/(p2[0] - p1[0] + eps)
        return math.atan2((p2[1]-p1[1]), (p2[0] - p1[0] + eps))/np.pi*180


    
    T1_big = np.zeros(X.shape)
    for i in range(hop, len(T1_big)-hop):
        for j in range(hop, len(T1_big[i])-hop): 
                
                X_temp = stencil(i, j, X, hop)
                Y_temp = stencil(i, j, Y, hop)
                U_temp = stencil(i, j, U, hop)
                V_temp = stencil(i, j, V, hop)
                
                #print(X_temp, '\n x', Y_temp, '\n y', U_temp, '\n u', V_temp, '\n v')
                
                cent = (hop, hop)
                t1_temp = Tau1(cent, X_temp, Y_temp, U_temp, V_temp)
                T1_big[i,j] = t1_temp 
                
    return T1_big 


def plt_3d(X, Y, S, hop = 1, title= "T1 function"):
    '''
    To plot the T1 field to visualize the 3d field better
    '''
    
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib as cm
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize = (8,8))
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X[hop:-hop,hop:-hop], Y[hop:-hop, hop:-hop],S[hop:-hop, hop:-hop],cmap = 'spring',\
                           linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_xlabel('y')
    ax.set_title(title)
    return 0



#-------------------------------------------------------------------------------------
#MCMC related functions
#-------------------------------------------------------------------------------------


def plot_corner(sampler, nburn, saveFolder = None):
    '''
    takes the sampler object as input 
    '''
    
    ndim = 5
    emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T
    import corner
    labels = ['Vort Circ','Vort R','x','y','Convective Vel.']
    label_kwargs = {'fontsize':16}
    f = corner.corner(emcee_trace.T, labels = labels,label_kwargs = label_kwargs)
    if saveFolder != None :
        f.savefig(saveFolder)
    return 


def doMCMC_V3(Usim1, Vsim1, x, y):
    import PODutils
    import PIVutils
    
    
    SR = 5
    #is this giving a bigger range to search for?
    #target = [Circ,r,xc,yc,Conv]

    bounds = [(0.01*2*np.pi*(x[1]-x[0]), None), (x[1]-x[0],np.max(x)),\
              (-1*SR*(x[1]-x[0]),SR*(x[1]-x[0])),(-1*SR*(x[1]-x[0]),SR*(x[1]-x[0])),\
              (-0.15, 0.15)]

    target1 = [0.08, 0.05, 0.01, 0.01, 0.02]

    ndim = len(target1)  # number of parameters in the model
    nwalkers = 50  # number of MCMC walkers
    nburn = 5000  # "burn-in" period to let chains stabilize
    nsteps = 10000  # number of MCMC steps to take

    initFit = target1

    np.random.seed(0)
    starting_guesses = np.random.random((nwalkers, ndim))
    starting_guesses = initFit+initFit*starting_guesses*0.01

    import emcee

    bounds[-1] = (-0.25,0.25)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, PODutils.log_posterior_V3_10r, args=[Usim1,Vsim1,x,y,bounds])
    sampler.run_mcmc(starting_guesses, nsteps)
    #%time sampler.run_mcmc(starting_guesses, nsteps)
    
    return sampler 

def doMCMC_V4(Usim1, Vsim1, x, y):
    import PODutils
    import PIVutils
    from importlib import reload 
    PODutils = reload(PODutils)
    
    SR = 5
    #is this giving a bigger range to search for?
    #target = [Circ,r,xc,yc,Conv]

    bounds = [(0.01*2*np.pi*(x[1]-x[0]), None), (x[1]-x[0],np.max(x)),\
              (-1*SR*(x[1]-x[0]),SR*(x[1]-x[0])),(-1*SR*(x[1]-x[0]),SR*(x[1]-x[0])),\
              (-0.15, 0.15)]

    target1 = [0.08, 0.05, 0.01, 0.01, 0.02]

    ndim = len(target1)  # number of parameters in the model
    nwalkers = 50  # number of MCMC walkers
    nburn = 5000  # "burn-in" period to let chains stabilize
    nsteps = 10000  # number of MCMC steps to take

    initFit = target1

    np.random.seed(0)
    starting_guesses = np.random.random((nwalkers, ndim))
    starting_guesses = initFit+initFit*starting_guesses*0.01

    import emcee

    bounds[-1] = (-0.25,0.25)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, PODutils.log_posterior10r_rect, args=[Usim1,Vsim1,x,y,bounds])
    sampler.run_mcmc(starting_guesses, nsteps)
    #%time sampler.run_mcmc(starting_guesses, nsteps)
    
    return sampler 



def get_xy(l):
        
        hop = int((l-1)/2)
        
        dx = 0.010118516377207676
        #this is x[1] - x[0] from the PIV data 
    
        x = np.linspace(-dx*hop,dx*hop, l)
        y = np.linspace(-dx*hop,dx*hop, l)
        return x, y

def get_xy_rect(l1, l2):
        
        hop1 = int((l1-1)/2)
        hop2 = int((l2-1)/2)
        
        dx = 0.010118516377207676
        #this is x[1] - x[0] from the PIV data 
    
        y = np.linspace(-dx*hop1,dx*hop1, l1)
        x = np.linspace(-dx*hop2,dx*hop2, l2)
        return x, y
    

def stencil(i, j, S, hop):
        stemp =  S[i-hop:i+hop+1, j-hop:j+hop+1]
        #if stemp.shape != (2*hop+1, 3): print('stencil issue.',stemp.shape, S.shape)
        return stemp

    
#---------------------------------------------------------------------------------------
#GG on actual data 
#---------------------------------------------------------------------------------------
#to work with T1 and T2 and get the blob radius and location.

#from T1 - use getCents() to get T1 peaks
#from T2 and above use getProps() to get all peaks and radii
#from above use getPP to get the biggest in the BB

#this is where we implement MCMC

def getCents(S):
    from scipy.ndimage.measurements import center_of_mass, label, find_objects
    s_label, _ = label(S)    
    #plt.imshow(S)
    #plt.colorbar()

    cents  = np.array(center_of_mass(S, s_label, [i+1 for i in range(_)]))
    
    #sometimes when the blob is one dimensional, div by 0 occurs and the center of mass is NAN
    #to get rid of these errors, the following is done!
    a = np.array([np.inf])
    b = a.astype(int)[0]
    
    cents_ = cents.astype(int)
    cents1 = np.array([i for i in cents_ if i[0]!= b])
    
    return cents1
import numpy as np
import matplotlib.pyplot as plt

def arg_min(cents, ac):
    diff = [np.sum(np.abs(ac-i)) for i in cents]
    return np.argmin(diff)

def getProps_V2(GG, cents, ushape):
    '''
    
    GG is the T2 mat
    cents is the center of mass from T1
    ushape is the u mat
    
    returns the blob closest to the center
    
    '''
    l = max(ushape)
    ac = np.array([int(l/2),int(l/2)])
    #print(cents)
    mid_cent = [cents[arg_min(cents, ac)]]
    return getProps(GG, mid_cent)
    
    
def getProps_rect(GG, cents):
    '''
    GG - T2 matrix
    cents - the center of mass from T1 matrix 
    
    returns - v_dict 
    with all the vortex props - rad and loc of given cents
    indexed by 1,2,3.. in the order in which they appear 
    
    bb_dict has bb  size
    xy_dict has cent loc
    '''
    from scipy.ndimage.measurements import center_of_mass, label, find_objects
    lab, _ = label(GG)

    locs = find_objects(lab)
    #print(locs)
    bb_dict = {}
    xy_dict = {}
    v = 1
    for loc in locs:
        v_prop = []
        base =np.zeros(GG.shape)

        base[loc] = 1
        for cent in cents:
            if base[cent[0], cent[1]] == 1: 
                
                len_ = base[loc].shape
                
                bb_dict[v] = [int(round(len_[0]/2 + 1)), int(round(len_[1]/2 + 1))]
                xy_dict[v] = cent
                
                v+=1
                break 
    return bb_dict, xy_dict 

def getProps(GG, cents):
    '''
    GG - T2 matrix
    cents - the center of mass from T1 matrix 
    
    returns - v_dict 
    with all the vortex props - rad and loc of given cents
    indexed by 1,2,3.. in the order in which they appear 
    
    bb_dict has bb  size
    xy_dict has cent loc
    '''
    from scipy.ndimage.measurements import center_of_mass, label, find_objects
    lab, _ = label(GG)

    locs = find_objects(lab)
    #print(locs)
    bb_dict = {}
    xy_dict = {}
    v = 1
    for loc in locs:
        v_prop = []
        base =np.zeros(GG.shape)

        base[loc] = 1
        for cent in cents:
            if base[cent[0], cent[1]] == 1: 
                
                len_ = max(base[loc].shape)
                
                bb_dict[v] = int(round(len_/2 + 1))
                xy_dict[v] = cent
                
                v+=1
                break 
    return bb_dict, xy_dict 


def getPP(complete_dict):
    bb, xy = complete_dict
    rad = max(bb.values())
    vif = [i for i in bb.keys() if bb[i] == rad]
    #print(vif)
    return vif[0]
    
    
def Uinf():
    return 1.5511073539843

    
#---------------------------------------------------------------------------------------
#load the initial data;
#---------------------------------------------------------------------------------------

def init_data():
    """
    used to load initial data to begin the simulations. 
    returns X, Y, U, V, Swirl, Cond,Prof 
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    from importlib import reload
    from scipy import interpolate
    import PIVutils
    import PODutils

    loadFolder = "C:/Users/Kommalapati sahil/Desktop/owen/data/"

    saveFile = 'RNV45-thumbs-unified-BBauto2x.hdf5'
    imgFolder = loadFolder + saveFile[:-5]

    noEdge = True
    interpVecs = True

    import os
    if not os.path.exists(imgFolder):
        os.makedirs(imgFolder)

    #check 
    import sys
    print("current path :", sys.executable)


    
    X, Y, U, V, Swirl, Cond, Prof = PIVutils.loadDataset("C:/Users/Kommalapati sahil/Desktop/owen/"+'/RNV45-RI2.mat',\
                                                         ['X','Y','U','V','Swirl'],['Cond','Prof'],matlabData = True)
    
    #print("Uinf", Cond["Uinf"][0][0])

    X = X/Cond["delta"]
    Y = Y/Cond["delta"]

    NanLocs = np.isnan(Swirl)
    uSize = Swirl.shape
    scale = (X[1,-1]-X[1,1])/(uSize[1]-1)

    #interpolate
    missVecs = np.zeros(U.shape)
    missVecs[np.isnan(U)] = 1
    PercentMissing = np.zeros(U.shape[2])
    for i in range(U.shape[2]):
        PercentMissing[i] = missVecs[:,:,i].sum()/(U.shape[0]*U.shape[1])*100

    if interpVecs:
        for i in range(uSize[2]):
            #print(i)
            f = interpolate.interp2d(X[0,:], Y[:,0], U[:,:,i], kind='linear')
            U[:,:,i] = f(X[0,:],Y[:,0])
            f = interpolate.interp2d(X[0,:], Y[:,0], V[:,:,i], kind='linear')
            V[:,:,i] = f(X[0,:],Y[:,0])   
            f = interpolate.interp2d(X[0,:], Y[:,0], Swirl[:,:,i], kind='linear')
            Swirl[:,:,i] = f(X[0,:],Y[:,0]) 

    #remove background noise
    Noise = np.std(Swirl,axis=(2,1))
    Noise = np.std(Noise[-5:])
    print(Noise)

    SwirlFilt = Swirl.copy()    #think this should completely copy the list, allowing me to try things

    NoiseFilt = 20      # Filter at 20 times rms of freestream swirl 

    #Swirl must be above a certain background value or it is zeroed
    SwirlFilt[np.absolute(Swirl)<NoiseFilt*Noise] = 0

    #noramlize with std
    SwirlStd = np.std(Swirl,axis=(2,1))
    #print(SwirlStd)

    #Normalize field by the std of Swirl
    SwirlFilt = SwirlFilt/SwirlStd.reshape(uSize[0],1,1) #match the SwirlStd length (123) with the correct index in Swirl (also 123)

    SwirlFiltBackup = SwirlFilt.copy()

    # Create thresholded field
    SwirlFilt = SwirlFiltBackup.copy()    #think this should completely copy the list, allowing me to try things

    #Then only keep those locations where swirls is greater than Thresh*SwirlStd

    #Unified V 5 major change! *here*
    ThreshSTD = 0.6
    SwirlFilt[np.absolute(SwirlFilt)<ThreshSTD] = 0
    SwirlFiltPro = SwirlFilt.copy()
    SwirlFiltPro[SwirlFiltPro>0] = 0
    SwirlFiltRet = SwirlFilt.copy()
    SwirlFiltRet[SwirlFiltRet<0] = 0
    
    return X, Y, U, V,Swirl, Cond,Prof, SwirlFiltPro, SwirlFiltRet, SwirlFilt

#--------------------------------------------------------------------------------------------

def psf(G):
    import PIVutils
    PIVutils.plotScalarField(G)
    return 

