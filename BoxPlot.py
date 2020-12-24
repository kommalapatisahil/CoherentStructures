import matplotlib.pyplot as plt
import numpy as np
import h5py
from importlib import reload
from scipy import interpolate
import PIVutils
import PODutils
import grafteaux as G
import automateG as AG
from scipy.ndimage.measurements import find_objects, label, center_of_mass
import prom2d as PP
import time


X, Y, U, V,Swirl, Cond,Prof, SwirlFiltPro, SwirlFiltRet, SwirlFilt = G.init_data()
Umean = Prof['U']

xog = X[0,:]
yog = Y[:,0]

del_x = (xog[0]-xog[1])
ar_th  = del_x*del_x*np.pi*25

# Box plot function

#bw auxilary function 2
def boxx(x = np.arange(5, 15), y = np.arange(10,20)):
    plt.boxplot(x, vert = False,
                positions = [1],
                boxprops= dict(linewidth=2.0, color='black'),
                whiskerprops=dict(linestyle='-',linewidth=2.0,
                                  color='black'),
                medianprops={"linewidth" : 2.0,
                            "color":'red'},
                showfliers = False,
               showmeans = True)
    plt.boxplot(y, vert = False,
                positions = [2],
                boxprops= dict(linewidth=2.0, color='black'),
                whiskerprops=dict(linestyle='-',linewidth=2.0,
                                  color='black'),
                medianprops={"linewidth" : 2.0,
                            "color":'red',
                            'label':'Median'},
                showfliers = False,
               showmeans = True, 
               meanprops = {'label':'mean'})
    plt.grid(which = 'both')
    plt.yticks([1,2], ['MCMC', 'MINN'],fontSize = 15 )
    plt.title('Box and Whisker plot - Error', fontSize = 15)
    plt.xlabel(r'$\zeta$', fontSize  = 15)
    plt.legend()
    plt.show()

    return 0

#bw auxialary function 1
def errorLLV2(params, U, V, x, y):
    import numpy as np
    import PIVutils
    import PODutils
    from scipy.integrate import simps
    
    big_ = max(U.shape)
    a,b = U.shape
    
    if len(x) > len(y): x1 = y1 = x
    else: x1 = y1 = y
    #print(int((big_-1)/2), '<-------')
    #print(x1.shape, y1.shape)
    [U2_, V2_] = PIVutils.genHairpinField_V3(int((big_-1)/2),*params,x=x1,y=y1)
    
    [U2, V2] = PODutils.Reshaped(U2_, V2_, U.shape)
    
    
    W = np.zeros(U.shape)
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X-params[2], Y-params[2])
    
    #print(R.shape, W.shape, 'RW')
    #print(x.shape, y.shape, 'xy')
    
    W[R<=params[1]] = 1
    W[R>=params[1]] = params[1]/R[R>=params[1]]
    
    #Determine integral of weighting function
    #Area = simps(simps(W, y), x)
    Area = simps(simps(W, x), y)
    
    #Area = 1
    obj = W*((U-U2)**2 + (V-V2)**2)
    
    return np.sum(obj)/np.sum(W)

#call this to plot box and whiskers plot (bw plot)
#files = [F1, F2, F3, ....]
#Fi = [mc_sols, mn_sols, props]

def plot_bw(files, verbose = False):
    MC_errors = []
    MN_errors = []
    from tqdm import tqdm
    for i in tqdm(files):
        for j in range(len(i[0])):

            props = i[0][j]
            mc_sol = i[1][j]
            mn_sol = i[2][j]

            if mc_sol == -1:
                print('flag-1!'); continue
            #print(props, mc_sol, mn_sol)
            bbdims = props['bbdims']
            cent = props['cent']
            frame = props['frame']

            yc, xc  = cent
            b1, b2, b3, b4 = bbdims
            if (b1+b2+1 ) %2 == 0 : b2+= 1
            if (b3+b4+1 ) %2 == 0 : b4+= 1

            U1 = U[yc-b1: yc+b2+1, xc-b3:xc+b4+1, frame]
            V1 = V[yc-b1: yc+b2+1, xc-b3:xc+b4+1, frame]

            Un = (U1-Umean[yc])/Cond['Uinf']
            x1, y1 = G.get_xy_rect(*Un.shape)
            try:
                MC_errors.append(errorLLV2(mc_sol, Un, V1, x1, y1))
                MN_errors.append(errorLLV2(mn_sol, Un, V1, x1, y1))
            except:
                if verbose:
                    print('flag-2')
    
    boxx(np.array(MC_errors), np.array(MN_errors))
    return 0

#plot the PDF of solutions obtained from MCMC and Min.

def plotBigHist(mc_sols, mn_sols, bins_ = 10):
    names = ['Circulation Strength \u03C4'+r' [$\frac{L^2}{T}$]', 
             r'Radius, $\frac{r}{\delta}$ [] ',
             r'X-coordinate of the center, $\frac{x}{\delta}$ []',
             r'Y-coordinate of the center, $\frac{y}{\delta}$ [] ',
             r'Convective Velocity $\frac{V_{c}}{V_\infty}$ []']
    plt.figure(figsize = (15, 15))

    for i in range(len(names)):
        
        val1 = [j[i] for j in mc_sols]
        y1, x1_ = np.histogram(val1, bins = bins_, density = False)
        x1 = [(x1_[i]+x1_[i+1])/2 for  i in range(len(x1_)-1)]
        
        
        val2 = [j[i] for j in mn_sols]
        y2, x2_ = np.histogram(val2, bins = bins_, density = False)
        x2 = [(x2_[i]+x2_[i+1])/2 for  i in range(len(x2_)-1)]
        plt.subplot(3,3,i+1)
        #plt.hist(val2, bins = bins_, density = True)
        plt.plot(x1, y1, '-*r', label = 'MCMC', lineWidth = 3)
        plt.plot(x2, y2, '->g', label = 'Minn', lineWidth = 3)
    
        plt.grid()
        plt.legend()
        
        plt.xlabel(names[i], fontSize = 15)
        plt.ylabel('PDF', fontSize = 15)
        
    return 0 
