''' 
This is a set of utility funcitons useful for analysing POD data. Plotting and data reorganization functions
'''

#Plot 2D POD modes 
def plotPODmodes2D(X,Y,Umodes,Vmodes,plotModes,saveFolder = None):
    '''
    Plot 2D POD modes
    
    Inputs: 
    X - 2D array with columns constant
    Y - 2D array with rows constant
    Umodes - 3D array with Umodes.shape[2] = the total number of modes plotted
    Vmodes - 3D array with Umodes.shape[2] = the total number of modes plotted
    plotModes = a list of modes to be plotted
    
    Output:
    Plots the modes corresponding to plotModes
    '''

    import matplotlib.pyplot as plt
    
    #assert plotModes.max()<=Umodes.shape[2], 'You asked for more modes than were calculated'
    assert Umodes.shape[2]==Vmodes.shape[2], 'There are different numbers of U and V modes. Thats not right...'
    
    for i in plotModes:
        f, ax = plt.subplots(1,2)
        f.set_figwidth(18)
        im1 = ax[0].pcolor(X,Y,Umodes[:,:,i],cmap='RdBu_r');
        im2 = ax[1].pcolor(X,Y,Vmodes[:,:,i],cmap='RdBu_r');

        ax[0].set_title('U - #' + str(i+1))
        ax[0].set_aspect('equal')
        ax[0].set_xlim([X.min(),X.max()])
        ax[0].set_ylabel('$y/\delta$', fontsize=20)
        ax[0].set_xlabel('$x/\delta$', fontsize=20)
        ax[0].tick_params(axis='x', labelsize=12)
        ax[0].tick_params(axis='y', labelsize=12)
        
        ax[1].set_title('V - #' + str(i+1))
        ax[1].set_aspect('equal')
        ax[1].set_xlim([X.min(),X.max()])
        ax[1].set_ylabel('$y/\delta$', fontsize=20)
        ax[1].set_xlabel('$x/\delta$', fontsize=20)
        ax[1].tick_params(axis='x', labelsize=12)
        ax[1].tick_params(axis='y', labelsize=12)

        cbar1 = f.colorbar(im1,ax=ax[0])
        im1.set_clim(-1*max(map(abs,cbar1.get_clim())), max(map(abs,cbar1.get_clim()))) 
        cbar2 = f.colorbar(im2,ax=ax[1])
        im2.set_clim(-1*max(map(abs,cbar2.get_clim())), max(map(abs,cbar2.get_clim()))) 
        
        if saveFolder is not None:
            f.savefig(saveFolder + '/Mode' + str(i+1) + '.tif', transparent=True, bbox_inches='tight', pad_inches=0)
        
        del im1,im2,cbar1,cbar2
        
#Plot 2D POD modes 
def plotPODmodes2D_Presentation(X,Y,Umodes,Vmodes,plotModes,saveFolder = None):
    '''
    Plot 2D POD modes
    
    Inputs: 
    X - 2D array with columns constant
    Y - 2D array with rows constant
    Umodes - 3D array with Umodes.shape[2] = the total number of modes plotted
    Vmodes - 3D array with Umodes.shape[2] = the total number of modes plotted
    plotModes = a list of modes to be plotted
    
    Output:
    Plots the modes corresponding to plotModes
    '''

    import matplotlib.pyplot as plt
    import numpy as np
    
    #assert plotModes.max()<=Umodes.shape[2], 'You asked for more modes than were calculated'
    assert Umodes.shape[2]==Vmodes.shape[2], 'There are different numbers of U and V modes. Thats not right...'
    
    for i in plotModes:
        f, ax = plt.subplots(1,2)
        f.set_figwidth(18)
        im1 = ax[0].pcolor(X,Y,Umodes[:,:,i],cmap='RdBu_r');
        im2 = ax[1].pcolor(X,Y,Vmodes[:,:,i],cmap='RdBu_r');

        #ax[0].set_title('U - #' + str(i+1))
        ax[0].set_aspect('equal')
        ax[0].set_xlim([X.min(),X.max()])
        ax[0].set_ylabel('$y/\delta$', fontsize=30)
        ax[0].set_xlabel('$x/\delta$', fontsize=30)
        ax[0].tick_params(axis='x', labelsize=20)
        ax[0].tick_params(axis='y', labelsize=20)
        
        ax[0].set_xticks(np.arange(0.5, 3, 0.5))
        ax[0].set_yticks(np.arange(0, 1.5, 0.25))
        
        ax[0].set_ylim([0,1.25])
        
        #ax[1].set_title('V - #' + str(i+1))
        ax[1].set_aspect('equal')
        ax[1].set_xlim([X.min(),X.max()])
        ax[1].set_ylabel('$y/\delta$', fontsize=30)
        ax[1].set_xlabel('$x/\delta$', fontsize=30)
        ax[1].tick_params(axis='x', labelsize=20)
        ax[1].tick_params(axis='y', labelsize=20)
        
        ax[1].set_xticks(np.arange(0.5, 3, 0.5))
        ax[1].set_yticks(np.arange(0, 1.5, 0.25))
        
        ax[1].set_ylim([0,1.25])
        
        cbar1 = f.colorbar(im1,ax=ax[0])
        im1.set_clim(-1*max(map(abs,cbar1.get_clim())), max(map(abs,cbar1.get_clim()))) 
        cbar2 = f.colorbar(im2,ax=ax[1])
        im2.set_clim(-1*max(map(abs,cbar2.get_clim())), max(map(abs,cbar2.get_clim()))) 
        
        if saveFolder is not None:
            f.savefig(saveFolder + '/Mode' + str(i+1) + '.tif', transparent=True, bbox_inches='tight', pad_inches=0)
        
        del im1,im2,cbar1,cbar2
        
        

#Plot 3D POD modes 
def plotPODmodes3D(X,Y,Umodes,Vmodes,Wmodes,plotModes,saveFolder=None):
    '''
    Plot 2D POD modes
    
    Inputs: 
    X - 2D array with columns constant
    Y - 2D array with rows constant
    Umodes - 3D array with Umodes.shape[2] = the total number of modes plotted
    Vmodes - 3D array with Umodes.shape[2] = the total number of modes plotted
    Wmodes - 3D array with Umodes.shape[2] = the total number of modes plotted
    plotModes = a list of modes to be plotted
    
    Output:
    Plots the modes corresponding to plotModes
    '''

    import matplotlib.pyplot as plt
    
    #assert plotModes.max()<=Umodes.shape[2], 'You asked for more modes than were calculated'
    assert Umodes.shape[2]==Vmodes.shape[2], 'There are different numbers of U and V modes. Thats not right...'
    assert Umodes.shape[2]==Wmodes.shape[2], 'There are different numbers of U and W modes. Thats not right...'
    
    for i in plotModes:
        f, ax = plt.subplots(1,3)
        f.set_figwidth(18)
        im1 = ax[0].pcolor(X,Y,Umodes[:,:,i],cmap='RdBu_r');
        im2 = ax[1].pcolor(X,Y,Vmodes[:,:,i],cmap='RdBu_r');
        im3 = ax[2].pcolor(X,Y,Wmodes[:,:,i],cmap='RdBu_r');

        ax[0].set_title('U - #' + str(i+1))
        ax[0].set_aspect('equal')
        ax[0].set_xlim([X.min(),X.max()])
        ax[0].set_ylabel('y(m)')
        ax[0].set_xlabel('x(m)')
        
        ax[1].set_title('V - #' + str(i+1))
        ax[1].set_aspect('equal')
        ax[1].set_xlim([X.min(),X.max()])
        ax[1].set_ylabel('y(m)')
        ax[1].set_xlabel('x(m)')
        
        ax[2].set_title('W - #' + str(i+1))
        ax[2].set_aspect('equal')
        ax[2].set_xlim([X.min(),X.max()])
        ax[2].set_ylabel('y(m)')
        ax[2].set_xlabel('x(m)')

        cbar1 = f.colorbar(im1,ax=ax[0])
        im1.set_clim(-1*max(map(abs,cbar1.get_clim())), max(map(abs,cbar1.get_clim()))) 
        cbar2 = f.colorbar(im2,ax=ax[1])
        im2.set_clim(-1*max(map(abs,cbar2.get_clim())), max(map(abs,cbar2.get_clim()))) 
        cbar3 = f.colorbar(im3,ax=ax[2])
        im3.set_clim(-1*max(map(abs,cbar3.get_clim())), max(map(abs,cbar3.get_clim()))) 
        
        if saveFolder is not None:
            f.savefig(saveFolder + '/Mode' + str(i+1) + '.tif', transparent=True, bbox_inches='tight', pad_inches=0)
        
        del im1,im2,im3,cbar1,cbar2,cbar3

#Reorganize modes matrix so that the modes can be easily plotted
def reconstructPODmodes(modes,uSize,num_modes,numC):
    '''
    Reconstruct the mode shapes for three component single plane data
    
    Inputs: 
    modes - outout from mr.compute_POD_matrices_snaps_method
    uSize - size of original velocity dataset
    num_modes - number of modes calculated by mr.compute_POD_matrices_snaps_method
    numC - number of velocity components
    
    Output:
    Umodes, Vmodes and optionally Wmodes
    '''       
    import numpy as np
    
    #Rearrange mode data to get mode fields
    modeSize = modes.shape
    Umodes = modes[0:uSize[0]*uSize[1],:];
    Umodes2 = np.zeros((uSize[0],uSize[1],num_modes))
    
    if numC >= 2:
        Vmodes = modes[uSize[0]*uSize[1]:2*uSize[0]*uSize[1],:];
        Vmodes2 = np.zeros((uSize[0],uSize[1],num_modes))
    if numC >= 3:
        Wmodes = modes[2*uSize[0]*uSize[1]:modeSize[0]+1,:];
        Wmodes2 = np.zeros((uSize[0],uSize[1],num_modes))

    Umodes.shape
    

    for i in range(num_modes):
        #i=1
        Umodes2[:,:,i] = np.reshape(Umodes[:,i],(uSize[0],uSize[1]))
        if numC >=2:
            Vmodes2[:,:,i] = np.reshape(Vmodes[:,i],(uSize[0],uSize[1]))
        if numC >=3:
            Wmodes2[:,:,i] = np.reshape(Vmodes[:,i],(uSize[0],uSize[1]))        

        #Umodes.shape
        #uSize[0]*uSize[1]
    if numC == 1:
        return [Umodes2]
    elif numC == 2:
        return [Umodes2, Vmodes2]
    elif numC == 3:
        return [Umodes2, Vmodes2, Wmodes2]
        
#Plot heatmaps of POD coefficients
def plotPODcoeff(C,modes,num_bins,bound=None,logscale=None,saveFolder=None):
    '''
    Reconstruct the mode shapes for three component single plane data
    
    Inputs: 
    C - matrix of coefficients (mode number, coefficent for each frame) 
    modes - indices of modes to be plotted 
    num_bins - size of bins to be plotted. Passed to hexbin
    bound - the axis bound. If none taken to be max coefficient
    logscale - describing whether or not to do the heatmap in a log scale
    
    Output:
    plots a grid of hexbin plots for each mode
    '''       
    import numpy as np
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    
    if bound == None:
        bound = round(np.max(np.absolute(C)))
    
    xedges = np.linspace(-1*bound, bound, num=num_bins)
    yedges = xedges;
    bound = 0.5*(xedges[1]+xedges[2]);
    
    Z, xedges, yedges = np.histogram2d(C[0], C[1], bins=(xedges, yedges))
    xv, yv = np.meshgrid(0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1]))

    fig, axs = plt.subplots(ncols=len(modes)-1,nrows=len(modes)-1,figsize=(9, 12))
    fig.subplots_adjust(hspace=0.01, left=0.01, right=1)
    
    #print(axs.shape)

    for i in range(len(modes)-1):
        for j in range(len(modes)-1):
            ax = axs[i,j]
            if j>=i:
                Z, xedges, yedges = np.histogram2d(C[i],C[j+1], bins=(xedges, yedges))
                if logscale == None:
                    hb = ax.pcolor(xv, yv, Z, cmap='hsv')
                else:
                    hb = ax.pcolor(xv, yv, np.log(Z+1), cmap='hsv')
                    
                ax.plot([-1*bound, bound],[0, 0],'--k')
                ax.plot([0, 0],[-1*bound, bound],'--k')
                
                if i == 0:
                    ax.set_xlabel('C{0}'.format(j+2))
                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position("top")
                    ax.tick_params(axis='x', labelsize=7)
                else:
                    ax.set_xticklabels([])

                    
                if j == len(modes)-2:
                    ax.yaxis.tick_right()
                    ax.set_ylabel('C{0}'.format(i+1))
                    ax.yaxis.set_label_position("right")
                    ax.tick_params(axis='y', labelsize=7)
                else:
                    ax.set_yticklabels([])
                    
                ax.set_xlim(bound,-1*bound)
                ax.set_ylim(bound,-1*bound)
                
                ax.set_aspect("equal")
                ax.set_adjustable("box-forced")

                #fig = plt.figure(figsize = [8,3])
                #hb = ax.hexbin(C[0], C[1], gridsize=10, cmap='OrRd')
                #plt.axis([-1*bound, bound, -1*bound, bound])
                #plt.axis('scaled')
                #cb = fig.colorbar(hb, ax=ax)
                #cb.set_label('counts')
                #cb.set_label('log10(N)')
            else:
                ax.axis('off')
                
    if saveFolder is not None:
        fig.savefig(saveFolder, transparent=True, bbox_inches='tight', pad_inches=0)
           
        
#Plot heatmaps of POD coefficients
def plotYposPODcoeff(ypos,C,modes,num_bins,bound=None,logscale=None,saveFolder=None):
    '''
    Reconstruct the mode shapes for three component single plane data
    
    Inputs: 
    ypos - wall-normal position of each thumbnail. [0 1]
    C - matrix of coefficients (mode number, coefficent for each frame) 
    modes - indices of modes to be plotted 
    num_bins - size of bins to be plotted. Passed to hexbin
    bound - the axis bound. If none taken to be max coefficient
    logscale - describing whether or not to do the heatmap in a log scale
    
    Output:
    plots a grid of hexbin plots for each mode
    '''       
    import numpy as np
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    
    if bound == None:
        bound = round(np.max(np.absolute(C)))
    
    #xedges = np.linspace(0, 1, num=num_bins);
    xedges = ypos
    xedges = np.concatenate([[xedges[0]-(xedges[1]-xedges[0])],xedges, [xedges[-1]+xedges[-1]-xedges[-2]]])
    #print(xedges)
    yedges = np.linspace(-1*bound, bound, num=num_bins);
    #bound = 0.5*(xedges[1]+xedges[2]);
    
    Z, xedges, yedges = np.histogram2d(C[0],C[1], bins=(xedges, yedges))
    xv, yv = np.meshgrid(0.5*(xedges[1:]+xedges[:-1]),0.5*(yedges[1:]+yedges[:-1]))
    
    fig, axs = plt.subplots(nrows=len(modes),figsize=(3, 3*len(modes)))
    fig.subplots_adjust(hspace=0.1, left=0.1, right=1)
    
    #print(axs.shape)

    for i in range(len(modes)):
        ax = axs[i]

        Z, xedges, yedges = np.histogram2d(C[0],C[i+1], bins=(xedges, yedges))
        Z = Z.T
        if logscale == None:
            hb = ax.pcolor(xv, yv, Z, cmap='hsv')
        else:
            hb = ax.pcolor(xv, yv, np.log(Z+1), cmap='hsv')

        ax.plot([-1*bound, bound],[0, 0],'--k')

        if i == len(modes)-1:
            ax.set_xlabel('$y/\delta$')
            ax.tick_params(axis='x', labelsize=7)
        else:
            ax.set_xticklabels([])

        ax.set_ylabel('C{0}'.format(i+1))
        ax.tick_params(axis='y', labelsize=7)

        ax.set_xlim(0,max(ypos))
        ax.set_ylim(-1*bound,bound)
        
    if saveFolder is not None:
        fig.savefig(saveFolder, transparent=True, bbox_inches='tight', pad_inches=0)

#Plot heatmaps of POD coefficients
def plotLLEscatter(C,ypos,St,modes,bound=None,thumb_frac=None,VecDist=None,saveFolder=None):
    '''
    Reconstruct the mode shapes for three component single plane data
    
    Inputs: 
    C - matrix of coefficients (mode number, coefficent for each frame) 
    ypos - distance from the wall for each thumbnail
    St - thumbnail data (not necessarily velocity, should use swirl)
    modes - indices of modes to be plotted 
    bound - the axis bound. If none taken to be max coefficient
    thumb_frac - fraction of thumbnails to show
    VecDist - length of each thumbnail vector pointing to coefficient location
    
    Output:
    plots a grid of hexbin plots for each mode
    '''       
    import numpy as np
    #from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    from matplotlib import offsetbox,colors
    
    if bound == None:
        bound = round(np.max(np.absolute(C)))
    if thumb_frac == None:
        thumb_frac = 0.5
    if VecDist == None:
        VecDist = 0.05

    fig, axs = plt.subplots(ncols=len(modes)-1,nrows=len(modes)-1,figsize=(9, 12))
    fig.subplots_adjust(hspace=0.01, left=0.01, right=1)
    
    colorize = dict(c=ypos, cmap=plt.cm.get_cmap('rainbow', 100))
    cmap='RdBu_r'
    
    C2 = C.copy().T
    
    for i in range(len(modes)-1):
        for j in range(len(modes)-1):
            ax = axs[i,j]
            if j>=i:
                hb = ax.scatter(C[i], C[j+1],s=2, facecolor='0.5', lw = 0, **colorize)
                    
                ax.plot([-1*bound, bound],[0, 0],'--k')
                ax.plot([0, 0],[-1*bound, bound],'--k')
               
                if i == 0:
                    ax.set_xlabel('C{0}'.format(j+2))
                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position("top")
                    ax.tick_params(axis='x', labelsize=7)
                else:
                    ax.set_xticklabels([])

                    
                if j == len(modes)-2:
                    ax.yaxis.tick_right()
                    ax.set_ylabel('C{0}'.format(i+1))
                    ax.yaxis.set_label_position("right")
                    ax.tick_params(axis='y', labelsize=7)
                else:
                    ax.set_yticklabels([])
                    
                ax.set_xlim(bound,-1*bound)
                ax.set_ylim(-1*bound,bound)
                
                ax.set_aspect("equal")
                ax.set_adjustable("box-forced")
                
                
                min_dist_2 = (thumb_frac * max(C2.max(0) - C2.min(0))) ** 2
                shown_images = np.array([2 * C2.max(0)])
                for k in range(C2.shape[0]):
                    dist = np.sum((C2[k] - shown_images) ** 2, 1)
                    if np.min(dist) < min_dist_2:
                        # don't show points that are too close
                        continue
                    shown_images = np.vstack([shown_images, C2[k]])

                    vecNorm = (C2[k,i]**2 + C2[k,j+1]**2)**0.5
                    imagebox = offsetbox.AnnotationBbox(
                        offsetbox.OffsetImage(St[:,:,k], cmap=cmap, norm=colors.Normalize(-50,50),zoom=1.5),
                                              xybox=VecDist*C2[k,[i,j+1]]/vecNorm+C2[k,[i,j+1]],xy=C2[k,[i,j+1]],       arrowprops=dict(arrowstyle="->"))
                    ax.add_artist(imagebox)
            else:
                ax.axis('off')
                
    if saveFolder is not None:
        fig.savefig(saveFolder, transparent=True, bbox_inches='tight', pad_inches=0)

def minfuncVecField(params, U, V, x, y):
    import numpy as np
    import PIVutils
    
    assert U.shape[0] == U.shape[1], 'Data must be a square matrix.'
    assert U.shape    == V.shape   , 'U and V fields must be the same size'

    [U2, V2] = PIVutils.genHairpinField(int((U.shape[0]-1)/2),*params,x=x,y=y)
    
    return np.sum(((U-U2)**2 + (V-V2)**2))

def minfuncVecField_V2(params, U, V, x, y):
    import numpy as np
    import PIVutils
    
    assert U.shape[0] == U.shape[1], 'Data must be a square matrix.'
    assert U.shape    == V.shape   , 'U and V fields must be the same size'

    [U2, V2] = PIVutils.genHairpinField_V2(int((U.shape[0]-1)/2),*params,x=x,y=y)
    
    return np.sum(((U-U2)**2 + (V-V2)**2))

def minfuncVecField_V3_og(params, U, V, x, y):
    import numpy as np
    import PIVutils
    from scipy.integrate import simps
    
    assert U.shape[0] == U.shape[1], 'Data must be a square matrix.'
    assert U.shape    == V.shape   , 'U and V fields must be the same size'

    [U2, V2] = PIVutils.genHairpinField_V3(int((U.shape[0]-1)/2),*params,x=x,y=y)
    
    BoxSize = int((U.shape[0]-1)/2)
    W = np.zeros([2*BoxSize+1,2*BoxSize+1])
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X-params[2], Y-params[2])
    W[R<=params[1]] = 1
    W[R>=params[1]] = params[1]/R[R>=params[1]]
    
    #a = W*((U-U2)**2 + (V-V2)**2)
    #print(a.shape)
    
    #Determine integral of weighting function
    Area = simps(simps(W, y), x)
    #Area = 1
    return np.sum(W*((U-U2)**2 + (V-V2)**2))/Area

def minfuncVecField_V3_10r(params, U, V, x, y):
    import numpy as np
    import PIVutils
    from scipy.integrate import simps
    
    assert U.shape[0] == U.shape[1], 'Data must be a square matrix.'
    assert U.shape    == V.shape   , 'U and V fields must be the same size'

    [U2, V2] = PIVutils.genHairpinField_V3(int((U.shape[0]-1)/2),*params,x=x,y=y)
    
    BoxSize = int((U.shape[0]-1)/2)
    W = np.zeros([2*BoxSize+1,2*BoxSize+1])
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X-params[2], Y-params[2])
    W[R<=params[1]] = 1
    W[R>=params[1]] = params[1]/R[R>=params[1]]
    
    #a = W*((U-U2)**2 + (V-V2)**2)
    #print(a.shape)
    
    #Determine integral of weighting function
    Area = simps(simps(W, y), x)
    #Area = 1
    obj = W*((U-U2)**2 + (V-V2)**2)
    
    #Area_lite = np.mean(W)*(x[-1]-x[0])*(y[-1]-y[0])
    #return np.sum(obj)/np.mean(W) #okay but not acceptable
    #return np.sum(obj)/np.mean(obj) #doesn't converge at all
    return np.sum(obj)/Area
    

def minfuncVecField_V3_12r(params, U, V, x, y):
    import numpy as np
    import PIVutils
    from scipy.integrate import simps
    
    assert U.shape[0] == U.shape[1], 'Data must be a square matrix.'
    assert U.shape    == V.shape   , 'U and V fields must be the same size'

    [U2, V2] = PIVutils.genHairpinField_V3(int((U.shape[0]-1)/2),*params,x=x,y=y)
    
    BoxSize = int((U.shape[0]-1)/2)
    W = np.zeros([2*BoxSize+1,2*BoxSize+1])
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X-params[2], Y-params[2])
    #WF2#W[R<=2*params[1]] = 1
    #WF2#W[R>=2*params[1]] = params[1]/R[R>=2*params[1]]
    
    #WF3#W = params[1]/R
    
    #WF4 
    W[R<=1.2*params[1]] = 1
    W[R>=1.2*params[1]] = params[1]/R[R>=1.2*params[1]]
    
    
    #a = W*((U-U2)**2 + (V-V2)**2)
    #print(a.shape)
    
    #Determine integral of weighting function
    Area = simps(simps(W, y), x)
    #Area = 1
    obj = W*((U-U2)**2 + (V-V2)**2)
    
    #Area_lite = np.mean(W)*(x[-1]-x[0])*(y[-1]-y[0])
    #return np.sum(obj)/np.mean(W) #okay but not acceptable
    #return np.sum(obj)/np.mean(obj) #doesn't converge at all
    return np.sum(obj)/Area
def minfuncVecField_V3_14r(params, U, V, x, y):
    import numpy as np
    import PIVutils
    from scipy.integrate import simps
    
    assert U.shape[0] == U.shape[1], 'Data must be a square matrix.'
    assert U.shape    == V.shape   , 'U and V fields must be the same size'

    [U2, V2] = PIVutils.genHairpinField_V3(int((U.shape[0]-1)/2),*params,x=x,y=y)
    
    BoxSize = int((U.shape[0]-1)/2)
    W = np.zeros([2*BoxSize+1,2*BoxSize+1])
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X-params[2], Y-params[2])
    #WF2#W[R<=2*params[1]] = 1
    #WF2#W[R>=2*params[1]] = params[1]/R[R>=2*params[1]]
    
    #WF3#W = params[1]/R
    
    #WF4 
    W[R<=1.4*params[1]] = 1
    W[R>=1.4*params[1]] = params[1]/R[R>=1.4*params[1]]
    
    
    #a = W*((U-U2)**2 + (V-V2)**2)
    #print(a.shape)
    
    #Determine integral of weighting function
    Area = simps(simps(W, y), x)
    #Area = 1
    obj = W*((U-U2)**2 + (V-V2)**2)
    
    #Area_lite = np.mean(W)*(x[-1]-x[0])*(y[-1]-y[0])
    #return np.sum(obj)/np.mean(W) #okay but not acceptable
    #return np.sum(obj)/np.mean(obj) #doesn't converge at all
    return np.sum(obj)/Area
def minfuncVecField_V3_16r(params, U, V, x, y):
    import numpy as np
    import PIVutils
    from scipy.integrate import simps
    
    assert U.shape[0] == U.shape[1], 'Data must be a square matrix.'
    assert U.shape    == V.shape   , 'U and V fields must be the same size'

    [U2, V2] = PIVutils.genHairpinField_V3(int((U.shape[0]-1)/2),*params,x=x,y=y)
    
    BoxSize = int((U.shape[0]-1)/2)
    W = np.zeros([2*BoxSize+1,2*BoxSize+1])
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X-params[2], Y-params[2])
    #WF2#W[R<=2*params[1]] = 1
    #WF2#W[R>=2*params[1]] = params[1]/R[R>=2*params[1]]
    
    #WF3#W = params[1]/R
    
    #WF4 
    W[R<=1.6*params[1]] = 1
    W[R>=1.6*params[1]] = params[1]/R[R>=1.6*params[1]]
    
    
    #a = W*((U-U2)**2 + (V-V2)**2)
    #print(a.shape)
    
    #Determine integral of weighting function
    Area = simps(simps(W, y), x)
    #Area = 1
    obj = W*((U-U2)**2 + (V-V2)**2)
    
    #Area_lite = np.mean(W)*(x[-1]-x[0])*(y[-1]-y[0])
    #return np.sum(obj)/np.mean(W) #okay but not acceptable
    #return np.sum(obj)/np.mean(obj) #doesn't converge at all
    return np.sum(obj)/Area

def minfuncVecField_V3_08r(params, U, V, x, y):
    import numpy as np
    import PIVutils
    from scipy.integrate import simps
    
    assert U.shape[0] == U.shape[1], 'Data must be a square matrix.'
    assert U.shape    == V.shape   , 'U and V fields must be the same size'

    [U2, V2] = PIVutils.genHairpinField_V3(int((U.shape[0]-1)/2),*params,x=x,y=y)
    
    BoxSize = int((U.shape[0]-1)/2)
    W = np.zeros([2*BoxSize+1,2*BoxSize+1])
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X-params[2], Y-params[2])
    #WF2#W[R<=2*params[1]] = 1
    #WF2#W[R>=2*params[1]] = params[1]/R[R>=2*params[1]]
    
    #WF3#W = params[1]/R
    
    #WF4 
    W[R<=0.8*params[1]] = 1
    W[R>=0.8*params[1]] = params[1]/R[R>=0.8*params[1]]
    
    
    #a = W*((U-U2)**2 + (V-V2)**2)
    #print(a.shape)
    
    #Determine integral of weighting function
    Area = simps(simps(W, y), x)
    #Area = 1
    obj = W*((U-U2)**2 + (V-V2)**2)
    
    #Area_lite = np.mean(W)*(x[-1]-x[0])*(y[-1]-y[0])
    #return np.sum(obj)/np.mean(W) #okay but not acceptable
    #return np.sum(obj)/np.mean(obj) #doesn't converge at all
    return np.sum(obj)/Area

def minfuncVecField_V3_18r(params, U, V, x, y):
    import numpy as np
    import PIVutils
    from scipy.integrate import simps
    
    assert U.shape[0] == U.shape[1], 'Data must be a square matrix.'
    assert U.shape    == V.shape   , 'U and V fields must be the same size'

    [U2, V2] = PIVutils.genHairpinField_V3(int((U.shape[0]-1)/2),*params,x=x,y=y)
    
    BoxSize = int((U.shape[0]-1)/2)
    W = np.zeros([2*BoxSize+1,2*BoxSize+1])
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X-params[2], Y-params[2])
    #WF2#W[R<=2*params[1]] = 1
    #WF2#W[R>=2*params[1]] = params[1]/R[R>=2*params[1]]
    
    #WF3#W = params[1]/R
    
    #WF4 
    W[R<=1.8*params[1]] = 1
    W[R>=1.8*params[1]] = params[1]/R[R>=1.8*params[1]]
    
    
    #a = W*((U-U2)**2 + (V-V2)**2)
    #print(a.shape)
    
    #Determine integral of weighting function
    Area = simps(simps(W, y), x)
    #Area = 1
    obj = W*((U-U2)**2 + (V-V2)**2)
    
    #Area_lite = np.mean(W)*(x[-1]-x[0])*(y[-1]-y[0])
    #return np.sum(obj)/np.mean(W) #okay but not acceptable
    #return np.sum(obj)/np.mean(obj) #doesn't converge at all
    return np.sum(obj)/Area

def minfuncVecField_V3_24r(params, U, V, x, y):
    import numpy as np
    import PIVutils
    from scipy.integrate import simps
    
    assert U.shape[0] == U.shape[1], 'Data must be a square matrix.'
    assert U.shape    == V.shape   , 'U and V fields must be the same size'

    [U2, V2] = PIVutils.genHairpinField_V3(int((U.shape[0]-1)/2),*params,x=x,y=y)
    
    BoxSize = int((U.shape[0]-1)/2)
    W = np.zeros([2*BoxSize+1,2*BoxSize+1])
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X-params[2], Y-params[2])
    #WF2#W[R<=2*params[1]] = 1
    #WF2#W[R>=2*params[1]] = params[1]/R[R>=2*params[1]]
    
    #WF3#W = params[1]/R
    
    #WF4 
    W[R<=2.4*params[1]] = 1
    W[R>=2.4*params[1]] = params[1]/R[R>=2.4*params[1]]
    
    
    #a = W*((U-U2)**2 + (V-V2)**2)
    #print(a.shape)
    
    #Determine integral of weighting function
    Area = simps(simps(W, y), x)
    #Area = 1
    obj = W*((U-U2)**2 + (V-V2)**2)
    
    #Area_lite = np.mean(W)*(x[-1]-x[0])*(y[-1]-y[0])
    #return np.sum(obj)/np.mean(W) #okay but not acceptable
    #return np.sum(obj)/np.mean(obj) #doesn't converge at all
    return np.sum(obj)/Area


def minfuncVecField_V3_20r(params, U, V, x, y):
    import numpy as np
    import PIVutils
    from scipy.integrate import simps
    
    assert U.shape[0] == U.shape[1], 'Data must be a square matrix.'
    assert U.shape    == V.shape   , 'U and V fields must be the same size'

    [U2, V2] = PIVutils.genHairpinField_V3(int((U.shape[0]-1)/2),*params,x=x,y=y)
    
    BoxSize = int((U.shape[0]-1)/2)
    W = np.zeros([2*BoxSize+1,2*BoxSize+1])
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X-params[2], Y-params[2])
    #WF2#W[R<=2*params[1]] = 1
    #WF2#W[R>=2*params[1]] = params[1]/R[R>=2*params[1]]
    
    #WF3#W = params[1]/R
    
    #WF4 
    W[R<=2*params[1]] = 1
    W[R>=2*params[1]] = params[1]/R[R>=2*params[1]]
    
    
    #a = W*((U-U2)**2 + (V-V2)**2)
    #print(a.shape)
    
    #Determine integral of weighting function
    Area = simps(simps(W, y), x)
    #Area = 1
    obj = W*((U-U2)**2 + (V-V2)**2)
    
    #Area_lite = np.mean(W)*(x[-1]-x[0])*(y[-1]-y[0])
    #return np.sum(obj)/np.mean(W) #okay but not acceptable
    #return np.sum(obj)/np.mean(obj) #doesn't converge at all
    return np.sum(obj)/Area
    

def minfuncVecField_V4(params, U, V, x, y):
        
    #weighting function
    # w (R<r) = 1
    # w (R>r) = abr/(x^2 + a^2)
    #serpentine with a radius a = (R)
    
    import numpy as np
    import PIVutils
    from scipy.integrate import simps
    import math
    
    assert U.shape[0] == U.shape[1], 'Data must be a square matrix.'
    assert U.shape    == V.shape   , 'U and V fields must be the same size'

    [U2, V2] = PIVutils.genHairpinField_V3(int((U.shape[0]-1)/2),*params,x=x,y=y)
    
    BoxSize = int((U.shape[0]-1)/2)
    W = np.zeros([2*BoxSize+1,2*BoxSize+1])
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X-params[2], Y-params[2])
    
    k = 1
    r = k*params[1]
    
    
    W[R<=r] = 1
    W[R>=r] = r*R[R>=r]/(R[R>=r]**2 + r**2)
    #serpentine 
    
    #a = W*((U-U2)**2 + (V-V2)**2)
    #print(a.shape)
    
    #Determine integral of weighting function
    Area = simps(simps(W, y), x)
        
    return np.sum(W*((U-U2)**2 + (V-V2)**2))/Area

def minfuncVecField_V5(params, U, V, x, y):
        
    #weighting function
    # w (R<r) = 1
    # w (R>r) = abr/(x^2 + a^2)
    #serpentine with a radius a = (R)
    
    import numpy as np
    import PIVutils
    from scipy.integrate import simps
    import math
    
    assert U.shape[0] == U.shape[1], 'Data must be a square matrix.'
    assert U.shape    == V.shape   , 'U and V fields must be the same size'

    [U2, V2] = PIVutils.genHairpinField_V3(int((U.shape[0]-1)/2),*params,x=x,y=y)
    
    BoxSize = int((U.shape[0]-1)/2)
    W = np.zeros([2*BoxSize+1,2*BoxSize+1])
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X-params[2], Y-params[2])
    
    k = 1
    r = k*params[1]
    
    R = np.array(R)
    W[R<=r] = 1
    W[R>=r] = np.sqrt(R[R>=r]**2 + r**2)+ r*R[R>=r]/(R[R>=r]**2 + r**2)
    #serpentine + circle 
    
    #a = W*((U-U2)**2 + (V-V2)**2)
    #print(a.shape)
    
    #Determine integral of weighting function
    Area = simps(simps(W, y), x)
        
    return np.sum(W*((U-U2)**2 + (V-V2)**2))/Area



def log_prior(params,bounds):
    
    import numpy as np
    
    for i in range(len(bounds)):
        if bounds[i][0] is not None:
            if params[i] < bounds[i][0]:
                return -np.inf
        if bounds[i][1] is not None:
            if params[i] > bounds[i][1]:
                return -np.inf
                        
    return 0

def log_prior_V3(params,bounds):
    
    import numpy as np
    
    for i in range(len(bounds)):
        if i != 2 and i !=3:
            if bounds[i][0] is not None:
                if params[i] < bounds[i][0]:
                    return -np.inf
            if bounds[i][1] is not None:
                if params[i] > bounds[i][1]:
                    return -np.inf
    xc = params[2]
    yc = params[3]
    rad = (bounds[2][1]/2)
    return -(xc**2+yc**2)/(2*rad**2) or window/6
            
    return 0

def log_posterior(params, U, V, x, y, bounds):
    import numpy as np
    #params[0:8] = np.absolute(params[0:8])
    value = log_prior(params,bounds)
    if np.isneginf(value):
        return -np.inf
    else:
        return value + -1*minfuncVecField(params, U, V, x, y)
    
def log_posterior_V2(params, U, V, x, y, bounds):
    import numpy as np
    #params[0:8] = np.absolute(params[0:8])
    value = log_prior(params,bounds)
    if np.isneginf(value):
        return -np.inf
    else:
        return value + -1*minfuncVecField_V2(params, U, V, x, y)
    
def log_posterior_V3(params, U, V, x, y, bounds):
    import numpy as np
    #params[0:8] = np.absolute(params[0:8])
    value = log_prior_V3(params,bounds)
    if np.isneginf(value):
        return -np.inf
    else:
        return value + -1*minfuncVecField_V3(params, U, V, x, y)
def log_posterior_V3_10r(params, U, V, x, y, bounds):
    import numpy as np
    #params[0:8] = np.absolute(params[0:8])
    value = log_prior_V3(params,bounds)
    if np.isneginf(value):
        return -np.inf
    else:
        return value + -1*minfuncVecField_V3_10r(params, U, V, x, y)
def log_posterior_V3_08r(params, U, V, x, y, bounds):
    import numpy as np
    #params[0:8] = np.absolute(params[0:8])
    value = log_prior_V3(params,bounds)
    if np.isneginf(value):
        return -np.inf
    else:
        return value + -1*minfuncVecField_V3_08r(params, U, V, x, y)

def log_posterior_V3_12r(params, U, V, x, y, bounds):
    import numpy as np
    #params[0:8] = np.absolute(params[0:8])
    value = log_prior_V3(params,bounds)
    if np.isneginf(value):
        return -np.inf
    else:
        return value + -1*minfuncVecField_V3_12r(params, U, V, x, y)
def log_posterior_V3_14r(params, U, V, x, y, bounds):
    import numpy as np
    #params[0:8] = np.absolute(params[0:8])
    value = log_prior_V3(params,bounds)
    if np.isneginf(value):
        return -np.inf
    else:
        return value + -1*minfuncVecField_V3_14r(params, U, V, x, y)
def log_posterior_V3_16r(params, U, V, x, y, bounds):
    import numpy as np
    #params[0:8] = np.absolute(params[0:8])
    value = log_prior_V3(params,bounds)
    if np.isneginf(value):
        return -np.inf
    else:
        return value + -1*minfuncVecField_V3_16r(params, U, V, x, y)
def log_posterior_V3_18r(params, U, V, x, y, bounds):
    import numpy as np
    #params[0:8] = np.absolute(params[0:8])
    value = log_prior_V3(params,bounds)
    if np.isneginf(value):
        return -np.inf
    else:
        return value + -1*minfuncVecField_V3_18r(params, U, V, x, y)
def log_posterior_V3_20r(params, U, V, x, y, bounds):
    import numpy as np
    #params[0:8] = np.absolute(params[0:8])
    value = log_prior_V3(params,bounds)
    if np.isneginf(value):
        return -np.inf
    else:
        return value + -1*minfuncVecField_V3_20r(params, U, V, x, y)

def log_posterior_V3_24r(params, U, V, x, y, bounds):
    import numpy as np
    #params[0:8] = np.absolute(params[0:8])
    value = log_prior_V3(params,bounds)
    if np.isneginf(value):
        return -np.inf
    else:
        return value + -1*minfuncVecField_V3_24r(params, U, V, x, y)

    
def log_posterior_V4(params, U, V, x, y, bounds):
    import numpy as np
    #params[0:8] = np.absolute(params[0:8])
    value = log_prior_V3(params,bounds)
    if np.isneginf(value):
        return -np.inf
    else:
        return value + -1*minfuncVecField_V4(params, U, V, x, y)
    

def log_posterior_V5(params, U, V, x, y, bounds):
    import numpy as np
    #params[0:8] = np.absolute(params[0:8])
    value = log_prior_V3(params,bounds)
    if np.isneginf(value):
        return -np.inf
    else:
        return value + -1*minfuncVecField_V5(params, U, V, x, y)
    
def resh(b, sh):
    #make sure that the sh is always 
    # (odd, odd)
    i, j = sh
    if i == b.shape[0]:
        return b[:,int(b.shape[0]/2-j/2): int(b.shape[0]/2+j/2)]
    return b[int((b.shape[0]-i)/2): int((b.shape[0]+i)/2), :]

def Reshaped(U, V, sh):
    U1 = resh(U, sh)
    V1 = resh(V, sh)
    return [U1, V1]
    
    
#----------------------------------------------------------------------------
#rectangular stuff
def minfuncVecField10r_rect(params, U, V, x, y):
    import numpy as np
    import PIVutils
    from scipy.integrate import simps
    
    big_ = max(U.shape)
    a,b = U.shape
    
    if len(x) > len(y): x1 = y1 = x
    else: x1 = y1 = y
        
    [U2_, V2_] = PIVutils.genHairpinField_V3(int((big_-1)/2),*params,x=x1,y=y1)
    
    [U2, V2] = Reshaped(U2_, V2_, U.shape)
    
    
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
    
    return np.sum(obj)/Area


def log_posterior10r_rect(params, U, V, x, y, bounds):
    import numpy as np
    #params[0:8] = np.absolute(params[0:8])
    value = log_prior_V3(params,bounds)
    if np.isneginf(value):
        return -np.inf
    else:
        return value + -1*minfuncVecField10r_rect(params, U, V, x, y)
    
    

