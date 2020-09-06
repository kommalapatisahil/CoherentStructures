'''
This is a set of functions for dealing with PIV data
Most functions are for 2D data. 
'''

#Import matlab PIV data
def loadDataset(path,mats,structs,matlabData = None):
    '''
    Import a matlab data file
    
    Inputs: 
    path - Path to the matlab file
    mats - names of the matlab matrices to be imported.
    structs - names of the structures to be imported as dictionaries
    matlabData - flag to transpose matrices if data comes from matlab
    
    Output:
    Each of the matrices and structures in the order you specified in the input
    '''
    
    import numpy as np
    import h5py
    
    if matlabData is None:
        matlabData = False

    f = h5py.File(path)
    
    print(list(f.keys()))
    
    Temp = np.asarray(['Swirl'])
    
    ret = []

    for i in mats:
        #print(i)
        Temp = np.asarray(f[i])
        if matlabData:
            if Temp.ndim == 2:
                Temp = np.transpose(Temp,(1,0))
            elif Temp.ndim == 3:
                Temp = np.transpose(Temp,(2,1,0))
        ret.append(Temp)
        del Temp

    for i in structs:
        #print(i)
        TempS = {k : f[i][k].value       #Generate a dictionary linking all values in cond with their names
             for k in f[i].keys()}
        ret.append(TempS)
        del TempS
        
    f.close()
        
    return ret

#Save data to an HDF5 file
def saveDataset(path,names,data,DictNames,DictData):
    '''
    Save dataset to an HDF5 file
    
    Inputs: 
    path - Path to the save file
    names - names for each set of data in list
    data - list of data to be saved.
    DictNames - Names of a list of dicts to save as subgroups
    DictData - Data in the dicts
    
    Output:
    An HDF5 file at the path specified
    '''
    
    import h5py
    
    import os
    if os.path.exists(path):
        question = 'Delete original file (default: no)'
        choice = query_yes_no(question, default="no")
        if choice:
            os.remove(path)
            print("Original file deleted")

    f = h5py.File(path)
    
    for i in range(len(names)):
        #print(names[i])
        f.create_dataset(names[i], data=data[i])
        
    for j in range(len(DictNames)):  
        Set = f.create_group(DictNames[j]) #,(len(list(Cond.items())),)
        for i in list(DictData[j].items()):
            Set.create_dataset(i[0], data=i[1])
        
    print("File saved")
    f.close()


#Plot 2D scalar field 
def plotScalarField(S,X=None,Y=None,bound=None,saveFolder=None):
    '''
    Plot 2D scalar fields
    
    Inputs: 
    X - 2D array with columns constant
    Y - 2D array with rows constant
    S - the 2D field to be plotted. Must be the same size and shape as X,Y
    bound = the symetric bound of the colorbar (-bound, bound), detault is max of abs(S)/5
    
    Output:
    Displays a plot of the scalar field
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    if bound is None:
        bound = np.round(np.max(np.absolute(S))/5)
        
    f = plt.figure(figsize = [8,3])
    if X is None:
        plt.pcolor(S, cmap='RdBu_r');
        plt.axis('scaled')
        plt.xlim([0, S.shape[1]])
        plt.ylim([0, S.shape[0]])
    else:
        plt.pcolor(X,Y,S, cmap='RdBu_r');
        plt.axis('scaled')
        plt.xlim([X.min(), X.max()])
        plt.ylim([Y.min(), Y.max()])
    
    plt.clim([-1*bound, bound])
    plt.colorbar()

    if saveFolder is not None:
        f.savefig(saveFolder, transparent=True, bbox_inches='tight', pad_inches=0)
    
    return [f, plt.gca()]


#Gets locations of distinct scalar blobs in each frame that are bigger than a certain threshold (in number of vectors)
def findBlobsSlow(S,Thresh=None):
    '''
    Note!!!! This code does not work with 3D data yet. 
    Finds distinct blobs of a scalar that are bigger than a certain size (Thresh)
    
    Inputs:
    S - sets of 2D scalar fields that have already been thresholded (0s or 1s)
    Thresh - Number of vectors that must be contained in a blob. If not defined, then no threshold filter will be used
    
    Outputs:
    cent
    labelled_array
    num_features
    features_per_frame
    
    '''
    import numpy as np
    from scipy.ndimage.measurements import label,find_objects,center_of_mass
    import copy
    
    uSize = S.shape
    
    labeled_array, num_features = label(S)
    print('There are ', num_features, ' features initially identified')
    
    BBsize = []
    
    if Thresh is not None:
        loc = find_objects(labeled_array)
        labeled_array_init = copy.copy(labeled_array)
        labeled_array[:] = 0;
        num_features_init = copy.copy(num_features)
        num_features = 0;
        for i in range(num_features_init):
            #print(np.max(labeled_array_init[loc[i]]),labeled_array_init[loc[i]],np.count_nonzero(labeled_array_init[loc[i]]))
            #print(labeled_array_init[loc[i]])
            #print(np.max(labeled_array_init[loc[i]]),np.count_nonzero(labeled_array_init[loc[i]]))
            if np.count_nonzero(labeled_array_init[loc[i]])>Thresh:
                #print('yes')
                num_features += 1;
                labeled_array[labeled_array_init==i+1] = num_features

        print('A total of ', num_features, ' are larger than the threshold size')
    
    features_per_frame = np.zeros(uSize[2],dtype=int);
    cent = [];
    for i in range(uSize[2]):
        bbTemp = []
        features_per_frame[i] = len(np.unique(labeled_array[:,:,i])[1:])
        cent.append(center_of_mass(S[:,:,i],labeled_array[:,:,i],np.unique(labeled_array[:,:,i])[1:]))
        
        
    return [num_features, features_per_frame, labeled_array, cent]




#Gets locations of distinct scalar blobs in each frame that are bigger than a certain threshold (in number of vectors)

def findBlobs(S,Thresh=None,EdgeBound=None):
    '''
    Finds distinct blobs of a scalar that are bigger than a certain size (Thresh)
    Now new and improved! and much faster!
    
    Inputs:
    S - sets of 2D scalar fields that have already been thresholded (0s or 1s). The third dimension denotes the frame
    Thresh - Number of vectors that must be contained in a blob. If not defined, then no threshold filter will be used
    EdgeBound - Crops all blobs that are too close to the edge of the domain. No crop if left as none. 
    
    Outputs:
    cent - 
    labelled_array - The labeled array of blobs (in format of ndimage.measurements.label function). This is all the labels including the ones that might be too close to edge of domain
    num_features - Total number of features accross datasets
    features_per frame - Number of features identified in each frame
    
    '''
    import numpy as np
    from scipy.ndimage.measurements import label,find_objects,center_of_mass
    
    uSize = S.shape
    
    if S.ndim == 3:
        str_3D=np.array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],

       [[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]],

       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]], dtype='uint8')
        #str_3D = str_3D.transpose(2,1,0)

        labeled_array, num_features = label(S.transpose(2,1,0),str_3D)
        labeled_array = labeled_array.transpose(2,1,0)
    else:
        labeled_array, num_features = label(S)
            
    #print(np.unique(labeled_array))
    #print(np.unique(labeled_array[:,:,0]))
    #print(labeled_array.shape)
    
    print('There are ', num_features, ' features identified')
    
    if Thresh is not None:
        loc = find_objects(labeled_array)
        labeled_array_out = labeled_array.copy()
        
        counts = np.bincount(labeled_array.ravel())
        
        ind = np.where(counts>Thresh)[0][1:]
        mask = np.in1d(labeled_array.ravel(), ind).reshape(labeled_array.shape)
        labeled_array_out[~mask] = 0
        
        [_, labeled_array_out] = np.unique(labeled_array_out,return_inverse=True)
        labeled_array_out = labeled_array_out.reshape(labeled_array.shape)
        
        num_features_out = len(ind)
        
        print('A total of ', num_features_out, ' are larger than the threshold size')
    else:
        labeled_array_out = labeled_array
        num_features_out = num_features
        
            
    features_per_frame = np.zeros(uSize[2],dtype=int);
    cent = [];
    for i in range(uSize[2]):
        features_per_frame[i] = len(np.unique(labeled_array_out[:,:,i])[1:])
        cent.append(center_of_mass(S[:,:,i],labeled_array_out[:,:,i],np.unique(labeled_array_out[:,:,i])[1:]))
        
    #Round center locations to nearest index
    for i in range(len(cent)):
        for j in range(len(cent[i])):
            cent[i][j] = (int(round(cent[i][j][0])), int(round(cent[i][j][1])))
    
    #Remove all centers too close to edge of domain
    if EdgeBound is not None:
        newCent = []
        for i in range(len(cent)):
            newCent.append([])
            features_per_frame[i] = 0
            for j in range(len(cent[i])):
                if (cent[i][j][0]>EdgeBound-1 and cent[i][j][0]<uSize[0]-EdgeBound) and (cent[i][j][1]>EdgeBound-1 and cent[i][j][1] <uSize[1]-EdgeBound):
                    newCent[i].append(cent[i][j])
                    features_per_frame[i]+=1
        num_features_out = sum(features_per_frame)
        cent = newCent
        
        print('Of these', num_features_out, ' are far enough away from edge of domain')

    return [num_features_out, features_per_frame, labeled_array_out, cent]





# given the centers of a blobs, this function disects the vector field into a number of thumbnails of size frame x frame
def getThumbnails2D(mats,cent,BoxSize):
    '''
    Given the centers of a blobs, this function disects the vector field into a number of thumbnails of size frame x frame. 
    All vectors inside frame but outside domain are padded with nans
    
    Inputs:
    mats - A list of matrices from which to find thumbnails
    cent - centers of each each vortex as output by FindBlobs
    BoxSize - final thumbnail size is BoxSize*2+1 squared
    
    Outputs:
    Thumbnail versions of each of the matrices in mats
    '''
    import numpy as np
    
    uSize = mats[0].shape
    out = []
    
    #find out how many features there are 
    #Round all centroids to integers
    num_features = 0
    for i in range(len(cent)):
        for j in range(len(cent[i])):
            #print(i, j)
            cent[i][j] = (int(round(cent[i][j][0])), int(round(cent[i][j][1])))
            num_features += 1
            
    for k in range(len(mats)):
        #initialize thumbnail matrices
        U = np.zeros([2*BoxSize+1,2*BoxSize+1,num_features]) 
        U[:] = np.NAN

        #pad out velocity fields so that there are NaNs around in all directions
        U2 = np.zeros([uSize[0]+2*BoxSize,uSize[1]+2*BoxSize,uSize[2]])    
        U2[:] = np.NAN
        U2[BoxSize:-1*BoxSize,BoxSize:-1*BoxSize,:] = mats[k].copy() 

        #Now get the thumbnails
        thumb = 0
        for i in range(len(cent)):
            for j in range(len(cent[i])):
                U[:,:,thumb] = U2[cent[i][j][0]:cent[i][j][0]+2*BoxSize+1,cent[i][j][1]:cent[i][j][1]+2*BoxSize+1,i]  
                thumb+=1
                
        out.append(U)       
        del U   
        
    return out

def getThumbnails2D_V4(mats,cent,BoxSize):
    '''
    Given the centers of a blobs, this function disects the vector field into a number of thumbnails of size frame x frame. 
    All vectors inside frame but outside domain are padded with nans
    
    Inputs:
    mats - A list of matrices from which to find thumbnails
    cent - centers of each each vortex as output by FindBlobs
    BoxSize - final thumbnail size is BoxSize*2+1 squared
    
    Outputs:
    Thumbnail versions of each of the matrices in mats
    '''
    import numpy as np
    
    uSize = mats[0].shape
    out = []
    
    #find out how many features there are 
    #Round all centroids to integers
    num_features = 0
    for i in range(len(cent)):
        for j in range(len(cent[i])):
            #print(i, j)
            cent[i][j] = (int(round(cent[i][j][0])), int(round(cent[i][j][1])))
            num_features += 1
            
    for k in range(len(mats)):
        #initialize thumbnail matrices
        U = np.zeros([2*BoxSize+1,2*BoxSize+1,num_features]) 
        U[:] = np.NAN

        GG = mats[k].copy
        #Now get the thumbnails
        thumb = 0
        for i in range(len(cent)):
            for j in range(len(cent[i])):
                if thumb ==390: print(cent[i][j])
                U[:,:,thumb] = mats[k][cent[i][j][0]-BoxSize:cent[i][j][0]+BoxSize+1,cent[i][j][1]-BoxSize:cent[i][j][1]+BoxSize+1,i]  
                thumb+=1
                
        out.append(U)       
        del U   
        
    return out


def getRandomThumbnails2D(Uf,Vf,Sf,numSamp,BoxSize):
    import numpy as np
    
    uSize = Uf.shape
    
    Pos = np.random.rand(3,numSamp)
    Pos[0] = Pos[0]*(uSize[0]-2*BoxSize-1)+BoxSize
    Pos[1] = Pos[1]*(uSize[1]-2*BoxSize-1)+BoxSize
    Pos[2] = Pos[2]* (uSize[2]-1)
    Pos = Pos.round().astype(int)
    
    #print(Pos[:,0])
    #print(np.ndarray.min(Pos[0]))
    #print(np.ndarray.max(Pos[0]))
    #print(np.ndarray.min(Pos[1]))
    #print(np.ndarray.max(Pos[1]))
    #print(np.ndarray.min(Pos[2]))
    #print(np.ndarray.max(Pos[2]))
    #print(max(Pos))
    
    #initialize thumbnail matrices
    Ut = np.zeros([2*BoxSize+1,2*BoxSize+1,numSamp])    
    Ut[:] = np.NAN
    Vt = Ut.copy()
    St = Ut.copy()
    #print(Ut.shape)
    
    #pad out velocity fields so that there are NaNs around in all directions
    Uf2 = np.zeros([uSize[0]+2*BoxSize,uSize[1]+2*BoxSize,uSize[2]])    
    Uf2[:] = np.NAN
    Vf2 = Uf2.copy()
    Sf2 = Uf2.copy()

    Uf2[BoxSize:-1*BoxSize,BoxSize:-1*BoxSize,:] = Uf.copy()
    Vf2[BoxSize:-1*BoxSize,BoxSize:-1*BoxSize,:] = Vf.copy()
    Sf2[BoxSize:-1*BoxSize,BoxSize:-1*BoxSize,:] = Sf.copy()
    
    #Now get the thumbnails
    thumb = 0
    for i in range(numSamp):
        #print(i)
        Ut[:,:,thumb] = Uf2[Pos[0,i]:Pos[0,i]+2*BoxSize+1,Pos[1,i]:Pos[1,i]+2*BoxSize+1,Pos[2,i]]  
        Vt[:,:,thumb] = Vf2[Pos[0,i]:Pos[0,i]+2*BoxSize+1,Pos[1,i]:Pos[1,i]+2*BoxSize+1,Pos[2,i]] 
        St[:,:,thumb] = Sf2[Pos[0,i]:Pos[0,i]+2*BoxSize+1,Pos[1,i]:Pos[1,i]+2*BoxSize+1,Pos[2,i]]  
        thumb+=1
            
    return [Ut, Vt, St, Pos]

def genHairpinField(BoxSize,Circ,r,rs,Ts,Rot,StagStren,Conv,x=None,y=None):
    '''
    Generates a theoretical hairpin vortex velocity field given a number of parameters. Returns U and V velocity fields
    
    Inputs:
    BoxSize - 2*Boxsize+1 is the number of vectors per side of box. 
    Circ - Circulation strength of vortex 
    r - diameter of vortex solid body rotation (constant vector magnitude outside core)
    rs, Ts - polar coordinate location of stagnation point
    Rot - Rotation of stagnation point shear layer
    StagStren - Vector magnitude of stagnation point velocity field (constant magnitude)
    Gvort - Width of blending gaussian for vortex
    Gstag - Width of blending gaussian for stagnation point
    Conv - Convective velocity of this vortex relative to the local mean
    
    Outputs:
    U - Streamwise velocity field
    V - Wall-normal velocity field
    '''
    
    import numpy as np
    import math
    
    import matplotlib.pyplot as plt
    
    #print((x.shape[0]-1)/2)
    
    if x is None:
        X, Y = np.meshgrid(np.arange(-1*BoxSize, BoxSize+1), np.arange(-1*BoxSize, BoxSize+1))
    else:
        assert BoxSize==(x.shape[0]-1)/2, 'The BoxSize does not match the length of the x vector. Thats not right...'
        assert BoxSize==(y.shape[0]-1)/2, 'The BoxSize does not match the length of the y vector. Thats not right...'
        X, Y = np.meshgrid(x, y)

    U = np.zeros([2*BoxSize+1,2*BoxSize+1])
    V = U.copy()
    R = np.hypot(X, Y)
    T = np.arctan2(Y,X)

    #Vortex
    Ut = Circ*R/(2*np.pi*r**2)
    #Ut[R>=r] = Circ/(2*np.pi*R[R>=r])
    Ut[R>=r] = Circ/(2*np.pi*r)      #make velocities constant outside core

    #Now convert back to cartesian velocities
    Uvort = Ut*np.sin(T)
    Vvort = -1*Ut*np.cos(T)

    #Create stagnation point flow
    Rot = Rot*np.pi/180*2
    Ts = Ts*np.pi/180
    xs = rs*np.cos(Ts)            #shift in stagnation point in x
    ys = rs*np.sin(Ts)            #shift in stagnation point in y
    #StagStren = 2;

    Xs = X-xs
    Ys = Y-ys;
    Ts = np.arctan2(Ys,Xs)

    M = np.hypot(Xs, Ys)
    U = M*np.cos(Ts-Rot)
    V = -1*M*np.sin(Ts-Rot)
    M = np.hypot(U, V)
    Ustag = U/M*StagStren
    Vstag = V/M*StagStren

    Ustag[np.isnan(U)] = 0
    Vstag[np.isnan(V)] = 0
    
    #Combine fields (Previously input params)
    #Gvort_x = Gvort      #Radius of gaussian weighting function for vortex field
    #Gvort_y = Gvort_x
    #Gstag = 5      #Radius of gaussian weighting function for stagnation point field
    
    FWHM = rs
    
    Gvort_x = (FWHM/(2*(2*np.log(2))**0.5))**2 ;
    Gvort_y = (FWHM/(2*(2*np.log(2))**0.5))**2 ;
    Gstag = (FWHM/(2*(2*np.log(2))**0.5))**2;

    Wvort = np.exp(-((X**2)/(2*Gvort_x)+(Y**2)/(2*Gvort_y)))+0.0001
    Wvort_inv = -1*Wvort+1                #invert the weightings so that only vortex appears at vortex location                       
    Rstag = np.hypot(Xs, Ys)
    #Wstag = np.exp(-((X-xs)**2/(2*Gstag)+(Y-ys)**2/(2*Gstag)))+0.0001
    Wstag = np.exp(-1*(Rstag)**2/(2*Gstag))
    #Wstag_inv = -1*Wstag+1
    
    #print(sum(sum(np.isnan(Uvort[:]))))
    #print(sum(sum(np.isnan(Ustag[:]))))
    #plt.plot(X[0,:],Wvort_inv[BoxSize+1,:])
    
    M = np.hypot(Uvort, Vvort)
    Vmax = Ut.max()
    M = np.hypot(Ustag, Vstag)
    Smax = M.max()
    #print(Vmax)
    #print(Smax)
    FlowRat = Smax/Vmax
    #print(FlowRat)
    
    #U = Uvort+Conv
    #V = Vvort
    
    #U = (Wvort+FlowRat*Wstag)
    #V = (Wvort+FlowRat*Wstag)

    #initial blend. Enforces the vortex and stagnation point exactly.
        # Problems when stagnation point flow is weak. Need to find a better way to represent this
    U = (Wvort*Uvort+FlowRat*Wstag*Ustag)/(Wvort+FlowRat*Wstag)+Conv
    V = (Wvort*Vvort+FlowRat*Wstag*Vstag)/(Wvort+FlowRat*Wstag)
    
    #Blend where stagnation and vortex are always of similar magnitude
        #Could remove stagnation strength magnitude in this case 
    #U = (Wvort*Uvort+Wstag*Ustag)/(Wvort+FlowRat*Wstag)+Conv
    #V = (Wvort*Vvort+Wstag*Vstag)/(Wvort+FlowRat*Wstag)
    
    #New blend: state location of stagnation point, but let strength vary such that stagnation point is enforced in summation               with vortex field. Would probably still need to have unphysical vortex.
    
    
    #For diagnostic purposes 
    #U = (Wvort)/(Wvort+FlowRat*Wstag)
    #V = (FlowRat*Wstag)/(Wvort+FlowRat*Wstag)
    
    #V = Wvort
    
    #U = (Wvort*Uvort/FlowRat+FlowRat*Wstag*Ustag)/(Wvort/FlowRat+FlowRat*Wstag)+Conv
    #V = (Wvort*Vvort/FlowRat+FlowRat*Wstag*Vstag)/(Wvort/FlowRat+FlowRat*Wstag)
    
    #U = (Wvort*Uvort+0*Wstag*Ustag)/(Wvort+0*Wstag)+Conv
    #V = (Wvort*Vvort+0*Wstag*Vstag)/(Wvort+0*Wstag)

    #U = (Wvort*Wstag_inv*Uvort+Wstag*Wvort_inv*Ustag)/(Wvort*Wstag_inv+Wstag*Wvort_inv)+Conv
    #V = (Wvort*Wstag_inv*Vvort+Wstag*Wvort_inv*Vstag)/(Wvort*Wstag_inv+Wstag*Wvort_inv)

    #U = (Wstag_inv*Uvort+Wvort_inv*Ustag)/(Wvort_inv+Wstag_inv)
    #V = (Wstag_inv*Vvort+Wvort_inv*Vstag)/(Wvort_inv+Wstag_inv)
    
    #print(sum(sum(np.isnan(U[:]))))
    #print(sum(sum(np.isnan(V[:]))))
    
    return [U, V]

def genHairpinField_V2(BoxSize,Circ,r,LineInt,LineAngle,Uin,Vin,Ulow,Vlow,Conv,x=None,y=None):
                       
    '''
    Generates a theoretical hairpin vortex velocity field given a number of parameters. Returns U and V velocity fields
    
    Inputs:
    BoxSize - 2*Boxsize+1 is the number of vectors per side of box. 
    Circ - Circulation strength of vortex 
    r - diameter of vortex solid body rotation (constant vector magnitude outside core)
    LinInt - 
    LineAngle - 
    Uin - 
    Vin - 
    Uin - 
    Vin - 
    Conv - Convective velocity of this vortex relative to the local mean
    
    Outputs:
    U - Streamwise velocity field
    V - Wall-normal velocity field
    '''
    
    import numpy as np
    import math
    
    #%matplotlib inline
    import matplotlib.pyplot as plt
    
    #print((x.shape[0]-1)/2)
    
    if x is None:
        X, Y = np.meshgrid(np.arange(-1*BoxSize, BoxSize+1), np.arange(-1*BoxSize, BoxSize+1))
    else:
        assert BoxSize==(x.shape[0]-1)/2, 'The BoxSize does not match the length of the x vector. Thats not right...'
        assert BoxSize==(y.shape[0]-1)/2, 'The BoxSize does not match the length of the y vector. Thats not right...'
        X, Y = np.meshgrid(x, y)

    U = np.zeros([2*BoxSize+1,2*BoxSize+1])
    V = U.copy()
    R = np.hypot(X, Y)
    T = np.arctan2(Y,X)

    #Vortex
    Ut = Circ*R/(2*np.pi*r**2)
    Ut[R>=r] = Circ/(2*np.pi*R[R>=r])
    #Ut[R>=r] = Circ/(2*np.pi*r)      #make velocities constant outside core

    #Now convert back to cartesian velocities
    Uvort = Ut*np.sin(T)
    Vvort = -1*Ut*np.cos(T)

    #Create shear layer
    VecIn = [Uin,Vin]
    VecLow = [Ulow,Vlow]
    #print(VecIn)
    #print(VecLow)
    dVec = [VecLow[0]-VecIn[0],VecLow[1]-VecIn[1]]
    #LineInt = 0
    #LineAngle = 30
    LineAngle = np.radians(-1*LineAngle)

    Us = U.copy()
    Vs = U.copy()

    X2 = X*np.cos(LineAngle) - Y*np.sin(LineAngle)
    Y2 = X*np.sin(LineAngle) + (Y-LineInt)*np.cos(LineAngle)

    Rs = np.hypot(X2,Y2)
    Ts = (-1*np.arctan2(Y2,X2)+np.pi)/(2*np.pi)
    Us = VecIn[0]+dVec[0]*(Ts)
    Vs = VecIn[1]+dVec[1]*(Ts)

    #Calculate combined field
    FWHM = 2*r
    Gvort_x = (FWHM/(2*(2*np.log(2))**0.5))**2      #Radius of gaussian weighting function for vortex field
    Gvort_y = Gvort_x
    Wvort = np.exp(-((X**2)/(2*Gvort_x)+(Y**2)/(2*Gvort_y)))
    Wvort_inv = -1*Wvort+1                #invert the weightings so that only vortex appears at vortex location                       
    U = Us*Wvort_inv/Wvort_inv.max()+Uvort+Conv
    V = Vs*Wvort_inv/Wvort_inv.max()+Vvort
    
    #print(Wvort_inv[BoxSize+1,:].shape)
    #print(X[0,:].shape)
    #plt.plot(X[0,:],Wvort_inv[BoxSize+1,:])

    return [U, V]

def genHairpinField_V3(BoxSize,Circ,r,xc,yc,Conv,x=None,y=None):
                       
    '''
    Generates a theoretical hairpin vortex velocity field given a number of parameters. Returns U and V velocity fields
    
    Inputs:
    BoxSize - 2*Boxsize+1 is the number of vectors per side of box. 
    Circ - Circulation strength of vortex 
    r - diameter of vortex solid body rotation (constant vector magnitude outside core)
    xs - 
    ys - 
    Conv - Convective velocity of this vortex relative to the local mean
    
    Outputs:
    U - Streamwise velocity field
    V - Wall-normal velocity field
    '''
    
    import numpy as np
    import math
    
    #%matplotlib inline
    import matplotlib.pyplot as plt
    
    #print((x.shape[0]-1)/2)
    
    if x is None:
        X, Y = np.meshgrid(np.arange(-1*BoxSize, BoxSize+1), np.arange(-1*BoxSize, BoxSize+1))
    else:
        assert BoxSize==(x.shape[0]-1)/2, 'The BoxSize does not match the length of the x vector. Thats not right...'
        assert BoxSize==(y.shape[0]-1)/2, 'The BoxSize does not match the length of the y vector. Thats not right...'
        X, Y = np.meshgrid(x, y)

    U = np.zeros([2*BoxSize+1,2*BoxSize+1])
    V = U.copy()
    R = np.hypot(X-xc, Y-yc)
    T = np.arctan2(Y-yc, X-xc)

    #Vortex
    Ut = Circ*R/(2*np.pi*r**2)
    Ut[R>=r] = Circ/(2*np.pi*R[R>=r])
    #Ut[R>=r] = Circ/(2*np.pi*r)      #make velocities constant outside core

    #Now convert back to cartesian velocities
    Uvort = Ut*np.sin(T)
    Vvort = -1*Ut*np.cos(T)

    #Create shear layer                   
    U = Uvort+Conv
    V = Vvort
    
    #print(Wvort_inv[BoxSize+1,:].shape)
    #print(X[0,:].shape)
    #plt.plot(X[0,:],Wvort_inv[BoxSize+1,:])

    return [U, V]

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    
    import sys
    
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
            
#Sahil's functions
#---------------------------------------------------------------------------
from collections import Counter
import numpy as np


#gets the number of neighbours around a cell
#answer max bound is 4
def getNbrs(S,p1,p2):
    ln = len(S)
    ct = 0
    if (p2+1)<ln and S[p1,p2+1] > 0: ct+=1
    if (p2-1)>=0 and S[p1,p2-1] > 0:  ct+=1
    if (p1-1)>=0 and S[p1-1,p2] > 0 :  ct+=1
    if (p1+1)<ln and S[p1+1,p2] > 0 : ct+=1
    return ct
        
#returns a matrix whose values are the getNbrs values
def vortMat(S,p1,p2,SS):
    if SS[p1,p2] != 0: #already visited
        return SS
    if S[p1,p2] == 0: #outside vortex swirl
        SS[p1,p2] = -1
        print("sd")
        return SS
    nbs = getNbrs(S,p1,p2)
    SS[p1,p2] = nbs

    if (p1-1>=0):SS = vortMat(S,p1-1,p2,SS)
    if (p1+1<len(S)):SS = vortMat(S,p1+1,p2,SS)
    if (p2-1>=0):SS = vortMat(S,p1,p2-1,SS)
    if (p2+1 < len(S)):SS = vortMat(S,p1,p2+1,SS)
    
    return SS

#calculates the area and the perimeter values based on the vortMat matrix
def vortAP(S,p1,p2):
    Ln  = len(S)
    Sz = np.zeros([Ln,Ln])
    Smat = vortMat(S,p1,p2,Sz)
    Sflt = Smat.flatten()
    
    SSc = Counter(Sflt)
    peri = sum([SSc[i] for i in SSc.keys() if i < 4 and i > 0])
    area = sum([SSc[i] for i in SSc.keys() if i >0])
    print("Area: ", area)
    print("Perimeter: ", peri)
    
    return peri, area

#-----------------------------------------

def getTop(S,p1,p2,SS):
    if SS[p1][p2] != 0:
        return 0
    SS[p1][p2] -= -1
    
    if (S[p1][p2] == 0):
        return 0
    
    if p1==0:
        return 0
    
    if S[p1-1][p2] != 0:
        return 1+getTop(S,p1-1,p2,SS)
    
    return max(getTop(S,p1,p2-1,SS),getTop(S,p1,p2+1,SS))

#updated version
#takes care of the new centers as the matrix is 'rotated'. 
def getLimits_V2(S,p1,p2):
    Cs = np.zeros([len(S),len(S[0])])
    Cs[p1,p2] = 1
    
    Ln1 = len(S)
    Ln2 = len(S[0])
    
    SS = np.zeros([Ln1,Ln2])
    h1  = getTop(S,p1,p2,np.zeros([len(S),len(S[0])]) )
    
    S2 = S.T[:]
    b1 = getTop(S2,p2,p1,np.zeros([len(S2),len(S2[0])]))
    
    S3 = np.array([S[Ln1-i-1] for i in range(Ln1)])
    h2 = getTop(S3,len(SS)-p1-1,p2,np.zeros([len(S3),len(S3[0])]))
    
    S4 = [S2[len(S2)-1-i] for i in range(len(S2))]
    b2 = getTop(S4, len(SS[0])-p2-1,p1,np.zeros([len(S4),len(S4[0])]))
    
    return [h1,h2,b1,b2]

def getEpsAP(S,p1,p2):
    h1,h2,b1,b2 = getLimits(S,p1,p2)
    areaE = np.pi*(h1+h2)*(b1+b2)
    periE = a*np.pi*np.sqrt(((h1+h2)**2 + (b1+b2)**2)/2)
    return areaE, periE


#----------------------------------------------------------------------------------

#bbtemp = [[len(y),len(y[0])] for y in  ]
#        BBsize.append(bbtemp)

def bbsize_fun(S):
    #can be optimized by labelling S again
    
    # S has [i,j,frame]
    import numpy as np
    from scipy.ndimage.measurements import label,find_objects,center_of_mass

    print(np.unique(S[:,:,0]), "FMLLL")
    S_ = np.zeros(S.shape)
    for i in range(len(S)):
        S_[:,:,i], num_features = label(S[:,:,i])
    #now all blobs have numbers 
    print(np.unique(S_[:,:,0]), "FMLL??L")

    print(num_features, "have been identified")
    
    #find the objects
    bbSize = []

    print(S[:,:,0])
    print(S.shape)
    
    
    for frame in range(len(S)): #each frame
        print(np.unique(S[:,:,frame]), "fghjkj")
        #should give out 1,2,3,...
        
        get_slices = find_objects(S_[:,:,frame])
        #returns sliced objects
        
        temp_BB = []
        ref_frame = S_[:,:,frame]
        for obj in get_slices:

                blob = ref_frame[obj]
                print(obj)
                1./0
                temp_BB.append([len(blob),len(blob[0])])
    
        bbSize.append(temp_BB)
    return bbSize
            
    
import numpy as np
from scipy.ndimage.measurements import label,find_objects,center_of_mass

    


import numpy as np
from scipy.ndimage.measurements import label,find_objects,center_of_mass

def findBlobs_V2(S,Thresh=None,EdgeBound=None):
    '''
    V2  also exports the extremes(*) of the blob
    Finds distinct blobs of a scalar that are bigger than a certain size (Thresh)
    Now new and improved! and much faster!
    
    Inputs:
    S - sets of 2D scalar fields that have already been thresholded (0s or 1s). The third dimension denotes the frame
    Thresh - Number of vectors that must be contained in a blob. If not defined, then no threshold filter will be used
    EdgeBound - Crops all blobs that are too close to the edge of the domain. No crop if left as none. 
    
    Outputs:
    cent - 
    labelled_array - The labeled array of blobs (in format of ndimage.measurements.label function). This is all the labels including the ones that might be too close to edge of domain
    num_features - Total number of features accross datasets
    features_per frame - Number of features identified in each frame
    BB_Size - the blob.size values
    
    '''
    import numpy as np
    from scipy.ndimage.measurements import label,find_objects,center_of_mass
    
    uSize = S.shape
    
    if S.ndim == 3:
        str_3D=np.array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],

       [[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]],

       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]], dtype='uint8')
        #str_3D = str_3D.transpose(2,1,0)

        labeled_array, num_features = label(S.transpose(2,1,0),str_3D)
        labeled_array = labeled_array.transpose(2,1,0)
    else:
        labeled_array, num_features = label(S)
            
    print('There are ', num_features, ' features identified')
    
    if Thresh is not None:
        loc = find_objects(labeled_array)
        labeled_array_out = labeled_array.copy()
        
        counts = np.bincount(labeled_array.ravel())
        
        ind = np.where(counts>Thresh)[0][1:]
        mask = np.in1d(labeled_array.ravel(), ind).reshape(labeled_array.shape)
        labeled_array_out[~mask] = 0
        
        [_, labeled_array_out] = np.unique(labeled_array_out,return_inverse=True)
        labeled_array_out = labeled_array_out.reshape(labeled_array.shape)
        
        num_features_out = len(ind)
        
        print('A total of ', num_features_out, ' are larger than the threshold size')
    else:
        labeled_array_out = labeled_array
        num_features_out = num_features
            
    features_per_frame = np.zeros(uSize[2],dtype=int);
    cent = [];
    BB_Size = []
    j = 0
    for i in range(uSize[2]):
        btemp = []
        features_per_frame[i] = len(np.unique(labeled_array_out[:,:,i])[1:])
        cent.append(center_of_mass(S[:,:,i],labeled_array_out[:,:,i],np.unique(labeled_array_out[:,:,i])[1:]))
        slices = find_objects(labeled_array_out[:,:,i])
        
        #remove the gigo slice values : check *note 1*
        btemp = [labeled_array_out[:,:,i][j].shape for j in slices[-len(cent[-1]):]]
        BB_Size.append(btemp)
        
    #Round center locations to nearest index
    for i in range(len(cent)):
        for j in range(len(cent[i])):
            cent[i][j] = (int(round(cent[i][j][0])), int(round(cent[i][j][1])))
    
    #Remove all centers too close to edge of domain
    if EdgeBound is not None:
        newCent = []
        newBB = []

        for i in range(len(cent)):
            newCent.append([])
            newBB.append([])
            features_per_frame[i] = 0
            for j in range(len(cent[i])):
                if (cent[i][j][0]>EdgeBound-1 and cent[i][j][0]<uSize[0]-EdgeBound) and (cent[i][j][1]>EdgeBound-1 and cent[i][j][1] <uSize[1]-EdgeBound):
                            
                    newCent[i].append(cent[i][j])
                    #remove the blob extremes from the bbsize as well : for consistency with cent
                    newBB[i].append(BB_Size[i][j])
                    features_per_frame[i]+=1
        
        num_features_out = sum(features_per_frame)
        cent = newCent
        BB_Size = newBB[:]
        print('Of these', num_features_out, ' are far enough away from edge of domain')
    return [num_features_out, features_per_frame, labeled_array_out, cent, BB_Size]

def getThumbnails2D_V2(mats,cent, BBM):
    '''
    *update (in V2): integrates the BB size of the blob into decising the thumbnails' size.*
    Given the centers of a blobs, this function disects the vector field into a number of thumbnails of size frame x frame. 
    All vectors inside frame but outside domain are padded with nans
    
    Inputs:
    mats - A list of matrices from which to find thumbnails
    cent - centers of each each vortex as output by FindBlobs
    *BoxSize - custom box size used from the findBlobs_V2 function  
    Outputs:
    Thumbnail versions of each of the matrices in mats
    '''
    import numpy as np
    
    uSize = mats[0].shape
    out = []
    
    #find out how many features there are 
    #Round all centroids to integers
    num_features = 0
    for i in range(len(cent)):
        for j in range(len(cent[i])):
            #print(i, j)
            cent[i][j] = (int(round(cent[i][j][0])), int(round(cent[i][j][1])))
            num_features += 1
    
    BB_ = []
    for i in BBM:
        BB_.extend(i)

    
    for k in range(len(mats)):
        #initialize thumbnail matrices
        
        U = []
        
        for i in range(num_features): #frame
            bbs = int(round( max(BB_[i])*0.5) )
            
            U_temp = np.zeros([2*bbs+1,2*bbs+1])
            U_temp[:] =  np.NAN 
            U.append(U_temp)
        
        #print(max(BB_[0])*1.2)
        #print(len(U))
        #print(len(U[1]))
        #print(len(U[1][1]))
        
        #pad out velocity fields so that there are NaNs around in all directions
        U2 = []
        for i in range(uSize[2]):
            bbs = int(round( max(BB_[i])*0.5) )
            U_temp = np.zeros([uSize[0]+2*bbs,uSize[1]+2*bbs])    
            U_temp[:] = np.NAN
            JJ = mats[k]
            U_temp[bbs:-1*bbs,bbs:-1*bbs] = JJ[:,:,i].copy() 
            U2.append(U_temp)
        
        #print(len(U2))
        #print(len(U2[0]))
        #print(len(U2[0][0]))
        cent_ = []
        for i in cent:
            cent_.extend(i)
        #Now get the thumbnails
        thumb = 0
        for i in range(len(cent)):
            U2_D = U2[i]
            for j in range(len(cent[i])):
                bbs = int(round( max(BB_[thumb])*1.2) )
                if j <=5 and i == 0 and k == 0:
                    print(bbs)
                #print(bbs)
                U[thumb] = U2_D[cent_[thumb][0]:cent_[thumb][0]+2*bbs+1,cent_[thumb][1]:cent_[thumb][1]+2*bbs+1]  
                thumb+=1
        #print(':D')
        out.append(U)       
        del U   
    #print('B-D')
    return out

def getThumbnails2Dx(mats,cent,BoxSize):
    '''
    Given the centers of a blobs, this function disects the vector field into a number of thumbnails of size frame x frame. 
    All vectors inside frame but outside domain are padded with nans
    
    Inputs:
    mats - A list of matrices from which to find thumbnails
    cent - centers of each each vortex as output by FindBlobs
    BoxSize - final thumbnail size is BoxSize*2+1 squared
    
    Outputs:
    Thumbnail versions of each of the matrices in mats
    '''
    import numpy as np
    
    uSize = mats[0].shape
    out = []
    
    #find out how many features there are 
    #Round all centroids to integers
    num_features = 0
    for i in range(len(cent)):
        for j in range(len(cent[i])):
            #print(i, j)
            cent[i][j] = (int(round(cent[i][j][0])), int(round(cent[i][j][1])))
            num_features += 1
            
    for k in range(len(mats)):
        #initialize thumbnail matrices
        U = np.zeros([2*BoxSize+1,2*BoxSize+1,num_features]) 
        U[:] = np.NAN

        #pad out velocity fields so that there are NaNs around in all directions
        U2 = np.zeros([uSize[0]+2*BoxSize,uSize[1]+2*BoxSize,uSize[2]])    
        U2[:] = np.NAN
        U2[BoxSize:-1*BoxSize,BoxSize:-1*BoxSize,:] = mats[k].copy() 

        #Now get the thumbnails
        thumb = 0
        for i in range(len(cent)):
            for j in range(len(cent[i])):
                U[:,:,thumb] = U2[cent[i][j][0]:cent[i][j][0]+2*BoxSize+1,cent[i][j][1]:cent[i][j][1]+2*BoxSize+1,i]  
                thumb+=1
                
        out.append(U)       
        del U   
    return out


'''
Notes:

#1 The label function labels blobs beginning at the frame 0 and goes on .
Now every frame has labels beginning not at 1 but at *(# of blobs in prev frame)* and the slice objects
for that index would be none.
To remove this None while getting the max bounds the slices[-len(cent[-1]):] is done!





'''