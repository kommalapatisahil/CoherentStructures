#functions used to automate the Grafteaux
import grafteaux as G
import numpy as np
def trueSlice(GG, cent):
    '''
    takes in GG  - T2 mat
    cent - relative center obtaineed from  T1
    
    returns the corresponding slice. 
    Idea: v_curr = V[slice]
    '''
    from scipy.ndimage.measurements import label, find_objects
    
    larr, _ = label(GG)
    ss = find_objects(larr)
    
    for slic in ss: 
        PP = np.zeros(GG.shape)
        PP[slic] = 1
        if PP[cent[0], cent[1]] == 1: return slic
    return 0

def getFinUVXY(frame, vid, cent_Pro, BB_Pro, U, V,SwirlFilt, Umean, Cond, bbmult = 1):
            #-------------------------------------------
            #from blobs get T1 and T2

    
            xc, yc = cent_Pro[frame][vid]
            a = int((max(BB_Pro[frame][vid])*bbmult-1)/2)
            l1, l2, l3 = U.shape

            if xc-a < 0 or yc-a <0 or xc+a >= l1 or yc+a >= l2: return '*' 

            ut = U[xc-a:xc+a+1,yc-a:yc+a+1,frame]
            ut_norm = (ut-Umean[xc])/Cond["Uinf"][0][0]
            vt = V[xc-a:xc+a+1,yc-a:yc+a+1,frame]
            
            
            x,y = G.get_xy(len(ut))
            X, Y = np.meshgrid(x,y)
            
            T1_curr = G.T1_big_mat(X, Y, ut_norm, vt, 1)
            T2_curr = G.T2_big_mat(X, Y, ut_norm, vt, 1)
            
            
            T1_th = T1_curr.copy()
            T1_th[T1_curr<0.8 ] = 0
            curr_cents = G.getCents(T1_th)
            
            T2_th = T2_curr.copy()
            T2_th[T2_curr< 2/np.pi] = 0
            
            #-----------------------------------------------------------
            #using T1 and T2 get new cents
            
            cents_ = G.getCents(T1_th)
            if len(cents_) == 0: return '*'
            
            compl_dict = G.getProps(T2_th, cents_)
            bigPP = G.getPP(compl_dict)
            
            rr_, xy_ = compl_dict
            rel_cent = xy_[bigPP]
            
            true_rad = rr_[bigPP]
            
            #----------> ** constraints ** <---------------
            
            if int(true_rad*1.25)<=5 : return '*' #print('Blob too small..EXP 1')
            if (np.mean(T1_th) == 0): return '*'#np peaks 
            if (len(curr_cents) >1) : return '*'#too many T1 peaks 
            
            curr_cent_abs = [(xc-a+rel_cent[0],yc-a+rel_cent[1])]
            #----------------------------------------------------
            #using true and rel cent get the U and V fields.
            
            c1, c2 = curr_cent_abs[0]
            
            s1 = slice(c1-true_rad, c1+true_rad+1)
            s2 = slice(c2-true_rad, c2 + true_rad+1)
            
            s_temp = SwirlFilt[s1,s2, frame]
            
            U_curr = U[s1,s2,frame]
            V_curr = V[s1,s2,frame]
            U_currn = (U_curr-Umean[int(c1)])/Cond['Uinf'][0][0]
            S_curr = SwirlFilt[s1, s2, frame]
            x,y = G.get_xy(len(U_curr))
            X,Y = np.meshgrid(x,y)
            
            props = {}
            props['frame']  = frame
            props['cent'] = curr_cent_abs[0]
            props['rad']  = true_rad
            
            return U_currn, V_curr, x, y , S_curr, props
            
def F1check(s, nburn):
    #checks multiple peaks
    from scipy.signal import find_peaks
    ndim = 5
    sampler = s.chain[:, nburn:, :] #burning the inital nburn steps
    trace = sampler.reshape(-1, ndim).T   #reshaping so that each row(parameter) has all the walker probs

    for i in range(5):
        h,b  = np.histogram(trace[i], bins = 90, density = True)
        pr = max(h)*5/100
        kk = find_peaks(h ,prominence=pr) #,threshold=thr
        peaks = len(kk[0]) 
        if  peaks >= 2: 
            #print(kk)
            print('F1 fail.')
            return 1
    return 0

def F3checkV2(s,nburn, thr):
    '''
    using 1% instead of 1 point
    '''
    ndim = 5
    sampler = s.chain[:, nburn:, :] #burning the inital nburn steps
    trace = sampler.reshape(-1, ndim).T   #reshaping so that each row(parameter) has all the walker probs

    for i in range(5):

        hist, bin_edges = np.histogram(trace[i,:],bins=100) #it was bins = auto earlier
        bin_mids = [(bin_edges[j]+bin_edges[j+1])/2 for j in range(len(bin_edges)-1)]
        
        thrld = max(hist)*thr/100 #1%
        
        lhs = np.mean(hist[:len(hist)*1//100])
        rhs = np.mean(hist[-len(hist)*1//100:])
        #print(lhs, rhs, thrld)
        if lhs > thrld or rhs>thrld : 
            #print('F3 fail.')
            return 1
    return 0


def F3check(s,nburn, thr):

    ndim = 5
    sampler = s.chain[:, nburn:, :] #burning the inital nburn steps
    trace = sampler.reshape(-1, ndim).T   #reshaping so that each row(parameter) has all the walker probs

    for i in range(5):

        hist, bin_edges = np.histogram(trace[i,:],bins=100) #it was bins = auto earlier
        bin_mids = [(bin_edges[j]+bin_edges[j+1])/2 for j in range(len(bin_edges)-1)]
        
        thrld = max(hist)*thr/100 #1%
        if hist[0] > thrld or hist[-1]>thrld : 
            #print('F3 fail.')
            return 1
    return 0

def sucRate(a, b):
    import numpy as np
    FTB = (np.array(a) == 1 )|( np.array(b) == 1)
    print('Success Rate: ', (1-sum(FTB)/len(FTB))*100 )
    return (1-sum(FTB)/len(FTB)) 

def SR2(a):
    '''
    a - samp list
    returns F3 fail success rate.
    '''
    import numpy as np
    FTB = np.zeros([len(a)])
    for i in range(len(a)):
        f3 = F3check(a[i], 5000, 95)
        if f3: FTB[i] = 1
    return (1-sum(FTB)/len(FTB)) 


#-----------------------------------------------------------------------------------------
#correction functions
#-----------------------------------------------------------------------------------------

def Mcorrection(props, U, V, Umean, Cond, curr_samp):
    '''
    corrects M-ultiple peaks 
    I/O: I | best case updated sampler, boolean to indicate success.
    '''
    p_samp = curr_samp
    c_samp = curr_samp
    
    frame = props['frame']
    cent = props['cent']
    rad  = props['rad']
    
    rth = 5#pixels
    xc, yc = cent

    rad-=1
    
    l1, l2, l3 = U.shape
    
    while (True):
            
            if xc-rad<0 or xc+rad+1 >= l1 or yc-rad<0 or yc+rad+1 >= l2:print('out of bounds!'); return p_samp, 0 

            u_curr = U[xc-rad:xc+rad+1, yc-rad:yc+rad+1, frame]
            v_curr = V[xc-rad:xc+rad+1, yc-rad:yc+rad+1, frame]
            u_currn = (u_curr-Umean[xc])/Cond['Uinf'][0][0]
            
            print('working on rad:', rad, ' rth-',rth)
            x, y = G.get_xy(len(u_curr))
            
            p_samp = [c_samp].copy()[0]
            c_samp = G.doMCMC_V3(u_currn, v_curr, x, y)

            G.plot_corner(c_samp, 5000)
            
            F1 = AG.F1check(c_samp, 5000)
            F3 = AG.F3check(c_samp, 5000, 98)
            print(F1, F3, 'F1F3')
            #here we expect the correction to finalize on a single peak
            if not F3 and not F1:   print('not f3, f1 anymore!'); return c_samp, 1
            elif not F1 and F3: print('got div so fail-prev samp returned'); return p_samp, 0
            else:  
                print('still f1')
                rad-=1
                if rad>=rth:
                    print('threshold *not* reached, curr r', rad)
                    continue 
                else: 
                    print('threshold reached, curr r', rad)
                    return c_samp, 0
            print('here_________________wtf___________<><><><>')
            break
            
            
def Dcorrection(props, U, V, Umean, Cond, curr_samp):
    '''
    corrects D-ivergece of peaks
    I/O: I | best case updated sampler, boolean to indicate success.
    '''
    p_samp = curr_samp
    c_samp = curr_samp
    
    frame = props['frame']
    cent = props['cent']
    rad  = props['rad']
    
    rth = int(2*props['rad'])
    xc, yc = cent
    mult = 1.1
    rad += 1

    l1, l2, l3 = U.shape
    
    while (True):
            
            if xc-rad<0 or xc+rad+1 >= l1 or yc-rad<0 or yc+rad+1 >= l2: print('out of bounds!'); return p_samp, 0 

            u_curr = U[xc-rad:xc+rad+1, yc-rad:yc+rad+1, frame]
            v_curr = V[xc-rad:xc+rad+1, yc-rad:yc+rad+1, frame]
            u_currn = (u_curr-Umean[xc])/Cond['Uinf'][0][0]
            
            print('working on rad:', rad, ' rth-',rth)
            x, y = G.get_xy(len(u_curr))
            
            p_samp = c_samp
            c_samp = G.doMCMC_V3(u_currn, v_curr, x, y)

            G.plot_corner(c_samp, 5000)
            
            F1 = F1check(c_samp, 5000)
            F3 = F3check(c_samp, 5000, 98)
            
            if not F3:  print('not f3 anymore!'); return c_samp, 1
            elif F3 and F1: print('f3 and f1');   return c_samp, 1 #not likely to happen; indicates a jump from F3 to f3+f1
            else:  
            
                print('still f3 and not f1')
                rad+=1
                if rad<=rth:
                    print('threshold *not* reached, curr r', rad)
                    continue 
                else: 
                    print('threshold reached, curr r', rad)
                    return c_samp, 0
            print('here_________________wtf___________<><><><>')
            break

#from these the goal is to finally find some corner plot which would make reasonalble sense!
#now we use GMM

            
def DcorrectionV2(props, U, V, Umean, Cond, curr_samp):
    '''
    only care about F3 fails! 
    
    corrects D-ivergece of peaks
    I/O: I | best case updated sampler, boolean to indicate success.
    1: success, 0: fail.
    '''
    p_samp = curr_samp
    c_samp = curr_samp
    
    frame = props['frame']
    cent = props['cent']
    rad  = props['rad']
    
    rth = int(2*props['rad'])
    xc, yc = cent
    mult = 1.1
    rad += 1

    l1, l2, l3 = U.shape
    
    while (True):
            
            if xc-rad<0 or xc+rad+1 >= l1 or yc-rad<0 or yc+rad+1 >= l2: print('out of bounds!'); return p_samp, 0 

            u_curr = U[xc-rad:xc+rad+1, yc-rad:yc+rad+1, frame]
            v_curr = V[xc-rad:xc+rad+1, yc-rad:yc+rad+1, frame]
            u_currn = (u_curr-Umean[xc])/Cond['Uinf'][0][0]
            
            print('working on rad:', rad, ' rth-',rth)
            x, y = G.get_xy(len(u_curr))
            
            p_samp = c_samp
            c_samp = G.doMCMC_V3(u_currn, v_curr, x, y)

            G.plot_corner(c_samp, 5000)
            
            F3 = F3check(c_samp, 5000, 98)
            rad+=1 
            if not F3: return c_samp, 1
            if rad <= rth: continue
            else: print('threshold reached!'); return c_samp, 0
            
def DcorrectionV3(Cond, U, V, Umean, props, curr_samp, verbose = False):
    '''
    only care about F3 fails! 
    takes increments to exact rectangular BB dims; 
    //instead of increasing radius solely.
    
    corrects D-ivergece of peaks
    I/O: I | best case updated sampler, boolean to indicate success.
    1: success, 0: fail.
    
    
    '''
    #props - center, frame, bbdims
    p_samp = curr_samp
    c_samp = curr_samp
    
    
    
    frame = props['frame']
    cent = props['cent']
    bbdims  = props['bbdims']
    
    rth = int(2*max(bbdims))
    xc, yc = cent
    mult = 1.1
    
    b1, b2, b3, b4 = bbdims+1

    l1, l2, l3 = U.shape
    
    while (True):

            #odd len box
            if (b1+b2+1)%2 == 0 : b2+=1 
            if (b3+b4+1)%2 == 0 : b4+=1

            #check bounds
            if xc-b1<0 or xc+b2+1 >= l1 or yc-b3<0 or yc+b4+1 >= l2: print('out of bounds!'); return p_samp, 0, [b1, b2, b3, b4]
            
            u_curr = U[xc-b1:xc+b2+1, yc-b3:yc+b4+1, frame]
            v_curr = V[xc-b1:xc+b2+1, yc-b3:yc+b4+1, frame]
            u_currn = (u_curr-Umean[xc])/Cond['Uinf'][0][0]
            
            print('working on rad:', (b1,b2,b3,b4), ' rth-',rth)
            x, y = G.get_xy_rect(u_curr.shape[0], u_curr.shape[1])
            
            p_samp = c_samp
            c_samp = G.doMCMC_V4(u_currn, v_curr, x, y)

            if verbose: G.plot_corner(c_samp, 5000)
            
            F3 = F3check(c_samp, 5000, 98)
            b1, b2, b3, b4 = np.array([b1,b2,b3,b4])+1 
            if not F3: return c_samp, 1, [b1, b2, b3, b4]
            if max(b1, b2, b3, b4) <= rth: continue
            else: print('threshold reached!'); return c_samp, 0, [b1, b2, b3, b4]
                       
def thr(S,th):
    S_ = S.copy()
    S_[S<th] = 0
    return S_

def cornerSol(samp, nburn):
    sol_set = []
    from scipy.signal import find_peaks 

    s_ = samp.chain[:,nburn:,:].reshape(-1,5).T
    #print(len(s_))
    for i in range(len(s_)):
        #loop over parameters

        pt = s_[i]
        y,x_ = np.histogram(pt, bins = 100)
        x = [(x_[i]+x_[i+1])/2 for i in range(len(x_)-1)]
        
        gg = find_peaks(y/np.max(y), height = 0.95 )[0] #prom = 95 (apx.)
        if len(gg) == 0: return -1 #F3 fail check
        sol_set.append(x[gg[0]])
    
    return sol_set

import PODutils 

def calError(props,cen, f, U, V, Cond, Umean, k=1.5):
    #return the difference using r
    s, r, x, y, v = props
    yo, xo = cen
    
    dx = 0.010118516377207676
    #print(r)
    x = int(round(x/dx))
    y = int(round(y/dx))
    
    #print(x,y)
    r*= k
    r = int(round(r/dx))
    #print(r)
    
    x+= xo
    y+= yo
    #print(x, y , r)
    if y-r>=0 and y+r < len(U) and x-r>=0 and x+r+1 < len(U[0]):

        U1 = U[y-r:y+r+1, x-r:x+r+1,f]
        V = V[y-r:y+r+1, x-r:x+r+1,f]
        Un = (U1-Umean[y])/Cond['Uinf']
        
        x1, y1  = G.get_xy_rect(Un.shape[0], Un.shape[1])
        print(x1.shape, Un.shape)
        m_res = PODutils.minfuncVecField10r_rect(props, Un, V, x1, y1)
        return m_res
    else : 
        #print(y-r>=0 , y+r < len(U) , x-r>=0 , x+r+1 < len(U[0]))
        return 0  
    

def DcorrectionV4(Cond, U, V, Umean, props, curr_samp, verbose = False, k1 = 1.5):
    '''
    returns -1 if ut of bounds, reaches threshold. 
    else returns good sampler, bbdims, residue 
    only care about F3 fails! 
    takes increments to exact rectangular BB dims; 
    //instead of increasing radius solely.
    
    corrects D-ivergece of peaks
    I/O: I | best case updated sampler, boolean to indicate success.
    1: success, 0: fail.
    
    
    '''
    import PODutils 
    #props - center, frame, bbdims
    p_samp = curr_samp
    c_samp = curr_samp
    
    
    
    frame = props['frame']
    cent = props['cent']
    bbdims  = props['bbdims']
    
    rth = int(2*max(bbdims))
    yc, xc = cent
    mult = 1.1
    
    b1, b2, b3, b4 = bbdims+1

    l1, l2, l3 = U.shape
    
    while (True):

            #odd len box
            if (b1+b2+1)%2 == 0 : b2+=1 
            if (b3+b4+1)%2 == 0 : b4+=1

            #check bounds
            if xc-b1<0 or xc+b2+1 >= l2 or yc-b3<0 or yc+b4+1 >= l1: print('out of bounds!'); return -1
            
            u_curr = U[yc-b1:yc+b2+1, xc-b3:xc+b4+1, frame]
            v_curr = V[yc-b1:yc+b2+1, xc-b3:xc+b4+1, frame]
            u_currn = (u_curr-Umean[yc])/Cond['Uinf'][0][0]
            
            print('working on rad:', (b1,b2,b3,b4), ' rth-',rth)
            x, y = G.get_xy_rect(u_curr.shape[0], u_curr.shape[1])
            
            p_samp = c_samp
            c_samp = G.doMCMC_V4(u_currn, v_curr, x, y)

            if verbose: G.plot_corner(c_samp, 5000)
            
            F3 = F3check(c_samp, 5000, 98)
            b1, b2, b3, b4 = np.array([b1,b2,b3,b4])+1 
            
            if not F3: 
                samp_sol = cornerSol(c_samp, 5000) #sol from sampler 
                #res_ = PODutils.minfuncVecField10r_rect(samp_sol, u_currn, v_curr, x, y)
                res_ = calError(samp_sol, cent, frame, U, V, Cond, Umean, k = k1)
                return c_samp,[b1, b2, b3, b4], res_
            
            if max(b1, b2, b3, b4) <= rth: continue
            else: print('threshold reached!'); return -1
      

            
def thrm(S,th):
    S_ = S.copy()
    S_[np.abs(S)<th] = 0
    return S_