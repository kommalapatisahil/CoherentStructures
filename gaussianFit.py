
def cornerSol_VG2(mc_samp, verb = False):
    
    '''
    Calculates the relative height of the 
    tallest peak.
    
    Then takes the gaussian fit for the top 50%
    of that data. 
    
    If not too small takes the top 100%.
    
    returns the mean of the gaussian fit as the 
    converged mc solution for that parameter. 
    
    Up1 - Includes try-except to take care of 
    unseen errors in curve fit. 
    
    Up2 - recursively decreases prominence if 
    the signal is too noisy. 
    
    '''
    import pylab as plb
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    
    def gaus(x,a,x0,sigma):
            #gaussian dist funciton
            #takes x, amplitude, mean, variance, 
            #returns the point on the gaussian curve. 
            
            return a*np.exp(-(x-x0)**2/(2*sigma**2))
    
    def gfit(x, y, res = 0):
        #third stop
        #gaussian fit happens here 
        #uses non linear least squares for gaussian fit. 
        #using curve fit.
        #returns fitted X, Y for plotting fit
        #and also fit parameters - mean, variance, peak amplitude 
        
        mean = x[np.argmax(y)]
        sigma = (x[1]-x[0])
        
        try:
            popt,pcov = curve_fit(gaus,x,y,
                              p0=[np.max(y)/2, mean, sigma],
                              bounds = ([0,np.min(x),0],[np.max(y), np.max(x), x[-1]-x[0]]),
                              maxfev = 50000)

        except:
            print('Curve fit is a fucking bitch.')
            le = len(x) if res == 0 else res
            x_ = np.linspace(x[0], x[-1], le)
            popt = [np.max(y),mean, sigma]
            return  x_, gaus(x_,*popt), popt

        le = len(x) if res == 0 else res
        
        x_ = np.linspace(x[0], x[-1], le)
        
        return  x_, gaus(x_,*popt), popt
    
    def promv2(c,d, plim = 0.3):
        #second stop
        #gets the relative height of the tallest peak. 
        #returns the left, right indices of the peak
        #and the actual location. 
        
        p1s = []
        p4= find_peaks(c/np.max(c), prominence = plim)[0]
        p1s.extend(p4)

        p2s  = []
        p2s.append(d[0])
        p3 = find_peaks(-c/np.max(c), prominence = plim)[0]
        p2s.extend(p3)
        p2s.append(d[-1])
        
        #this happens when the function is pretty much diverging. 
        if len(c[p1s]) == 0:
            print(f'Prom thr too high - no peaks found; reducing thr from {plim} to {plim/2}')
            return promv2(c,d,plim = plim/2)
        
        b1 = np.where(c==np.max(c[p1s]))[0][0] #bl - big peak location
        bl2 = np.sum(p2s-b1<0)
        bl, br = p2s[bl2-1:bl2+1] #big peak left, right 

        return bl, br, b1    
    
    def CGfit2(x_,y, ii = 'C'):
        #first stop
        #gets the appropriate bounds
        #then gets the gaussian fit and 
        #returns mean.
        
        x = np.arange(len(x_))
        wtf= promv2(y,x)
        bl, br, bp = wtf

        th = len(x)*10//100 

        if (br-bl) >= th: 

            if (br-bl)//2 >= th: 
                #50%
                bl2 = bl+ (bp-bl)*3//4
                br2 = br-(br-bp)*3//4

            else: 
                #100%
                bl2,br2 = bl, br
        else:     
            bl2,br2 = bl, br
            #raise flag 
            print(bl2, br2, ii)
            print("Gaussian Fit flag #1, 100% fit not enough for 15%")
            print("Continuing with 100%")

        
        gg=  gfit(x_[bl2:br2+1], y[bl2:br2+1], res  = 1000)
        
        if verb:
                plt.figure()
                plt.plot(x_, y, '-b', label = 'true')
                plt.plot(x_[bl2:br2+1], gaus(x_[bl2:br2+1], *gg[-1]),'-r', label = 'fit')
                plt.grid(alpha = 0.2)
                plt.legend()
        
        return gg[-1][1]
    
    #go over each parameter 
    mc_sol = []
    names = ['CS', 'R', 'X', 'Y', 'CV']

    mc_flat =  mc_samp.get_chain(discard=5000,flat=True)
    #mc_flat = mc_samp.copy()
    #print(mc_samp.shape)
    for i in range(5):
        
        pi = mc_flat[:,i]
        yh, x_ = np.histogram(pi, bins = 250)
        xh = np.array([(x_[j+1]+x_[j])/2 for j in range(len(x_)-1)])
        mc_sol.append(CGfit2(xh, yh,  ii = names[i]))
        if verb: plt.title(names[i]+" -" + str(round(mc_sol[-1],5)))
           
    return mc_sol
