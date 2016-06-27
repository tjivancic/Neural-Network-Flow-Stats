import numpy as np
import scipy.stats as st


def N_year_flood(Qvec, N=100, method='log-pearson-III'):
    """
    uses, by default, a 'log-pearson-III' distribution to predict the N year 
    return period streamflow given a list of peak streamflow value
    """
    if method=='log-pearson-III':
        if N > 100:
            print 'Warning: This implementation is considered valid only for N<=100'
        Ny = len(Qvec)
        p = 1.0-1.0/N
        z = st.norm.ppf(p)
        Y = np.log(Qvec)
        Ybar=np.mean(Y)
        SY = np.std(Y, ddof=1)
        gamma = Ny*sum((Y-Ybar)**3)/(Ny-1)/(Ny-2)/(SY**3)
        K = 2.0/gamma*(1.0+gamma*z/6.0-gamma**2/36.0)**3-2.0/gamma
        Yn = Ybar+K*SY
        return np.exp(Yn)
    else:
        raise ValueError("your method is not defined edit" +
                         " flowstats.N_year_flood or use"+ 
                         " an existing method")




#numyears=40
#Qvec = np.array([np.exp(random.gauss(0,1)) for i in range(numyears)])
#Q100 = N_year_flood(100, Qvec)
