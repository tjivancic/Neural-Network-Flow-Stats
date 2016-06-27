import sklearn.neural_network as nn
import pandas as pd
import random
import getflow as gf
import getbasin as gb
import networktools as nt
import flowstats as fs
import matplotlib.pyplot as plt

#reads peak streamflow values into a pandas array
flowframe = gf.readPeakFile('peak10yrs')

#calculates the 10 year return period streamflow
Q10 = gf.buildStatFrame(flowframe,fs.N_year_flood,'Q10', False,10)

#reads all of the relevant stats from the Falcone dataset
Stats = gb.readAllStats()

#combines the flood and land stats an retains only sites with both values
YX = pd.concat([Q10,Stats], axis=1, join='inner')

#decides which variables represet the X and Y frames
Xsub = YX.keys()[1:]
Ysub = YX.keys()[0:1]

#Normalizes all X values
YX[Xsub] = nt.normalizecolumns(YX[Xsub])

#shuffles indices so that there is no ordering by site id
(Ngage,Nvar)=YX.shape
indexshuff = random.sample(YX.index,Ngage)

#determines the largest values of Ntrain, Nval, and Ntest that can retain the 
#~3:1:1 ratio
Ntrain_max = Ngage*6/10
Nval_max = Ngage*2/10
Ntest_max = Ngage-Ntrain-Nval

Ntrain_list = [Ntrain_max/(2**i) for i in range(5)]
Nval_list = [Nval_max for n in Ntrain_list]
scores_N = [0 for i in range(5)]


#simple run using arbitrary parameters, best of 3
score_clf  = nt.bestof(3, nt.try_model,YX, indexshuff, 
                    Ntrain_max, Nval_max, (200, 100, 50))
nt.test_model(YX, indexshuff, Ntrain_max, Nval_max, score_clf[1], plot=True)





#this loop takes a while [might be good to parallelize, however it does use a 
#lot of ram]
for i in range(5):
    scores_N[i]= nt.bestof(10, nt.try_model,YX, indexshuff, 
                    Ntrain_list[i], 
                    Nval_list[i], 
                    (200, 100, 50))


#future runs ought to tweak the struture of the network, probably a good idea to
#increase the first layer to equal the number of variables.
    
