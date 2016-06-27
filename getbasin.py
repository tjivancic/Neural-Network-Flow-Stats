import pandas as pd
import numpy as np
folder = 'gages_basinchar_sept3_09/'

def classtobinary(data, classname):
    """
    takes input data, pandas dataframe and column name ,classname, and creates 
    a new logic column for each of n unique entry in the column named 
    <classname><n>
    """
    classvals = np.unique(data[classname])
    for i in range(len(classvals)):
        data[classname+str(i)] = 0 
        data[classname+str(i)][data[classname]==classvals[i]] = 1
    data.drop(classname,axis=1,inplace=True)
    return data

def read_falc_stats(filename,varlist,isclass, errorval=-999, setto=0):
    """
    given a filename from the falcone dataset, list of interesting variables, 
    and a logical vecotr telling if each variable is a classification variable, 
    returns the requested data as a pandas array
    """
    rawdata = pd.read_csv(folder+filename, sep='\t')
    data = rawdata.loc[:,['GAGE_ID']+varlist]
    for i in range(len(isclass)):
        if isclass[i]:
            data = classtobinary(data,varlist[i])
    data[data==errorval] = setto
    data.set_index('GAGE_ID', inplace=True)
    return data

def get_falc_headers(filename):
    """
    given a filename from the falcone dataset, returns a list of the headers
    less the site id
    """
    fin = open(folder+filename,'r')
    headline = fin.readline()
    fin.close()
    return headline.split()[1:]


def readAllStats():
    """
    reads all interesting data from the Falcone dataset and returns it as a 
    pandas array
    """
    if not os.path.exists(folder):
        try:
            os.system('tar -zxvf ' + folder[:-1] + '.tar.gz')
        except:
            raise NameError(folder+'does not exist. Confirm that the Falcone data is in the right place')
    data = pd.concat([read_falc_stats(filename = 'bas_classif.txt',
                    varlist = ['CLASS', 'AGGECOREGION', 'HYDRO_DISTURB_INDX', 
                         'NYEARS'],
                    isclass = [True,True,False,False]),
                 
                 read_falc_stats(filename = 'basinid.txt',
                    varlist = ['DRAIN_SQKM', 'LATITUDE', 'LONGITUDE'],
                    isclass = [False,False,False]),
                 
                 read_falc_stats(filename = 'bas_morph.txt',
                    varlist = ['BAS_COMPACTNESS'],
                    isclass = [False]),
                 
                 read_falc_stats(filename = 'climate.txt',
                    varlist = get_falc_headers('climate.txt'),
                    isclass = [False for i in get_falc_headers('climate.txt')]),
                 
                 read_falc_stats(filename = 'geology.txt',
                    varlist = ['GEOL_REEDBUSH_DOM','GEOL_REEDBUSH_DOM_PCT',
                    'GEOL_REEDBUSH_SITE','GEOL_HUNT_DOM_CODE','GEOL_HUNT_DOM_PCT',
                    'GEOL_HUNT_DOM_DESC','GEOL_HUNT_SITE_CODE'],
                    isclass = [True,False,True, True, False, True, True]),
                 
                 read_falc_stats(filename = 'hydro.txt',
                    varlist = get_falc_headers('hydro.txt'),
                    isclass = [False for i in get_falc_headers('hydro.txt')]),
                 
                 read_falc_stats(filename = 'hydromod_dams.txt',
                    varlist = get_falc_headers('hydromod_dams.txt'),
                    isclass = [False for i in 
                        get_falc_headers('hydromod_dams.txt')], 
                    errorval=-999, setto = 0.0),
                 
                 read_falc_stats(filename = 'hydromod_other.txt',
                    varlist = get_falc_headers('hydromod_other.txt'),
                    isclass = [False for i in 
                        get_falc_headers('hydromod_other.txt')], 
                    errorval=-999, setto = 0.0),
                 
                 read_falc_stats(filename = 'infrastructure.txt',
                    varlist = ['ROADS_KM_SQ_KM','NLCD01_IMPERV_PCT'],
                    isclass = [False,False]),
                 
                 read_falc_stats(filename = 'landscape_pat.txt',
                    varlist = ['FRAGUN_WATERSHED'],
                    isclass = [False]),
                 
                 read_falc_stats(filename = 'lc01_basin.txt',
                    varlist = get_falc_headers('lc01_basin.txt'),
                    isclass = [False for i in 
                        get_falc_headers('lc01_basin.txt')]),
                 
                 read_falc_stats(filename = 'lc01_mains100.txt',
                    varlist = get_falc_headers('lc01_mains100.txt'),
                    isclass = [False for i in 
                        get_falc_headers('lc01_mains100.txt')]),
                 
                 read_falc_stats(filename = 'lc01_mains800.txt',
                    varlist = get_falc_headers('lc01_mains800.txt'),
                    isclass = [False for i in 
                        get_falc_headers('lc01_mains800.txt')]),
                 
                 read_falc_stats(filename = 'lc01_rip100.txt',
                    varlist = get_falc_headers('lc01_rip100.txt'),
                    isclass = [False for i in 
                        get_falc_headers('lc01_rip100.txt')]),
                 
                 read_falc_stats(filename = 'lc01_rip800.txt',
                    varlist = get_falc_headers('lc01_rip800.txt'),
                    isclass = [False for i in 
                        get_falc_headers('lc01_rip800.txt')]),
                 
                 read_falc_stats(filename = 'nutrient_app.txt',
                    varlist = get_falc_headers('nutrient_app.txt'),
                    isclass = [False for i in 
                        get_falc_headers('nutrient_app.txt')]),
                 
                 read_falc_stats(filename = 'pest_app.txt',
                    varlist = get_falc_headers('pest_app.txt'),
                    isclass = [False for i in 
                        get_falc_headers('pest_app.txt')]),
                 
                 read_falc_stats(filename = 'prot_areas.txt',
                    varlist = get_falc_headers('prot_areas.txt'),
                    isclass = [False for i in 
                        get_falc_headers('prot_areas.txt')]),
                 
                 read_falc_stats(filename = 'soils.txt',
                    varlist = get_falc_headers('soils.txt'),
                    isclass = [False for i in 
                        get_falc_headers('soils.txt')]),
                 
                 read_falc_stats(filename = 'topo.txt',
                    varlist = get_falc_headers('topo.txt'),
                    isclass = [False for i in 
                        get_falc_headers('topo.txt')])
                 
#                 read_falc_stats(filename = 'reach.txt',
#                 varlist = ['Reach_FTYPE'],
#                 isclass = [True])
                ], axis=1)
    return data


