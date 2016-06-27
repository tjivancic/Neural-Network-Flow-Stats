import pandas as pd
import flowstats as fs
#def getPeakFileFromUSGS():

def readPeakFile(peakfile):
    """
    reads a USGS peak streamflow file and returns a pandas dataframe
    """
    flowcsv = pd.read_csv(peakfile, sep='\t')
    flowframe = flowcsv[['site_no','peak_dt','peak_va']]
    flowframe = flowframe[flowframe['peak_va'].notnull()]
    return flowframe


def buildStatFrame(flowframe, statfunc=fs.N_year_flood, colname = 'Q100', subset=False, *args):
    """
    given a peak flow dataframe, applies statfunc to for subsets sharing a value
    of 'site_no' (presumably statfunc calculates some relevant statistical value,
    by default it computes the 100 year streamflow as predicted by a log pearson
    type III distribution) returns1 the values as a pandas dataframe
    """
    if subset==False:
        subset = flowframe['site_no'].unique()
    framelist = []
    for sub in subset:
        subframe = flowframe['peak_va'][flowframe['site_no']==sub]
        if len(subframe)>10 and min(subframe) > 0.0:
            framelist.append([sub, statfunc(subframe,*args)])
    data = pd.DataFrame(data=framelist, columns = ['site_no',colname])
    data.set_index('site_no',inplace=True)
    return data

