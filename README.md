# Neural-Network-Flow-Stats
Builds a neural network to predict the n-year return period for streams with data given in the Falcone 2010 dataset.

Neural-Network-Flow-Stats uses the Falcone 2010 dataset of watershed features as input to a neural network model to predict n-year return period streamflow. It uses an annual peakstreamflow dataset retrieved from the USGS for all sites with more than 10 years of peakflows recorded as training data.

Motivation:
  Flood planning often involves caclculating the 100 year flowrate, Q100, which is the largest streamflow volume expected in any given period of 100 years. Historically a Log-Pearson III distribution of annual maximum streamflow is used to estimate Q100 at river sites. 
  However, many sites do not have sufficiently long records of maximum streamflow to fit a distribution. In those cases regional regression models based on the estimated Q100 of nearby river sites are often used. A regional regression model is a linear or logarithmic regression model typically utilizing a few significant watershed variables such as landcover statitistics, watershed area estimates. The models are calibrated using Q100 estimates from nearby measured sites.  In regional regression modeling the modeller must specifically select the regional sites to include, a limited number of watershed variables and explicitly account for interactions between variables. 
  Here, I instead use a neural network model for Q100. A neural network has a few advantages:
  1) Regularization can prevent overfitting even when using a large number of variables
  2) Using geographic/regional classification variables the network can determine on its own what nearby sites are relevant
  3) Interactions between varaibles are built in to the model and depend only on the geometry of the network


Functional Description:
floodnetwork.py shows an example methodology, the gist is this:
getbasin.readAllStats reads from the Falcone files extracted to './gages_basinchar_sept3_09/' and organizes the data into a pandas dataframe indexed by USGS id number and separates classification variables into binary variables. 

getflow.readPeakFile reads the modified (header removed) USGS file into a pandas dataframe

getflow.buildStatFrame calculates the n-year return period for each stream in the USGS file using a Log-Pearson III distribution

pd.concat([Q100,Stats], axis=1, join='inner') combines the lists to create the training data keeping only the sites with both Q100 and watershed data

nt.normalizecolumns normalizes each of the variables such that they hace a mean of 0 and range of 1

the rest of the file determines the best neural network structure by training on the combined dataset
