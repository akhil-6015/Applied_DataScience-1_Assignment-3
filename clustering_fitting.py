#importing libraries to handle the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as mtplt
import seaborn as sea
from scipy.optimize import curve_fit

#import the KMeans methods from the sklearn.cluster library 
from sklearn.cluster import KMeans



#create object to store the dataset in the dataframe using pandas library      
data_f1 = pd.read_csv("clusteringFittingData.csv", skiprows = 3)
print(data_f1)

#create an object and modify the dataset 
data_f = data_f1.drop(columns = ["1960","1961","1962","1963","1964",'1965',"1966","1967","1968",'1969',"1970",'1971',"1972",
                                 "1973","1974","1975","1976","1977","1978","1979","1980","1981","1982","1983","1984","1985",
                                 "1986","1987","1988","1989","1990","1991","1992","1993",'1994',"1995","1996","1997","1998",
                                 "1999","2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010"])
                      
                      
                         
#using drop function to remove unwanted data 
data_f = data_f.drop(columns = ["Unnamed: 66"])

#print the DataFrame to see the changes 
print(data_f)

#using isnull fucntion to find the null values in the data and count the total number of null values. 
data_f.isnull().sum()

#fillna fuction use for handel the null values in the data set  
data_f = data_f.fillna(0)

#again using isnull function and sum function to know that above code was work properly or not. 
data_f.isnull().sum()

#describe function use to describe the dataset 
data_f.describe()

df2 = data_f
df2 = df2.drop(["Country Code","Indicator Code"],axis =1)
df2.set_index("Indicator Name",inplace = True)
df2 = df2.loc["Agricultural land (% of land area)"]
df2 = df2.reset_index(level = "Indicator Name")
df2.groupby(["Country Name"]).sum()
print(df2)

df3 = df2.head(15)
df3.plot.bar(x= "Country Name",y = ['2011', '2013', '2015', '2017', '2019', '2021'],figsize = (15,5),edgecolor = "white")
mtplt.title("Agricultural land (% of land area)")
mtplt.show()

cleaned_df2=df2.drop(columns =["Indicator Name","Country Name","2021"],axis =1 )
print(cleaned_df2)

cleaned_df2.info()

cleaned_df2.isnull().sum()

kmeans_app = KMeans(n_clusters=5)

#fit() use the da_fr_norm dataframe and fit the dataframe and store the result of the kmeans
kmeans_app.fit(cleaned_df2)

#create object label and find fit_pedect() function
label = kmeans_app.fit_predict(cleaned_df2)
print(label)

#create object cluster_data and create the data frame
cluster_data = pd.DataFrame(label)
print(cluster_data)


#use concat() function to join two data set 
clus_data = pd.concat([cleaned_df2, cluster_data], axis=1, join='inner')
clus_data = clus_data.rename(columns={0: 'Cluster'})
print(clus_data)

#print the results of the cluster_centers of the kmeans_app data 
print(kmeans_app.cluster_centers_)

#filter rows of original data
filtered_label0 = cleaned_df2[label == 0]
fig = mtplt.figure(figsize = (15,9))

#plotting the results
sea.scatterplot(data =cleaned_df2, x="2012",y= "2020",hue = label)
mtplt.show()

#visualize dataframe with centroides
fig = mtplt.figure(figsize = (15,9))

#plot the scatter plot using seaborn and scatterplot()function
sea.scatterplot(data =cleaned_df2, x="2012",y= "2020",hue = label)
x = kmeans_app.cluster_centers_[:,0]
y = kmeans_app.cluster_centers_[:,1]

#plot the scatter plot with centroids
mtplt.scatter(x,y,marker = "o",c = "r",s = 90,label = "centroids")

#shows the labelling of the graph data
mtplt.legend()
mtplt.show()

"""
    This function is used to fit a curve to the dataset using the KMeans clustering algorithm. 
    It takes in the cluster centers, which are the x and y coordinates of the centroids, and uses a sine model 
    to fit the data, as well as output the coefficients and covariance of the coefficients. 
    It also plots the centroids and the fitted curve.

    Parameters:
        a (list): list of x coordinates of the centroids
        x (float): the coefficient of the sine function
        y (float): the coefficient of the sine function
        z (float): the coefficient of the sine function
        u (float): the coefficient of the sine function

    Returns:
        ans (list): list of fitted y coordinates
    """
def curvve_fitter():
    a = kmeans_app.cluster_centers_[:,0]
    b = kmeans_app.cluster_centers_[:,1]

    def model_test(a,x,y,z,u):
    
        return abs(x*np.sin(-y*a)+(z-u))
        
    param, param_cov = curve_fit(model_test, a, b)
    print("Sine function coefficients:")
    print(param)
    print("Covariance of coefficients:")
    print(param_cov)
    ans = abs(param[0]*(np.sin(param[1]*a)))
    fig = mtplt.figure(figsize = (15,9))
    sea.scatterplot(data =cleaned_df2, x="2012",y= "2020",hue = label)
    mtplt.plot(a, b, 'o', color ='red', label ="centroids")
    mtplt.plot(a, ans, '--',linewidth = 4, color ='blue', label ="curve_fit")
    mtplt.legend()
    mtplt.show()
curvve_fitter()





