import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors

#reading data
data = pd.read_csv("data.csv",low_memory=False)
#data.head()
#data = data.fillna(0)

#extracting required data (data exploration). We have just taken the first 1000 players from data
#data cleaning
req_fields = ['Overall', 'Potential', 'LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB','Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy','LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower','Jumping','Stamina','Strength','LongShots','Aggression','Interceptions','Positioning','Vision','Penalties','Composure','Marking','StandingTackle','SlidingTackle','GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes']

#goalkeepers didnt have values for LS,ST,RS,CB etc
data = data.fillna(0)

#We have just taken the first 1500 players from data
data = data.iloc[0:1500,:]

#data is in the form of range. converting it into integer values
def formulaCalculation(row):
    return eval(str(row[feature]))

for feature in req_fields:
    data[feature] = data.apply(formulaCalculation, axis=1)


req_data = data[req_fields]

#checking how many clusters should the data be divided into. We are using elbow method
sse = []
iter = [1,2,3,4,5,6,7,8,9,10]
for i in iter:
    km = KMeans(n_clusters=i, n_init=10, max_iter=1000)
    km.fit_predict(req_data)
    sse.append(km.inertia_)

plt.plot(iter, sse, '-o')

#applying kmeans for 4 clusters since 4 is the optimum number of clusters from the above graph
km = KMeans(
    n_clusters=4, init='random',
    n_init=10, max_iter=1000, 
    tol=1e-04, random_state=0
)

#predicting kmeans and clustering on req_data
y_km = km.fit_predict(req_data)

#plotting different clusters according to their roles
clusters = []
for i in range(4):
   clusters.append('Cluster ' + str(i+1))
plt.plot(clusters, km.cluster_centers_[:,2], color='blue')
plt.plot(clusters, km.cluster_centers_[:,3], color='blue')
plt.plot(clusters, km.cluster_centers_[:,4], color='blue')
plt.plot(clusters, km.cluster_centers_[:,5], color='blue')
plt.plot(clusters, km.cluster_centers_[:,6], color='blue')
plt.plot(clusters, km.cluster_centers_[:,7], color='blue')
plt.plot(clusters, km.cluster_centers_[:,8], color='blue')
plt.plot(clusters, km.cluster_centers_[:,9], color='blue')
plt.plot(clusters, km.cluster_centers_[:,10], color='blue')
plt.plot(clusters, km.cluster_centers_[:,11], color='blue')
plt.plot(clusters, km.cluster_centers_[:,12], color='blue')

plt.xlabel('Attacking features')
plt.show()

plt.plot(clusters, km.cluster_centers_[:,13], color='skyblue')
plt.plot(clusters, km.cluster_centers_[:,14], color='skyblue')
plt.plot(clusters, km.cluster_centers_[:,15], color='skyblue')
plt.plot(clusters, km.cluster_centers_[:,16], color='skyblue')
plt.plot(clusters, km.cluster_centers_[:,17], color='skyblue')
plt.plot(clusters, km.cluster_centers_[:,18], color='skyblue')
plt.plot(clusters, km.cluster_centers_[:,19], color='skyblue')
plt.plot(clusters, km.cluster_centers_[:,20], color='skyblue')
plt.plot(clusters, km.cluster_centers_[:,21], color='skyblue')
plt.xlabel('Midfielding features')
plt.show()

plt.plot(clusters, km.cluster_centers_[:,22], color='red')
plt.plot(clusters, km.cluster_centers_[:,23], color='red')
plt.plot(clusters, km.cluster_centers_[:,24], color='red')
plt.plot(clusters, km.cluster_centers_[:,25], color='red')
plt.plot(clusters, km.cluster_centers_[:,26], color='red')
plt.plot(clusters, km.cluster_centers_[:,27], color='red')
plt.xlabel('Defending features')
plt.show()


#GK
plt.plot(clusters, km.cluster_centers_[:,-1], color='green')
plt.plot(clusters, km.cluster_centers_[:,-2], color='green')
plt.plot(clusters, km.cluster_centers_[:,-3], color='green')
plt.plot(clusters, km.cluster_centers_[:,-4], color='green')
plt.plot(clusters, km.cluster_centers_[:,-5], color='green')

plt.xlabel('Goal Keeping features')
plt.show()

#getting the cluster-wise playesrs for analysis of each cluster
mydict = {i: np.where(km.labels_ == i)[0] for i in range(km.n_clusters)}

player_to_be_replaced = 'Cristiano Ronaldo'

#getting all the players from the same cluster
#assume we want the best replacement for Cristiano Ronaldo

#Have to specify the name in the below line for the player to be replaced.
#Also the name should exactly match the one in the csv original data file.


def player_replacement(player_to_be_replaced, req_data, data):



    players_location = req_data.loc[data['Name'] == player_to_be_replaced]
    player_number = players_location.index.values[0]
    req_cluster = km.labels_[player_number]
    player_ovl = req_data.iloc[player_number, 0]
    player_team = data.iloc[player_number,9]
    player_dist_from_centroid = km.transform(req_data)[player_number,km.labels_[player_number]]
    #get all players from the above cluster

    dictlist = []

    for key, value in mydict.items():
        j=0
        player_names = []
        #temp = [key,value]
        for i in value:
            if(player_number == i):
                k = j
            player_names.append(data.iloc[i,2])
            j = j+1
        temp = [key, player_names]
        dictlist.append(temp)
    #dictlist[req_cluster]

    #getting data of required cluster
    req_cluster_data = []
    for i in range(req_data.iloc[:,1].size):
        if km.labels_[i] == req_cluster:
            req_cluster_data.append(i)

    data_clus_req = []
    for i in req_cluster_data:
        data_clus_req.append(data.loc[i,['Name', 'Club', 'Contract Valid Until', 'Overall', 'Potential', 'LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB','Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy','LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower','Jumping','Stamina','Strength','LongShots','Aggression','Interceptions','Positioning','Vision','Penalties','Composure','Marking','StandingTackle','SlidingTackle','GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes']])
        #data_clus0.append(req_data.iloc[i,:])
    data_clus_req = pd.DataFrame(data_clus_req, columns=['Name', 'Club', 'Contract Valid Until', 'Overall', 'Potential', 'LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB','Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy','LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower','Jumping','Stamina','Strength','LongShots','Aggression','Interceptions','Positioning','Vision','Penalties','Composure','Marking','StandingTackle','SlidingTackle','GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes'])



    #taking care of NaN values
    data_clus_req = data_clus_req.fillna(0)
    #data_clus_req



    #Apply k-nearest neighbour to find the closest player as a replacement in the same cluster
    knn = NearestNeighbors(n_neighbors=8)
    knn.fit(data_clus_req.iloc[:,3:64])




    #get the replacement
    player_index = k
    replacements = []
    nearest_neighbors = knn.kneighbors(data_clus_req.iloc[player_index:player_index+1,3:64], return_distance=False)
    for i in nearest_neighbors:
        
        for j in i:
            if(data_clus_req.iloc[j,0] != data_clus_req.iloc[player_index,0]):
                #print(j)
                replacements.append(data_clus_req.iloc[j,0:3])
                #print(data_clus_req.iloc[j,0] + "---"+ data_clus_req.iloc[j,1] + "---" +data_clus_req.iloc[j,2])
    replacements = pd.DataFrame(replacements, columns=['Name', 'Club', 'Contract Valid Until'])
    return replacements
    
replacements = player_replacement(player_to_be_replaced, req_data, data)
print("Following players are the potential replacements of "+player_to_be_replaced)
replacements

#reading the evaluation set which was collected from different people
evaluation_set = pd.read_csv("841_evaluation_set.csv",low_memory=False)

#running the player replacement model for the players in evaluation set
#checking for the efficiency of the model
count_true = 0
count_total = 0
for player_test in evaluation_set.iterrows():
#     print(type(player_test[1]['Transferred player']))
#     print(player_test[1]['Transferred player'])
#    print(type(player_test[1]['Replacement']))
    replacements = player_replacement(player_test[1]['Transferred player'], req_data, data)
#    print(type(replacements.Name))
    if(player_test[1]['Replacement'] in replacements.Name.values):
        count_true+=1
        count_total+=1
    else:
        count_total+=1

efficiency = (count_true/count_total)*100
efficiency
