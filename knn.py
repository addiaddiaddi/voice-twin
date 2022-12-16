import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import json
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, pairwise
from sklearn.manifold import TSNE
import seaborn as sns

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

warnings.simplefilter(action='ignore', category=FutureWarning)

raw_data = json.loads(open('demo.json').read())

# highlow.json.bak - theclassall.json.bak - demo.json

#raw_data = {"addison": [[1,2,3,4],[5,6,7,8]], "bro": [[3,3,3,3],[1,1,1,1]]}
df = pd.DataFrame()


#print(df.head())

for name in raw_data.keys():
    #print("len: raw_data[name]",len(raw_data[name]))
    for d in raw_data[name]:
        row = {'name': name}

        for i in range(len(d[0])):
            row[d[0][i]] = d[1][i]
        #print(row)
        df = df.append(row,ignore_index=True)

#fprint(row)

print()

labels = df['name'].tolist()
# print(labels)
df.drop(['name'], axis = 1, inplace = True)
df.drop(df.columns[np.arange(1000,4096)],axis=1,inplace=True)
# print(df.head())


scaler = MinMaxScaler()
y = labels
X_pre = df.values.tolist()
X = []

for x_pre in X_pre:
    maxval = max(x_pre)
    x = []
    for i in x_pre:
        x.append(i/maxval)
    X.append(x)


plt.scatter(np.arange(0,len(X[12])),X[12],s=1)


plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=2, p=1,
                     weights='uniform')

predictions = knn.predict(X_test)
print(predictions,y_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(knn.kneighbors_graph(X_test,n_neighbors=None, mode='distance'))
# # print(knn.kneighbors(X_test[1].reshape(1,-1)))
# # print(knn.kneighbors(X_test[2].reshape(1,-1)))
# print()
# print(pairwise.euclidean_distances(X_test[0].reshape(1,-1),X_train[11].reshape(1,-1)))
import time
distance_matrix = []

min_dist = 999999
max_dist = 0

for i in X:
    distances = []

    for j in X:
        distance = pairwise.euclidean_distances(np.array(i).reshape(1,-1),np.array(j).reshape(1,-1))
        #str_num = str(round((distance[0][0]-3)/2,2))

        max_dist = max([max_dist,distance[0][0]])
        if distance[0][0] > 0.1:
            min_dist = min([min_dist,distance[0][0]])

        str_num=str(round(distance[0][0],2))
        if len(str_num) == 3:
            str_num += '0'
        distances.append(str_num)

    distance_matrix.append(distances)

print("",end='     ')
for l in y:

    print(l[0:4], end = '   ')

    if l == 'ben':
        print(' ',end="")
print()

for i in range(len(distance_matrix)):
    print(y[i][0:4],end =' ')
    if y[i] == 'ben':
        print(' ',end="")
    for j in range(len(distance_matrix)):

            dist = float(distance_matrix[i][j])
            if dist == 0:
                print('0.00',end='   ')
                continue
            pct = ((dist - min_dist) / (max_dist-min_dist))
            R = int(255 * pct)
            G = int(255 * (1-pct))
            #
            # print(colored(R,G,0,distance_matrix[i][j]),end='  ')
            pct = str(round(pct,2))
            if len(pct) ==3:
                pct+='0'
            print(colored(R,G,0,pct) ,end='  ')

    print()
print()
print("",end='     ')
for l in y:

    print(l[0:4], end = '   ')

    if l == 'ben':
        print(' ',end="")
print()
for i in range(len(distance_matrix)):
    print(y[i][0:4],end =' ')
    if y[i] == 'ben':
        print(' ',end="")
    for j in range(len(distance_matrix)):

            dist = float(distance_matrix[i][j])
            if dist == 0:
                print('0.00',end='   ')
                continue
            pct = ((dist - min_dist) / (max_dist-min_dist))
            R = int(255 * pct)
            G = int(255 * (1-pct))
            print(colored(R,G,0,distance_matrix[i][j]) ,end='  ')

    print()

import matplotlib.pyplot as plt
import networkx as nx
import random

start = time.time()
G = nx.Graph()
edge_weights = []
a = []
for Y in y:
    a.append(Y+str(random.randint(1,100)))

y = a
for i in range(len(distance_matrix)):
    for j in range(len(distance_matrix)):
        if i == j:
            # print('0.00',end="  ")
            continue
        diff = max_dist - min_dist
        weight = float(distance_matrix[i][j]) - min_dist
        weight = diff - weight
        weight = round(weight,2)

        edge_weights.append(weight)
        G.add_edge(y[i],y[j],weight=weight)
    #     print(round(float(weight),2),end= "  ")
    # print()

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] is not None]

pos = nx.spring_layout(G, seed=1,k=1)  # positions for all nodes - seed for reproducibility

# nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1)

# node labels
nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
# edge weight labels
# edge_labels = nx.get_edge_attributes(G, "weight")
# nx.draw_networkx_edge_labels(G, pos, edge_labels,font_size=4)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()

plt.show()

#------------------------------------------
#import spring_3D
import plotly.graph_objects as go

for i in range(1):
    spring_3D = nx.spring_layout(G, dim = 3, k = 0.5,seed=1) # k regulates the distance between nodes
    # weights = [G[u][v]['weight'] for u,v in edges]
    # nx.draw(G, with_labels=True, node_color='skyblue', font_weight='bold',  width=weights, pos=pos)

    # we need to seperate the X,Y,Z coordinates for Plotly
    # NOTE: spring_3D is a dictionary where the keys are 1,...,6
    x_nodes= [spring_3D[key][0] for key in spring_3D.keys()] # x-coordinates of nodes
    y_nodes = [spring_3D[key][1] for key in spring_3D.keys()] # y-coordinates
    z_nodes = [spring_3D[key][2] for key in spring_3D.keys()] # z-coordinates

    #we need to create lists that contain the starting and ending coordinates of each edge.
    x_edges=[]
    y_edges=[]
    z_edges=[]

    #create lists holding midpoints that we will use to anchor text
    xtp = []
    ytp = []
    ztp = []

    #need to fill these with all of the coordinates
    edges = G.edges()

    for edge in edges:
        #format: [beginning,ending,None]
        x_coords = [spring_3D[edge[0]][0],spring_3D[edge[1]][0],None]
        x_edges += x_coords
        xtp.append(0.5*(spring_3D[edge[0]][0]+ spring_3D[edge[1]][0]))

        y_coords = [spring_3D[edge[0]][1],spring_3D[edge[1]][1],None]
        y_edges += y_coords
        ytp.append(0.5*(spring_3D[edge[0]][1]+ spring_3D[edge[1]][1]))

        z_coords = [spring_3D[edge[0]][2],spring_3D[edge[1]][2],None]
        z_edges += z_coords
        ztp.append(0.5*(spring_3D[edge[0]][2]+ spring_3D[edge[1]][2]))


    etext = [f'weight={w}' for w in edge_weights]

    trace_weights = go.Scatter3d(x=xtp, y=ytp, z=ztp,
        mode='markers',
        marker =dict(color='rgb(125,125,125)', size=1), #set the same color as for the edge lines
        text = etext, hoverinfo='text')

    #create a trace for the edges
    trace_edges = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode='lines',
        line=dict(color='black', width=2),
        hoverinfo='none')

    #create a trace for the nodes
    trace_nodes = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode='markers',
        text=list(spring_3D.keys()),
        marker=dict(symbol='circle',
                size=10,
                color='skyblue')
        )

    #Include the traces we want to plot and create a figure
    data = [trace_edges, trace_nodes, trace_weights]
    print('endtime', time.time()-start)
    fig = go.Figure(data=data)

    fig.show()
