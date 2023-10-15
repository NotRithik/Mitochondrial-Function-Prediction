# MitoCode_Functions

import pandas as pd
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog


# Just a function to open a folder dialog
def open_folder_dialog(hd = 'Select Folder'):
    app = QApplication([])
    options = QFileDialog.Options()
    options |= QFileDialog.ShowDirsOnly  # Only allow selecting directories/folders
    folder_path = QFileDialog.getExistingDirectory(None, hd, options=options)
    # close dialog box
    app.quit()
    return folder_path

# function to open a file dialog
def open_file_dialog(hd = 'Select File'):
    app = QApplication([])
    options = QFileDialog.Options()
    # options |= QFileDialog.ShowDirsOnly  # Only allow selecting directories/folders
    file_path = QFileDialog.getOpenFileName(None, hd, options=options)
    app.quit()
    return file_path



class node: 
    def __init__(self, dat, pointer, dist, conn): 
        self.start = dat 
        self.next = pointer
        self.dist = dist
        self.conn = conn
# class for a node

def Dist(pointOne, pointTwo): # get the distance between two points
    a = 0 
    for i in range(len(pointOne)):  # this allows us to consider it for an n dimensional array
        a += (pointOne[i] - pointTwo[i])**2
    return np.sqrt(a)

def GetData(k): 
    dir = k
    for file in os.listdir(dir):
        if file.endswith(".txt"):
            skelePos = pd.read_table(os.path.join(dir, file))
        if file.endswith(".gnet"):
            nodeDist = pd.read_table(os.path.join(dir, file)) # a list of the nodes and the distance b/w them
        if file.endswith(".coo"): 
            nodeList = pd.read_table(os.path.join(dir, file)) # a list of the nodes and their coordinates
        if file.endswith(".cc"): 
            compList = pd.read_table(os.path.join(dir, file)) # a list of the nodes and their coordinates
    return nodeDist, nodeList, skelePos, compList

def findDistofNodes(node1, node2, temp):  # temp is a clean node distance thing
    t1 = temp.loc[(temp.X == node1) & (temp.Y == node2)]
    if len(t1) < 1:
        t1 = temp.loc[(temp.X == node2) & (temp.Y  == node1)]
    if len(t1) < 1:
        print("Node's are not connected directly, so no distance between them") 
        return None
    return t1.dist.values[0]


# explain what findLength does

def findLength(test, temp): 
    # test = bigDf1.loc[bigDf1.cc == k]
    linids = []
    lens = []
    endparts = []

    for i in set(test.line_id): 
        secTest = test.loc[test.line_id == i]
        b = secTest.loc[secTest.nodeState == True]
        if len(b) < 2: 
            endparts.append(secTest)
        else: 
            linids.append(b)
        
    for i in linids: 
        nodesInid = i.node.values
        lens.append(findDistofNodes(nodesInid[0], nodesInid[1], temp))

    for j in endparts: 
        lens.append(findLength2(j))
    
    return np.sum(lens)

def findAvgWidth(df): 
    widths = df['width_(um)'].values
    return np.mean(widths)

def findNumberofNodes(df): 
    return len(df.loc[df.nodeState == True])

def findPixelIntensity(df): 
    pixint = df.pix_ratio.mean()
    return pixint

def findVolume2(df): 
    vals = df[['x', 'y', 'z']].values
    widths = df['width_(um)'].values
    volume = []
    for i in range(len(vals) - 1):
        diff_of_vals = vals[i + 1] - vals[i]
        avg_width = (widths[i + 1] + widths[i])/2
        volume.append(np.sqrt((diff_of_vals * avg_width)**2))
    return np.sum(volume)

def findLength2(df): # find's length of the entire thing
    import numpy as np 

    X = set(df.line_id.values)
    bigDisList = []
    for i in X: 
        bigDisList = []
        tf = df.loc[df.line_id == i]
        vals = df[['x', 'y', 'z']].values
        disList = []
        for i in range(len(vals) - 1): 
            disList.append(Dist(vals[i], vals[i + 1]))
        bigDisList.append(np.sum(disList))
    return np.sum(bigDisList)

#################################################################
# Calculcations
#################################################################

import pandas as pd
import matplotlib.pyplot as plt

def getVolumeAndGroup(nodeTree, compList): # adds volume and group 
    nodeTree["Volume"] = None
    nodeTree["Group"] = None
    ntv = nodeTree.values
    clv = compList.values
    for i in ntv: 
        for j in clv:
            if i[0] == j[0] or i[1] == j[0]: 
                i[3] = j[2]
                i[4] = j[1]
    nodeTree = pd.DataFrame(ntv, columns=["X", "Y", "dist", "Volume", "Group"])
    return nodeTree
# function to clean the nodeDist, as it is has different columns
def cleanNodeDist(nodeDist, compList):  
    a = nodeDist[nodeDist.columns[0]]
    lis = np.array(list(a.items()), dtype = object)
    A = []

    for i in lis: 
        A.append([i[0], i[1]])
    nlis = np.array(lis, dtype = object)
    X = []
    Y = []
    dist = []

    for i in range(len(nlis)): 
        x, y = nlis[i][0]
        X.append(x)
        Y.append(y)
        dist.append(nlis[i][1])

    X = np.array(X)
    Y = np.array(Y)
    dist = np.array(dist)
    nTree = pd.DataFrame({"X": X, "Y":Y, "dist":dist})
    return getVolumeAndGroup(nTree, compList)
# cleaning data for node coordinates
def cleanCoordinate(nodeLists):
    A = []
    for i in nodeLists.columns: 
        A.append(float(i))
    Blist = []
    Blist.append(A)
    # 
    for i in nodeLists.values: 
        Blist.append(list(i))
    return pd.DataFrame(np.array(Blist), columns= ["x", "y", "z"])
# adding nodes

def addNodes(skelePos, nodeCoordinates):
    roundSkele = skelePos.round(4)
    a = nodeCoordinates[['x', 'y', 'z']].values.tolist()
    b = roundSkele[['x', 'y', 'z']].values.tolist()

    roundSkele["nodeState"] = False
    roundSkele["node"] = None
    notFoundList = [] # list for coordinates that weren't found because of rounding

    for i in range(len(b)): 
        for j in range(len(a)): 
            if b[i] == a[j]: 
                roundSkele.at[i, "nodeState"] = True
                roundSkele.at[i, "node"] = j    
    return roundSkele

# get the group given the node
def getCCfromNode(compList3, node): 
    a = compList3.loc[compList3.Node == node]
    return a.Belonging_CC.values[0]

# get the node given the group
def getNodefromCC(compList3, cc): 
    a = compList3.loc[compList3.Belonging_CC == cc]
    return a.Node.values
# Find the groups 

def FindGroup(df, compList): 
    if len(df.node.values) >1:
        liz = list(filter(None, df.node.values))
        if len(liz) >= 1:
            Node = liz[0]
            n = list(compList.values[:, 0])
            cc = compList.values[:, 1]
            return cc[n.index(Node)]
        else: 
            return -1
        
# generate the dataframe with
def genBigDf(skelePos, compList, nodeLists):
    nodeCoordinates = cleanCoordinate(nodeLists)
    nodeCoordinates.to_csv("nodeCoordinates_1.csv")

    roundSkele = addNodes(skelePos, nodeCoordinates)
    roundSkele.to_csv("roundSkele_2.csv")

    maxline_id = max(roundSkele.line_id.values)


    roundSkele["cc"] = None
    test = []

    for i in range(maxline_id + 1):
        df = roundSkele.loc[roundSkele.line_id == i]
        test.append(df)

    for i in range(len(test)): 
        test[i].cc = FindGroup(test[i], compList)
    return pd.concat(test)

# find the distance between nodes 
def lengthList(bigDf1, temp): # produces a list with the longest (in length) to shortest (in length)
        maxcc = bigDf1.cc.max()
        Ls = []
        for k in range(int(maxcc)):
            test = bigDf1.loc[bigDf1.cc == k]
            linids = []
            lens = []
            endparts = []

            for i in set(test.line_id): 
                secTest = test.loc[test.line_id == i]
                b = secTest.loc[secTest.nodeState == True]
                if len(b) < 2: 
                    endparts.append(secTest)
                else: 
                    linids.append(b)
                
            for i in linids: 
                nodesInid = i.node.values
                lens.append(findDistofNodes(nodesInid[0], nodesInid[1], temp))

            for j in endparts: 
                lens.append(findLength2(j))
            
            Ls.append(np.sum(lens))

            # Ls.append([np.sum(lens), int(k)])
        # sorted_ls = sorted(Ls, key=lambda x: x[0], reverse=True)
        return Ls

def findBranching(df):
    return len(set(df.line_id.values))

def getLineIdFromCC(df, c): 
    a = df.loc[df.cc == c]
    return set(a.line_id.values)
    
def findCCfromLineId(df, linid):
    a = df.loc[df.line_id == linid]
    return a.cc.values[0] 

# getting it for the entire thing

def getFinalDat_entire(bigDf, temp): 
    maxcc = max(bigDf.cc)
    dat = []
    datfunc = []
    for i in range(int(maxcc + 1)):
        d = bigDf[bigDf.cc == i]
        dat.append([i, findLength(d, temp), findVolume2(d), findBranching(d), findNumberofNodes(d), findPixelIntensity(d)])
        # datfunc.append([findPixelIntensity(d)])
    sorted_ls = sorted(dat, key=lambda x: x[2], reverse=True)
    return pd.DataFrame(dat, columns=['cc', 'Length', 'Volume', 'Number of Branches', 'Number of Nodes', 'pixint'])

def getFinalDat_element(bigDf, temp): 
    maxcc = max(bigDf.line_id)
    dat = []
    datfunc = []
    for i in range(int(maxcc + 1)):
        d = bigDf[bigDf.line_id == i]
        dat.append([i, findLength(d, temp), findVolume2(d), findAvgWidth(d), findCCfromLineId(d, i), findPixelIntensity(d)])
       
    sorted_ls = sorted(dat, key=lambda x: x[2], reverse=True)
    return pd.DataFrame(dat, columns=['lineid', 'Length', 'Volume', 'Width', 'cc', 'pixint'])



####################################################
# This below are functions for getting the functional data
# We do this by taking the x, y z values from mitograph
# And converting them into the pixel values
# These are turned into corresponding pixel points 
# And then we get the intensity values from the pixel points
####################################################

def getPixelFromGeometrical(x_geometric, y_geometric, z_geometric): 
    pixel_size_x = 0.104
    pixel_size_y = 0.104
    pixel_size_z = 0.5

    x_pixel = int((x_geometric - 0) / pixel_size_x)
    y_pixel = int((y_geometric - 0) / pixel_size_y)
    z_pixel = int((z_geometric - 0) / pixel_size_z)
    return x_pixel, y_pixel, z_pixel

def getPixelIntensityFromPixel(arr, x, y, z):     
    # use x, y, z as the mid point of the arr, and generate a 3x3x3 array, and then get the average of that
    # if it's a an edge, then just return the average of the possible 3x3x3 array
    # write code below
    # 
    return np.mean(arr[z - 1:z + 2, x - 1:x + 2, y - 1:y + 2])


    

def addFunctional(df, image, image2):
    for i in range(len(df)): 
        x1, y2, z3 = df.loc[i, ['x', 'y', 'z']].values
        x, y, z = getPixelFromGeometrical(x1, y2, z3)

        # IMAGE 1 AND IMAGE 2 CAN BE WHATEVER YOU WANT IT TO BE 

        IMAGE1 = getPixelIntensityFromPixel(image, y, x, z)
        IMAGE2 = getPixelIntensityFromPixel(image2, y, x, z)

        df.loc[i, 'pixint1'] = getPixelIntensityFromPixel(image, y, x, z)
        df.loc[i, 'pixint2'] = getPixelIntensityFromPixel(image2, y, x, z)         

        #####################################################   
        #                                                   #
        #               CHANGING THE RATIO'S                #
        #                                                   #
        #####################################################

        if a and b != 0: 
            df.loc[i, 'pix_ratio'] = np.round(b/a, 3) # RATIO OF CHANNEL 2 (B) OVER CHANNEL 1 (A) 
        else: 
            df.loc[i, 'pix_ratio'] = None 
            
        df.loc[i, 'pposx'] = x
        df.loc[i, 'pposy'] = y
        df.loc[i, 'pposz'] = z
 
    return df
