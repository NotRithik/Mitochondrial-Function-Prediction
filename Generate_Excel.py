
# %%
import pandas as pd
import igraph as ig
import sys
import os
from Mitocode_Intensity import * 
from PIL import Image
import numpy as np
import tifffile
from MitoCode_Functions import open_file_dialog, open_folder_dialog
import matplotlib.pyplot as plt

# %%
# create  a function that takes in the filename, and generates the Table dataframe

def Mitograph_Table(FileName, graph):
    List = graph.decompose()
    Table = pd.DataFrame(columns=['nodes', 'edges', 'length_(um)', 'vol_from_img_(um3)', 'cc', 'branches'])
    temp_Table = []
    for g in List:
        branches = sum([1 for degree in g.degree() if degree == 1])  # Count the number of nodes with degree 1
        if branches == 2 and g.vcount() == 2:
            branches = 1
        degree = g.degree()
        temp_Table.append([pd.DataFrame([[g.vcount(), g.ecount(), sum(g.es['Length']), max(g.vs['cc_vol']), g.vs['cc'][0], branches, degree[0]]],
                                         columns=['nodes', 'edges', 'length_(um)', 'vol_from_img_(um3)', 'cc', 'branches', 'degree'])][0])
    Table = pd.concat(temp_Table)
    return Table


# %%
def list_gen(image, z, y, x):
    intensityList = []
    for i in range(len(x)): 
        intensityList.append(getPixelIntensityFromPixel(image, z[i], y[i], x[i]))
        # intensityList.append(getPixelIntensityFromPixel(image, x[i], y[i], z[i]))
    return intensityList

def list_gen2(image, z, y, x):
    intensityList = []
    for i in range(len(x)): 
        intensityList.append(getPixelIntensityFromPixelNoSubarray(image, z[i], y[i], x[i]))
    return intensityList

def list_gen3(image, z, y, x, width):
    intensityList = []
    for i in range(len(x)): 
        intensityList.append(getPixelIntensityFromPixelWidth(image, z[i], y[i], x[i], width[i]))
    return intensityList


# create a function that takes in the file name, the channel directories, and gives out the position dataframe
def Mitograph_Position(FileName, channel488dir, channel405dir, channelstructdir):

    # read the image
    channel488 = tifffile.imread(channel488dir)
    channel405 = tifffile.imread(channel405dir)
    channelStruct = tifffile.imread(channelstructdir)

    position = pd.read_table(FileName + '.txt', skiprows=0, sep='\t')

    # drop the pixel_intensity column if it exists


    position['x_pixel'], position['y_pixel'], position['z_pixel'] = zip(*position.apply(lambda row: getPixelFromGeometrical(row['x'], row['y'], row['z']), axis=1))

    # convert all the pixels into integers
    position['x_pixel'] = position['x_pixel'].astype(int)
    position['y_pixel'] = position['y_pixel'].astype(int)
    position['z_pixel'] = position['z_pixel'].astype(int)

    position['y_pixel'] = channel488.shape[1] - position['y_pixel']

  ################################

   # UNCOMMENT BELOW BLOCK FOR SUBARRAY. LIST_GEN = SUBARRAY
   # LIST_GEN3 = WIDTH 
   # lIST_GEN2 = NO SUBARRAY
  ############################

    # # # # # # # # add the intensity to the dataframe
    position['pixel_intensity_488'] = list_gen(channel488, position['z_pixel'], position['y_pixel'], position['x_pixel'],)
    position['pixel_intensity_405'] = list_gen(channel405, position['z_pixel'], position['y_pixel'], position['x_pixel']) 
    position['pixel_intensity_ratio'] = position['pixel_intensity_405']/position['pixel_intensity_488']
    position['pixel_intensity_555'] = list_gen(channelStruct, position['z_pixel'], position['y_pixel'], position['x_pixel'])       
    # # # # # if there is division by 0 error, then set the pixel_intensity to 0

  ################################

   # UNCOMMENT BELOW BLOCK FOR WIDTH. LIST_GEN = SUBARRAY
   # LIST_GEN3 = WIDTH 
   # lIST_GEN2 = NO SUBARRAY
  ############################
    # position['pixel_intensity_488'] = list_gen3(channel488, position['z_pixel'], position['y_pixel'], position['x_pixel'], position['width_(um)'])
    # position['pixel_intensity_405'] = list_gen3(channel405, position['z_pixel'], position['y_pixel'], position['x_pixel'], position['width_(um)'])
    # position['pixel_intensity_ratio'] = position['pixel_intensity_405']/position['pixel_intensity_488']
    # position['pixel_intensity_555'] = list_gen3(channelStruct, position['z_pixel'], position['y_pixel'], position['x_pixel'], position['width_(um)'])

  ################################

   # UNCOMMENT BELOW BLOCK FOR PIXEL. LIST_GEN = SUBARRAY
   # LIST_GEN3 = WIDTH 
   # lIST_GEN2 = NO SUBARRAY
  ############################

    # # # # # add the intensity to the dataframe
    # position['pixel_intensity_488'] = list_gen2(channel488, position['z_pixel'], position['y_pixel'], position['x_pixel'],)
    # position['pixel_intensity_405'] = list_gen2(channel405, position['z_pixel'], position['y_pixel'], position['x_pixel']) 
    # position['pixel_intensity_ratio'] = position['pixel_intensity_405']/position['pixel_intensity_488']
    # position['pixel_intensity_555'] = list_gen2(channelStruct, position['z_pixel'], position['y_pixel'], position['x_pixel'])   
    return position


# %%
# given the lsit of coords, find the distance between adjacent points

def get_distance(coords):
    distance = []
    for i in range(len(coords)-1):
        distance.append(np.sqrt((coords[i][0]-coords[i+1][0])**2 + (coords[i][1]-coords[i+1][1])**2 + (coords[i][2]-coords[i+1][2])**2))
    return sum(distance)

# %%
# create a function to generate the element_Table
def Mitograph_Element_Table(position):
    # position = Mitograph_Position(FileName, channel488dir, channel405dir, channelstructdir)
    element_Table = pd.DataFrame(columns=['line_id', 'element_length_(um)', 'element_average_width', 'element_pixel_intensity_488', 'element_pixel_intensity_405', 'element_pixel_intensity_555'])
    templist = []
    for i in range(max(position['line_id'])+1):
        cc = position.loc[position.line_id == i]
        coords = list(zip(cc['x'], cc['y'], cc['z']))
        a = pd.DataFrame([[i, get_distance(coords), cc['width_(um)'].mean(), cc['pixel_intensity_488'].mean(), cc['pixel_intensity_405'].mean(), cc['pixel_intensity_555']]], columns=['line_id', 'element_length_(um)', 'element_average_width', 'element_pixel_intensity_488', 'element_pixel_intensity_405', 'element_pixel_intensity_555'])
        # a = pd.DataFrame([[i, get_distance(coords), cc['width_(um)'].mean(), cc['pixel_intensity_488'].mean(), cc['pixel_intensity_405'].mean()]], columns=['line_id', 'element_length_(um)', 'element_average_width', 'element_pixel_intensity_488_width', 'element_pixel_intensity_405'])

        
        a['element_pixel_intensity_ratio'] = a['element_pixel_intensity_405']/a['element_pixel_intensity_488']
        # a['element_pixel_intensity_width_ratio'] = a['element_pixel_intensity_405_width']/a['element_pixel_intensity_488_width']
        templist.append(a)
    element_Table = pd.concat(templist)
    return element_Table

# %%
# given a line_id find the nearest node to that line
def find_nearest_node(line_id, position, node_position):
    # find the x,y,z coordinates of the line
    x = position.loc[position['line_id'] == line_id, 'x'].values[0]
    y = position.loc[position['line_id'] == line_id, 'y'].values[0]
    z = position.loc[position['line_id'] == line_id, 'z'].values[0]

    # find the nearest node to the line
    node_position['distance'] = ((node_position['x'] - x) ** 2 + (node_position['y'] - y) ** 2 + (
                node_position['z'] - z) ** 2) ** 0.5 
    nearest_node = node_position.loc[node_position['distance'] == min(node_position['distance']), 'node'].values[0]
    return nearest_node

# find line_id given the node
def find_line_id_node(node, node_position, position):
    x = node_position.loc[node_position['node'] == node, 'x'].values[0]
    y = node_position.loc[node_position['node'] == node, 'y'].values[0]
    z = node_position.loc[node_position['node'] == node, 'z'].values[0]
    position['distance'] = ((position['x'] - x) ** 2 + (position['y'] - y) ** 2 + (position['z'] - z) ** 2) ** 0.5
    line_id = position.loc[position['distance'] == min(position['distance']), 'line_id'].values
    point_id = position.loc[position['distance'] == min(position['distance']), 'point_id'].values
    return line_id

def find_point_id_node(node, node_position, position): 
    x = node_position.loc[node_position['node'] == node, 'x'].values[0]
    y = node_position.loc[node_position['node'] == node, 'y'].values[0]
    z = node_position.loc[node_position['node'] == node, 'z'].values[0]
    position['distance'] = ((position['x'] - x) ** 2 + (position['y'] - y) ** 2 + (position['z'] - z) ** 2) ** 0.5
    point_id = position.loc[position['distance'] == min(position['distance']), 'point_id'].values
    return point_id

# return a list of the line id's in a cc



# %%
# write a function to get the average degree of an element in the graph


# %%
# get all the line_id in a cc and return as a list 
def get_line_id_cc_2(cc, node_position):
    line_id = node_position.loc[node_position['cc'] == cc, 'line_id'].values
    # make line_ id one list 
    line_id = [item for sublist in line_id for item in sublist]
    # make it unique
    line_id = list(set(line_id))
    return line_id

# given a list of line_id's get the average pixel intensity for those line_id's using position dataframe
def get_pixel_intensity_cc(line_id, position, choice):
    if choice == '405':
        return  position.loc[position['line_id'].isin(line_id), 'pixel_intensity_405'].mean()
    elif choice == '555': 
        return  position.loc[position['line_id'].isin(line_id), 'pixel_intensity_555'].mean()
    else:
        return  position.loc[position['line_id'].isin(line_id), 'pixel_intensity_488'].mean()


# using the above two functions, get the pixel intensity for each cc and add it to the merged_Table


# %%
# function to generate merged_Table
   
def calculate_coeff_of_variance(df, position):
    # calculate the coefficient of variance for each cc_x    

    df['coeff_of_variance'] = position['pixel_intensity_ratio'].std(ddof=0)/df['pixel_intensity_ratio'].mean() * 100
    return df


def Mitograph_Merged_Table(FileName, Vol, position, graph, channel488dir, channel405dir, channelstructdir):
    Table = Mitograph_Table(FileName, graph)
    node_position = pd.read_table(FileName + '.coo', skiprows=0, sep='\t', names=['x', 'y', 'z'])
    node_position['node'] = [i for i in range(0, len(node_position))]

    # merge node_position with Vol on node
    node_position = pd.merge(node_position, Vol, on='node')
    # add line_id to node_position
    node_position['line_id'] = [find_line_id_node(node, node_position, position) for node in node_position['node']]
    # merge Table and node_position on cc

    merged_Table = pd.merge(Table, node_position, on='cc')
    # this is all at a cc level
    merged_Table['pixel_intensity_405'] =  [get_pixel_intensity_cc(get_line_id_cc_2(cc, node_position), position, choice = '405') for cc in merged_Table['cc']]
    merged_Table['pixel_intensity_488'] = [get_pixel_intensity_cc(get_line_id_cc_2(cc, node_position), position, choice = '488') for cc in merged_Table['cc']]
    merged_Table['pixel_intensity_555'] = [get_pixel_intensity_cc(get_line_id_cc_2(cc, node_position), position, choice = '555') for cc in merged_Table['cc']]

    merged_Table['pixel_intensity_ratio'] = merged_Table['pixel_intensity_405'] / merged_Table['pixel_intensity_488']
    return merged_Table

# %%
# create a function of the above that takes in graph and cc
def get_degree_distribution(graph, cc):
    graph_cc = graph.subgraph(graph.vs.select(cc_eq=cc))
    degree = graph_cc.degree()
    degree = pd.DataFrame(degree, columns=['degree'])
    # do the value_counts but with the index as the degree
    degree = degree['degree'].value_counts().reset_index()
    degree.columns = ['degree', 'count']
    # convert this into a list so that it's like 1: 6, 3: 2, 4: 1
    degree = degree.values.tolist()
    degree = dict(degree)

    # returns it in the format of {degree: count}
    return degree
# find the name of the nodes in graph_cc1
# create a function that finds the line_ids in a cc ant takes in graph and cc

# given graph and cc return the line_id's in that cc
def get_line_id_cc(graph, cc, Vol, node_position, position):
    graph_cc = graph.subgraph(graph.vs.select(cc_eq=cc))
    nodes = [graph_cc.vs[i]['name'] for i in range(len(graph_cc.vs))]
    # for each node find the line_id
    line_id = [find_line_id_node(node, node_position, position) for node in nodes]
    # make line_ id one list
    line_id = [item for sublist in line_id for item in sublist]
    # make it unique
    line_id = list(set(line_id))
    return line_id


# %%
# given a degree distribution, find the average degree but exclude the 1 degree nodes

def get_average_degree(degree_distribution):
    total_degree = 0
    total_nodes = 0
    for k,v in degree_distribution.items():
        total_nodes += v
        if k != 1:
            total_degree += k*v
    if total_nodes == 0:
        return 0
    else:
        return total_degree/total_nodes

def get_largest_sum(degree_distribution):
    total_degree = 0
    total_nodes = 0
    total_count = 0
    total_max = 0
    for k,v in degree_distribution.items():
            if k > total_max:
                total_max = k
                total_count = v
    
    if total_max == 0: 
        return 0 
    else: 
        return (total_max*total_count)

# %%
# write a function to generate cc_Table
def Mitograph_cc_Table(FileName, Vol, position, node_position, channel488dir, channel405dir, channelstructdir, graph):
    merged_Table = Mitograph_Merged_Table(FileName, Vol, position, graph, channel488dir, channel405dir, channelstructdir)
    cc_Table = merged_Table[['cc', 'length_(um)', 'vol_from_img_(um3)', 'branches', 'nodes', 'edges', 'pixel_intensity_ratio', 'pixel_intensity_488', 'pixel_intensity_405', 'pixel_intensity_555']]
    cc_Table = cc_Table.drop_duplicates(subset=['cc'])
    # order it in ascendign by cc
    # for each cc get the degree distribution
    dd_temp = [get_degree_distribution(graph, cc) for cc in cc_Table['cc']]
    cc_Table['degree_distribution'] = dd_temp
    cc_Table['cc_average_degree_excludeFreeEnds'] = [get_average_degree(x) for x in dd_temp]
    cc_Table['cc_max_PK'] = [get_largest_sum(x) for x in dd_temp]
    cc_Table['cc_average_connectivity'] = 1/cc_Table['cc_average_degree_excludeFreeEnds']
    cc_Table['line_id'] = [get_line_id_cc(graph, cc, Vol, node_position, position) for cc in cc_Table['cc']]
    cc_Table['diameter'] = 2*np.sqrt(cc_Table['vol_from_img_(um3)']/(np.pi*cc_Table['length_(um)']))
    cc_Table = cc_Table.sort_values(by=['cc'])
    # rename columns to be more informative
    cc_Table = cc_Table.rename(columns={'cc': 'cc', 'cc_max_PK': 'cc_max_PK', 'length_(um)': 'cc_length_(um)', 'vol_from_img_(um3)': 'cc_vol_from_img_(um3)', 'branches': 'branches', 'pixel_intensity_ratio': 'cc_pixel_intensity_ratio', 'pixel_intensity_488':'cc_pixel_intensity_488', 'pixel_intensity_405':'cc_pixel_intensity_405', 'pixel_intensity_555': 'cc_pixel_intensity_555', 'nodes': 'nodes', 'edges': 'edges', 'line_id': 'line_id'})
    cc_Table.reset_index(drop=True, inplace=True)
    return cc_Table

# %%
# FileName = '/Users/birat/Onedrive/Desktop/New folder/AnalysisFiles_Tol4hr5_2_S/structural/Tol4hr5_2_S_tif3'
# channel488dir = '/Users/birat/OneDrive/Desktop/New folder/AnalysisFiles_Tol4hr5_2_S/functional488/Tol4hr5_2_S_functional488.tif'
# channel405dir = '/Users/birat/OneDrive/Desktop/New folder/AnalysisFiles_Tol4hr5_2_S/functional405/Tol4hr5_2_S_functional405.tif'
# channelstructdir = '/Users/birat/OneDrive/Desktop/New folder/AnalysisFiles_Tol4hr5_2_S/structural/Tol4hr5_2_S_tif3.tif'
# # read the image
# channel488 = tifffile.imread(channel488dir)
# channel405 = tifffile.imread(channel405dir)
# channelStruct = tifffile.imread(channelstructdir)


def Mitograph_Graph(FileName):
    # Read information from the gnet file, and create a graph object
    # generate a function that takes in the filename and gives out the graph
    G = pd.read_table(FileName + '.gnet', skiprows=1, sep='\t', names=['Source', 'Target', 'Length'])
    graph = ig.Graph.TupleList(G.itertuples(index=False), directed=False, edge_attrs=['Length'])
    layout = graph.layout_auto()  # Automatic layout calculation
    graph.vs['degree'] = graph.degree()
    # ig.plot(graph, layout=layout)
    Vol = pd.read_table(FileName + '.cc', skiprows=0, sep='\t')
    Vol.columns = ['node', 'cc', 'vol_cc']

    ids = [int(node['name']) for node in graph.vs]
    graph.vs['cc_vol'] = [Vol.loc[Vol['node'] == node, 'vol_cc'].values[0] for node in ids]
    graph.vs['cc'] = [Vol.loc[Vol['node'] == node, 'cc'].values[0] for node in ids]
    return graph, Vol
# graph, Vol= Mitograph_Graph(FileName)
# given a cc_Table, find the fusion and fission events
# fusion 5 is the k longest cc, divided by the total length of the cc, time 100
def fusion(cc_Table, k):
    # get the k longest elements
    # sort by length_(um), and use head to get the top k cc
    k_longest = cc_Table.sort_values(by=['cc_length_(um)'], ascending=False).head(k)
    # get the total length of the k longest cc
    total_length = sum(k_longest['cc_length_(um)'])
    # get the total length of all the cc
    total_length_all = sum(cc_Table['cc_length_(um)'])
    # get the fusion
    fusion = (total_length/total_length_all)*100

    return fusion

def fission(cc_Table):
    return len(cc_Table)/cc_Table['cc_length_(um)'].sum()


# %%
def get_degree_distribution_node(graph, cc):
    graph_cc = graph.subgraph(graph.vs.select(cc_eq=cc))
    degree = graph_cc.degree()
    degree = pd.DataFrame(degree, columns=['degree'])
    # do the value_counts but with the index as the degree
    degree = degree['degree'].value_counts().reset_index()
    degree.columns = ['degree', 'count']
    # convert this into a list so that it's like 1: 6, 3: 2, 4: 1
    degree = degree.values.tolist()
    degree = dict(degree)
    # returns it in the format of {degree: count}
    return degree

def add_avg_PK_line_id(nodedf): 
    # get the average degree for each line_id
    avg_degree = nodedf.groupby('line_id')['degree'].mean().reset_index()
    # rename degree to avg_degree
    avg_degree.rename(columns={'degree': 'avg_PK_Of_element'}, inplace=True)
    # merge avg_degree with nodedf on line_id
    nodedf = pd.merge(nodedf, avg_degree, on='line_id', how='left')
    nodedf['element_connectivity'] = 1/nodedf['avg_PK_Of_element']
    return nodedf

def Mitograph_node_position(FileName, Vol, position, graph):
    node_position = pd.read_table(FileName + '.coo', skiprows=0, sep='\t', names=['x', 'y', 'z'])
    node_position['node'] = [i for i in range(0, len(node_position))]
    
    # merge node_position with Vol on node
    node_position['line_id'] = [find_line_id_node(i, node_position, position) for i in node_position['node']]
    node_position['point_id'] = [find_point_id_node(i, node_position, position) for i in node_position['node']]
    
    # make node_position['point_id'] not a list
    node_position['point_id'] = [i[0] for i in node_position['point_id']]
    # the below is actually the PK
    node_position['degree'] = [graph.vs.select(name = int(i)).degree()[0] for i in node_position['node']]
    
    node_position = pd.merge(node_position, Vol, on='node')
    node_position = node_position.explode('line_id').reset_index(drop=True)
    
    # Get the average degree of a line_id
    node_position = add_avg_PK_line_id(node_position)
    return node_position

def calculate_coeff_of_variance_cc(df):
    unique_cc_x_values = df['cc_x'].unique()
    
    for cc_x in unique_cc_x_values:
        # Get the subset of the dataframe for the current cc_x
        subset = df[df['cc_x'] == cc_x]
        
        # Calculate the coefficient of variance for the current cc_x
        cv = subset['element_pixel_intensity_ratio'].std(ddof=0) / subset['element_pixel_intensity_ratio'].mean() * 100
        
        # Assign the calculated CV to all rows with the current cc_x
        df.loc[df['cc_x'] == cc_x, 'coeff_of_variance_cc'] = cv
    
    return df

def calculate_coeff_of_variance_line_id(df):
    unique_line_id_values = df['line_id'].unique()
    
    for line_id in unique_line_id_values:
        # Get the subset of the dataframe for the current cc_x
        subset = df[df['line_id'] == line_id]
        
        # Calculate the coefficient of variance for the current cc_x
        cv = subset['element_pixel_intensity_ratio'].std(ddof=0) / subset['element_pixel_intensity_ratio'].mean() * 100
        
        # Assign the calculated CV to all rows with the current cc_x
        df.loc[df['line_id'] == line_id, 'coeff_of_variance_line_id'] = cv
    
    return df

# Assuming your dataframe is called df and has columns 'cc_x' and 'pixel_intensity_ratio'
# Replace these with your actual column names

# Apply the function to calculate coefficient of variance

def Generate_Mito_Tables(dir):
    if dir == '':
        main_folder_dir = open_folder_dialog('Select Folder with AnalysisFiles')
    else:
        main_folder_dir = dir
    folders_directory = [os.path.join(main_folder_dir, i) for i in os.listdir(main_folder_dir)]
    folders_directory = [i for i in folders_directory if os.path.isdir(i)]

    folders = os.listdir(main_folder_dir)
    # remove non folders from this list
    folders = [i for i in folders if "." not in i]


    cc_pdList = []
    pos_pdList = []
    node_pdList = []
    ele_pdList =[]
    complete_df = []
    alllist = []
    folder_dir = folders_directory
    print("{} folders found. Now running...".format(len(folder_dir)))
    for k in range(len(folder_dir)):

        #
        # GETTING THE FOLDER DIRECTORIES 
        # THIS IS BASED ON HOW THE FOLDERS ARE NAMED

        # folder_dir = folders_directory[k] 
        current_folder = folder_dir[k]+ "/structural"
        # look inside current_folder and find the file which is a .tif file

        current_file = [i for i in os.listdir(current_folder) if ".tif" in i][0]
        current_file = os.path.join(current_folder, current_file)[: -4]

        folder_name = (current_folder.split("/")[-2]).split("AnalysisFiles_")[1]

        # channel488 = open_file_dialog('Select 488 channel')[0]
        # make channel488 
        channel488_folder = folders_directory[k] + "/functional488/"
        channel488dir = os.path.join(channel488_folder, os.listdir(channel488_folder)[0])

        # repeat for channel405
        channel405_folder = folders_directory[k] + "/functional405/"
        channel405dir= os.path.join(channel405_folder, os.listdir(channel405_folder)[0])
        
        # repeat for channelstruct
        channelstruct_folder = folders_directory[k] + "/structural/"

        x = [i for i in os.listdir(channelstruct_folder) if ".tif" in i]

        channelstructdir= os.path.join(channelstruct_folder, x[0])

        # 
        # GENERATING THE TABLES
        #
        graph, Vol = Mitograph_Graph(current_file)
        position = Mitograph_Position(current_file, channel488dir, channel405dir, channelstructdir)

        node_pos = Mitograph_node_position(current_file, Vol, position, graph)
        node_pos = node_pos.explode('line_id').reset_index(drop=True)

        mitoTable = Mitograph_cc_Table(current_file, Vol, position, node_pos, channel488dir, channel405dir, channelstructdir, graph)
        element = Mitograph_Element_Table(position)

        element['folder_name'] = folder_name
        position['folder_name'] = folder_name
        node_pos['folder_name'] = folder_name
        mitoTable['folder_name'] = folder_name
        element['element_Volume_Voxel'] = element['element_length_(um)'] * element['element_average_width']

        cc_pdList.append(mitoTable)
        pos_pdList.append(position)
        ele_pdList.append(element)
        node_pdList.append(node_pos)

        name = ''
        # name = current_file.split('\\')[-1].split('_t')[0]
        # position = Mitograph_Position(current_file, channel488dir, channel405dir, channelstructdir)
        channelstru = tifffile.imread(channelstructdir)

        # # temp df is the dataframe with the line_id > 0 and z == 3
        # plotting the overlay
        tempdf = position
        xpos = tempdf['x_pixel'].values
        ypos = tempdf['y_pixel'].values
        zpos = tempdf['z_pixel'].values

        # creating the skeleton we see
        blank = np.zeros_like(channelstru)
        for i in range(len(xpos)):
            blank[zpos[i]][ypos[i], xpos[i]] = 1

        zprojblank = np.max(blank, axis=0)
        zprojblank = np.max(zprojblank) - zprojblank
        zproject = np.max(channelstru, axis=0)

        myarray = zprojblank.max() - zprojblank * 10000
        myarray = np.where(myarray == 1, 255, 0)
        my_array = myarray.astype(np.uint8) 
        # Create an image from the array
        image = Image.fromarray(my_array)
        image.save(os.path.join(current_folder, 'zproject.png'))
        image = Image.fromarray(my_array)


        a = pd.read_table(current_file + '.mitograph', skiprows=6, sep = '\t', header=None, names = ['vol_from_voxels_(um)',	'avg_width_(um)',	'std_width_(um)',	'total_length_(um)',	'vol_from_length_(um3)',	'nodes',	'edges',	'components'])
        a = a.iloc[0]
        a['Mean Channel 2'] = position.pixel_intensity_488.mean()
        a['Mean Channel 1'] = position.pixel_intensity_405.mean()
        a['Mean Channel Ratio'] = position.pixel_intensity_ratio.mean()

        a['Smallest Element Ratio'] = element.element_pixel_intensity_ratio.min()
        a['Largest Element Ratio'] = element.element_pixel_intensity_ratio.max()

        a.to_frame()
        fusion_a = pd.read_table(current_file + '.mitograph', skiprows=11, sep = '\t', header=None, names = ['nodes',	'edges',	'cc_length_(um)',	'cc_vol_from_img_(um3)'])
        a['fusion 5'] = fusion(fusion_a, 5)
        a['fission'] = fission(fusion_a)
        f1 = fusion(fusion_a, 1)
        a['fusion 1'] = f1
        a['folder_name'] = folder_name
        long_cc_id = mitoTable.sort_values(by=['cc_length_(um)'], ascending=False)['cc'].head(1).values[0]
        long_cc_length = mitoTable.sort_values(by=['cc_length_(um)'], ascending=False)['cc_length_(um)'].head(1).values[0]
        short_cc_id = mitoTable.sort_values(by=['cc_length_(um)'], ascending=True)['cc'].head(1).values[0]
        short_cc_length = mitoTable.sort_values(by=['cc_length_(um)'], ascending=True)['cc_length_(um)'].head(1).values[0]
        
        a['Longest CC (id / μm)'] = str(long_cc_id) +  ' / ' + str( np.round(long_cc_length, 3))
        a['Shortest CC (id / μm)'] = str(short_cc_id) +  ' / ' + str( np.round(short_cc_length, 3))
        bigA = a
        a = bigA[['folder_name', 'vol_from_voxels_(um)',	'avg_width_(um)',	'std_width_(um)',	'total_length_(um)',	'vol_from_length_(um3)',	'nodes',	'edges',	'components', 'fusion 1', 'fusion 5', 'fission', 'Longest CC (id / μm)', 'Shortest CC (id / μm)']] 
        funca = bigA[['Mean Channel 1', 'Mean Channel 2', 'Mean Channel Ratio', 'Smallest Element Ratio', 'Largest Element Ratio']]
        # for all columns in a, change the (um) to μm 
        # a is a series

        a.index = [i.replace('um', 'μm') for i in a.index]
        # capitalize all the column names
        
        mitotabletemp = mitoTable[mitoTable['edges'] > 1]
        a['number_of_networks'] = len(mitotabletemp['cc'].unique())
        a['Number of Standalone Mitochondria'] = a['components'] - a['number_of_networks']
        
        # capitalize the first letter


        a.index = [i.capitalize() for i in a.index]

        #
        # READING THE MTIOGRAPH FILE TO GET THE INDIVIDUAL COMPONENTS
        # 

        position = position.replace([np.inf, -np.inf], np.nan)

        # create a folder to save the csv files
        if not os.path.exists(current_file.split("/structural")[0] + "/Mitograph_Tables"):
            os.makedirs(current_file.split("/structural")[0] + "/Mitograph_Tables")
 
        savedir = current_file.split("/structural")[0] + "/Mitograph_Tables"

        cc_pdList[k].to_csv(savedir + "/cc_Table" + name + ".csv")
        pos_pdList[k].to_csv(savedir + "/point_Table" + name + ".csv")
        ele_pdList[k].to_csv(savedir + "/element_Table" + name + ".csv")
        node_pdList[k].to_csv(savedir + "/node_Table" + name + ".csv")
        # complete_df[k].to_csv(savedir + "/complete_Table" + name + ".csv")

        # make folder_name the first column

        alllist.append(a.to_frame().T)
        a.to_csv(savedir + "/summary_table" + name + ".csv")
        funca.to_csv(savedir + "/summary_functional_Table" + name + ".csv")

        pospd2 = pos_pdList[k]
        ccpd2 = cc_pdList[k]
        elePd2 = ele_pdList[k]
        nodepd2 = node_pdList[k]
        # te2 = pd.concat(alllist[k])

        ccpd_exploded2 = []

        # for i in range(len(pospd2)):
        #     # drop distance from pospd[i]
        #     # if there is a distance column in pospd[i]
        #     # if 'distance' in pospd2[i].columns:
        #     #         pospd2[i].drop('distance', axis=1, inplace=True)
        m2 = elePd2.merge(pospd2, on='line_id', how='left')
        df_expanded2 = ccpd2.explode('line_id').reset_index(drop=True)
        # merge df_expanded and m on line_id 
        df_expanded_temp2 = pd.merge(df_expanded2, m2, on='line_id', how='left')
        # drop the folder name when merging
        df_expanded_temp2.drop('folder_name_y', axis=1, inplace=True)
        # rename folder_name_x to folder_name
        df_expanded_temp2.rename(columns={'folder_name_x': 'folder_name'}, inplace=True)

        node_merge2 = df_expanded_temp2.merge(nodepd2, on=['line_id', 'point_id'], how='left')
        # re order line_id to come after cc_average_degree_excludeFreeEnds

        ccpd_exploded2.append(node_merge2)


        fullLit2 = pd.concat(ccpd_exploded2)
        fullLit2 = calculate_coeff_of_variance_cc(fullLit2)
        fullLit2 = calculate_coeff_of_variance_line_id(fullLit2)
        # fullLit = calculate_coeff_of_variance(fullLit)
        fullLit2.to_csv(folder_dir[k] + "/full_Table_" + os.path.dirname(main_folder_dir[0]).split('/')[-1] + folder_name + ".csv")

            #'/home/mitosim2/BIRATAL/Tcdd1s_subsetted'


    pospd = pos_pdList
    ccpd = cc_pdList
    elePd = ele_pdList
    nodepd = node_pdList
    # te = pd.concat(alllist)

    ccpd_exploded = []

    for i in range(len(pospd)):
        # drop distance from pospd[i]
        # if there is a distance column in pospd[i]
        if 'distance' in pospd[i].columns:
                pospd[i].drop('distance', axis=1, inplace=True)
        m = elePd[i].merge(pospd[i], on='line_id', how='left')
        df_expanded = ccpd[i].explode('line_id').reset_index(drop=True)
        # merge df_expanded and m on line_id 
        df_expanded_temp = pd.merge(df_expanded, m, on='line_id', how='left')
        # drop the folder name when merging
        df_expanded_temp.drop('folder_name_y', axis=1, inplace=True)
        # rename folder_name_x to folder_name
        df_expanded_temp.rename(columns={'folder_name_x': 'folder_name'}, inplace=True)

        node_merge = df_expanded_temp.merge(nodepd[i], on=['line_id', 'point_id'], how='left')
        # re order line_id to come after cc_average_degree_excludeFreeEnds
        ccpd_exploded.append(node_merge)

 

    # add the coeff of variance
    fullLit = pd.concat(ccpd_exploded)
    fullLit = calculate_coeff_of_variance_cc(fullLit)
    fullLit = calculate_coeff_of_variance_line_id(fullLit)
    # fullLit = calculate_coeff_of_variance(fullLit)
    fullLit.to_csv(main_folder_dir + "/full_Table_" + os.path.dirname(main_folder_dir[0]).split('/')[-1] + ".csv")
    print("csv File generated")
    fullNode = pd.concat(nodepd)
    fullNode.to_csv(main_folder_dir + "/full_node_Table_" + os.path.dirname(main_folder_dir[0]).split('/')[-1] + ".csv")
    return alllist, pos_pdList, cc_pdList, node_pdList, fullLit

def main():
    print("Welcome to the Excel File Generator! This will generate files for all the folders in the selected folder")
    print("Please select the folder that contains all the folders. This would be the folder which contains the folders with the name AnalysisFiles_")
    try: 
        if sys.argv[2]:
            dir = sys.argv[2]
            print(dir)
        else: 
            dir = ''
    except: 
        IndexError
        dir = ''


    a, p, cc, node, full = Generate_Mito_Tables(dir)

# main()
