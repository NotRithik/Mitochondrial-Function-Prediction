# Generating the Functional Information: 
# What is required, the x, y, z coordinates, and the channels. 
# Ideally the coordinates are in the form of a pandas data base. 
########################
import numpy as np
import pandas as pd

def getPixelFromGeometrical(x_geometric, y_geometric, z_geometric): 
    pixel_size_x = 0.104
    pixel_size_y = 0.104
    pixel_size_z = 0.5

    x_pixel = round((x_geometric ) / pixel_size_x)
    y_pixel = round((y_geometric ) / pixel_size_y)
    z_pixel = round((z_geometric ) / pixel_size_z)
    return x_pixel, y_pixel, z_pixel


# generate subarray of the image, of size 3 x 3 x 3 if the midpoint is not on the edge
# if the midpoint is on the edge, then the subarray will be smaller

def getPixelIntensityFromPixelWidth(image, z, y, x, width): 
    # get width in terms of pixels 
    # Mitograph returns width as diameter, so we need to divide by 2 to get radius
    width_pixel = int((width/2) // 0.104)
    subarray = image[z, y - width_pixel:y + width_pixel, x - width_pixel:x + width_pixel]
    return subarray.mean()
    # return a voxel of the subarray which is of that 


def getPixelIntensityFromPixel(image, z, y, x):
    # get the dimensions of the image

    z_dim = image.shape[0]
    y_dim = image.shape[1]
    x_dim = image.shape[2]

    
    # get the subarray
    if x == 0: 
        x_start = 0
        x_end = 1
    elif x == x_dim - 1: 
        x_start = x_dim - 2
        x_end = x_dim - 1
    else: 
        x_start = x - 1
        x_end = x + 1
    
    if y == 0: 
        y_start = 0
        y_end = 1
    elif y == y_dim - 1: 
        y_start = y_dim - 2
        y_end = y_dim - 1
    else: 
        y_start = y - 1
        y_end = y + 1
    
    if z == 0: 
        z_start = 0
        z_end = 1
    elif z == z_dim - 1: 
        z_start = z_dim - 2
        z_end = z_dim - 1
    else: 
        z_start = z - 1
        z_end = z + 1
    
    subarray = image[z, y_start:y_end, x_start:x_end]

    if subarray.mean() == None: 
        print('none')
        return image[z][y][x]
    return subarray.mean()

# create a similar function but no subarray, just returns the point
def getPixelIntensityFromPixelNoSubarray(image, z, y, x):
    return image[z][y][x]

    # rewrite getPixelFromGeometrical 
def addIntensityToDataFrame(dataframe, channel, channelName):
    # get the x y z coordinates
    x_geometric = dataframe['x'].to_numpy()
    y_geometric = dataframe['y'].to_numpy()
    z_geometric = dataframe['z'].to_numpy()
    
    # convert the x y z coordinates to pixels
    x_pixel, y_pixel, z_pixel = getPixelFromGeometrical(x_geometric, y_geometric, z_geometric)
    
    # get the intensity of the pixel
    intensity = []
    secIntensity = []
    for i in range(len(x_pixel)): 
        subarray = getPixelIntensityFromPixel(channel, int(z_pixel[i]),int(y_pixel[i]), int(x_pixel[i]))
        nosubarray = getPixelIntensityFromPixelNoSubarray(channel, int(z_pixel[i]),int(y_pixel[i]), int(x_pixel[i]))
        secIntensity.append(nosubarray)
        intensity.append(np.mean(subarray))
        # add the pixels as well of the subarray
    dataframe[ 'pixel x'] = x_pixel
    dataframe[ 'pixel y'] = y_pixel
    dataframe[ 'pixel z'] = z_pixel
    
    # add the intensity to the dataframe
    
    dataframe[channelName + ' (subarray)'] = intensity
    dataframe[channelName + ' (nosubarray)'] = secIntensity
    return dataframe
