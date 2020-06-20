'''
  File name: anms.py
  Author:
  Date created:
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed 
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''
import numpy as np

def anms2(cimg, max_pts):
  threshold = np.amax(cimg) * 0.1
  cimg_filter = cimg.copy()
  cimg_filter[cimg > threshold] = 1
  cimg_filter[cimg <= threshold] = 0

  H = cimg_filter.shape[0]
  W = cimg_filter.shape[1]
  print(H)
  print(W)
  x,y = np.meshgrid(np.arange(W), np.arange(H))
  x_list = x[cimg_filter == 1]
  y_list = y[cimg_filter == 1]
  mag_list = cimg[cimg_filter == 1]
  # resize binary array
  size = np.count_nonzero(cimg_filter == 1)
  x_list = x_list.reshape(size, 1)
  y_list = y_list.reshape(size, 1)
  mag_list = mag_list.reshape(size, 1)

  # size by size
  distance_list = np.sqrt((x_list - np.transpose(x_list))**2 + (y_list - np.transpose(y_list))**2)
  binary_map = mag_list < 0.9* np.transpose(mag_list)
  
  radius_list = binary_map * distance_list
  radius_list[radius_list == 0] = np.inf
  radius_list = np.amin(radius_list, axis=1)  # find min for each row
  index_list = np.argsort(radius_list)
  index_list_sorted = index_list[::-1]

  x = x_list[index_list_sorted][:max_pts]
  y = y_list[index_list_sorted][:max_pts]
  rmax = radius_list[index_list_sorted][max_pts-1]
  
  # convert a list of lists to a flat list
  x = [item for elem in x for item in elem]
  y = [item for elem in y for item in elem]
  return x, y, rmax


    
    