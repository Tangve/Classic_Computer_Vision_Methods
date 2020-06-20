'''
  File name: click_correspondences.py
  Author: 
  Date created: 
'''

'''
  File clarification:
    Click correspondences between two images
    - Input im1: target image
    - Input im2: source image
    - Output im1_pts: correspondences coordiantes in the target image
    - Output im2_pts: correspondences coordiantes in the source image
'''

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class cpselect_recorder:
	def __init__(self, img1,img2):

		fig, (self.Ax0, self.Ax1) = plt.subplots(1, 2, figsize = (20, 20))

		self.Ax0.imshow(img1)
		self.Ax0.axis('off')

		self.Ax1.imshow(img2)
		self.Ax1.axis('off')

		fig.canvas.mpl_connect('button_press_event', self)
		self.left_x = []
		self.left_y = []
		self.right_x = []
		self.right_y = []

	def __call__(self, event):
		circle = plt.Circle((event.xdata, event.ydata),color='r')
		if event.inaxes == self.Ax0:
			self.left_x.append(event.xdata)
			self.left_y.append(event.ydata)
			self.Ax0.add_artist(circle)
			plt.show()
		elif event.inaxes == self.Ax1:
			self.right_x.append(event.xdata)
			self.right_y.append(event.ydata)
			self.Ax1.add_artist(circle)	
			plt.show()	

def click_correspondences(img1,img2):
    resize_img1=np.array(Image.fromarray(img1).resize([300, 300]))
    resize_img2=np.array(Image.fromarray(img2).resize([300, 300]))
    point = cpselect_recorder(resize_img1,resize_img2)
    plt.show()
    point_left = np.concatenate([(np.array(point.left_x)*img1.shape[1]*1.0/300)[...,np.newaxis],\
								(np.array(point.left_y)*img1.shape[0]*1.0/300)[...,np.newaxis]],axis = 1)
    point_right = np.concatenate([(np.array(point.right_x)*img2.shape[1]*1.0/300)[...,np.newaxis],\
								(np.array(point.right_y)*img2.shape[0]*1.0/300)[...,np.newaxis]],axis = 1)
    plt.scatter(point_left[:,0], point_left[:,1])
    plt.imshow(img1)
    plt.show()
    plt.scatter(point_right[:,0], point_right[:,1])
    plt.imshow(img2)
    plt.show()
    print(point_left)
    print(point_right)
    point_left.tofile('im1')
    point_right.tofile('im2')
    return point_left, point_right
    '''
    Tips:
      - use 'matplotlib.pyplot.subplot' to create a figure that shows the source and target image together
      - add arguments in the 'imshow' function for better image view
      - use function 'ginput' and click correspondences in two images in turn
      - please check the 'ginput' function documentation carefully
        + determine the number of correspondences by yourself which is the argument of 'ginput' function
        + when using ginput, left click represents selection, right click represents removing the last click
        + click points in two images in turn and once you finish it, the function is supposed to 
          return a NumPy array contains correspondences position in two images
  '''
  

  # TODO: Your code here

if __name__ == "__main__":
    sourceImg=plt.imread('source.jpg')
    targetImg=plt.imread('target.jpg')
    sourceImg=np.array(Image.fromarray(sourceImg).resize([300, 300]))
    targetImg=np.array(Image.fromarray(targetImg).resize([300, 300]))
    point_source,point_target=click_correspondences(sourceImg,targetImg)
    '''
    warp_frac=np.linspace(0,1,100)
    dissolve_frac=np.linspace(0,1,100)
    morphed_im=morph_tri(sourceImg,targetImg,point_source,point_target, warp_frac, dissolve_frac)
    '''
