'''
  File name: morph_tri.py
  Author:
  Date created:
'''

'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: target image
    - Input im2: source image
    - Input im1_pts: correspondences coordiantes in the target image
    - Input im2_pts: correspondences coordiantes in the source image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''

from scipy.spatial import Delaunay
from interp import interp2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import imageio

def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
    '''
    h1,w1=im1[:,:,0].shape
    h2,w2=im2[:,:,0].shape
    '''
    h1,w1=300,300
    h2,w2=300,300
    morphed_im=np.zeros([60,max(h1,h2),max(w1,w2),3])
    for i in range(len(warp_frac)):
        print('The %i picture' %(i+1))
        ave=im1_pts*(1-warp_frac[i])+im2_pts*warp_frac[i]
        '''
        h=h1*(1-warp_frac[i])+h2*warp_frac[i]
        w=w1*(1-warp_frac[i])+w2*warp_frac[i]
        h=int(h)
        w=int(w)
        '''
        h=300
        w=300
        Tri = Delaunay(ave)
        tri_index=Tri.simplices
        #find point
        pic=np.ones([h,w,3])
        X1=np.ones([300,300])
        Y1=np.ones([300,300])
        X2=np.ones([300,300])
        Y2=np.ones([300,300])
        for j in range(h):
            for k in range(w):
                triinx=Tri.find_simplex(np.array([k,j]))
                pointinx=tri_index[triinx]
                A=ave[pointinx[0]]
                B=ave[pointinx[1]]
                C=ave[pointinx[2]]
                b=np.array([[k],[j],[1]])
                matrixA=np.array([[A[0],B[0],C[0]],[A[1],B[1],C[1]],[1,1,1]])
                para=np.linalg.solve(matrixA,b)
                x1=para[0][0]*im1_pts[pointinx[0]]+para[1][0]*im1_pts[pointinx[1]]+para[2][0]*im1_pts[pointinx[2]]
                x2=para[0][0]*im2_pts[pointinx[0]]+para[1][0]*im2_pts[pointinx[1]]+para[2][0]*im2_pts[pointinx[2]]
                X1[j][k]=x1[0]
                Y1[j][k]=x1[1]
                X2[j][k]=x2[0]
                Y2[j][k]=x2[1]
        R1=interp2(im1[:,:,0],X1,Y1)
        R2=interp2(im2[:,:,0],X2,Y2)
        G1=interp2(im1[:,:,1],X1,Y1)
        G2=interp2(im2[:,:,1],X2,Y2)
        B1=interp2(im1[:,:,2],X1,Y1)
        B2=interp2(im2[:,:,2],X2,Y2)
        R=R1*(1-dissolve_frac[i])+R2*dissolve_frac[i]
        G=G1*(1-dissolve_frac[i])+G2*dissolve_frac[i]
        B=B1*(1-dissolve_frac[i])+B2*dissolve_frac[i]
        pic[:,:,0]=np.copy(R)
        pic[:,:,1]=np.copy(G)
        pic[:,:,2]=np.copy(B)
        morphed_im[i]=pic
    return morphed_im

if __name__ == "__main__":
    sourceImg=plt.imread('source.jpg')
    targetImg=plt.imread('target.jpg')
    sourceImg=np.array(Image.fromarray(sourceImg).resize([300, 300]))
    targetImg=np.array(Image.fromarray(targetImg).resize([300, 300]))
    im1=np.fromfile('im1')
    im2=np.fromfile('im2')
    im1.shape=[int(len(im1)/2),2]
    im2.shape=[int(len(im2)/2),2]
    add=np.array([[0,0],[0,300],[300,0],[300,300],[0,150],[150,0],[150,300],[300,150]])
    im1=np.concatenate((add,im1),axis=0)
    im2=np.concatenate((add,im2),axis=0)
    warp_frac=np.linspace(0,1,60)
    dissolve_frac=np.linspace(0,1,60)
    morphed_im=morph_tri(sourceImg,targetImg,im1,im2, warp_frac, dissolve_frac)
    
    frames, x, y, z = morphed_im.shape
    morph_list = []
    for i in range(0, frames):
        image = Image.fromarray(morphed_im[i, :, :, :], 'RGB')
        morph_list.append(morphed_im[i, :, :, :])
        image.save('photo.jpg')
        image.show()
        plt.show()
        imageio.mimsave('./morphResults.gif', morph_list)