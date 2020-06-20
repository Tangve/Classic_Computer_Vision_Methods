For this project, we need to run two functions

First, run click_correspondence in terminal will save two numpy array im1 and im2 
which are correspondence points array

Then, run morph_tri to give you result gif named morphResults

Somethings need to be mentioned:

My input are two square images, and resize then to 300 by 300, so if you want to use 
images that are not square, you need to change some parameters

I used 60 images to produce gif. If you run morph_tri, it will first produce 60 images by 
sequence(they look strange, but it is fine), then produce a gif called morphResults in the 
folder.

I used interp2 function which was provided by TA in the first project.

There is no function in helper.py
