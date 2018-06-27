from  skimage import io
s = io.imread('..\\ui\\Img\\00F266.bmp')
io.imsave('1.png',s*255)