import cv2
import numpy as np

def lane_roi(image):
  imshape = image.shape
  #gray = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  vertices = np.array([[(0,imshape[0]),(0,imshape[0]*0.7),(imshape[1]/2-80,imshape[0]*0.55),(imshape[1]/2+80,imshape[0]*0.55),(imshape[1],imshape[0]*0.7) ,(imshape[1],imshape[0])]], dtype = np.int32)
  mask = np.zeros(image.shape, image.dtype)
  cv2.fillPoly(mask, vertices,(100,150,100))
  result = cv2.addWeighted(image,1,mask,0.4,0)
  return result
