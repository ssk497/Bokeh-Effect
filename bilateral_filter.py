import cv2
import numpy as np
import math
def bilateral_filter_own(source, w, sigma_d, sigma_r):
    arr=np.arange(-w,w+1)
    X,Y = np.meshgrid(arr,arr);
    G=np.exp(-(pow(X,2)+pow(Y,2))/(2*sigma_d*sigma_d))
    filtered_image = np.zeros(source.shape)
    for i in range(0,len(source)):
        for j in range(0,len(source[1])):
            if(~np.isnan(source[i][j])):
                iMin = max(i-w,0)
                iMax = min(i+w,len(source)-1)
                jMin = max(j-w,0)
                jMax = min(j+w,len(source[0])-1)
                #print iMin,iMax,jMin,jMax
                
                I = source[iMin:iMax+1,jMin:jMax+1]
                P=(I-source[i][j])
                P=pow(P,2)
                H=np.exp(-1*P/(2*sigma_r*sigma_r))
                G1=G[iMin-i+w:iMax-i+w+1,jMin-j+w:jMax-j+w+1]
                F=np.multiply(H,G1)
                num=sum(np.multiply(F.flatten(),I.flatten()))
                den=sum(F.flatten())
                filtered_image[i][j]=num/den
        print(i)
    return filtered_image

def bilateral_filter(A,w,sigma):
    return bilateral_filter_own(A,w,sigma[0],sigma[1])

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') -min_val) / (max_val-min_val)
    return out

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])    
