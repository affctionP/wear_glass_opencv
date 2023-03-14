import cv2 
import numpy as np
import math
def slop(x1,x2,y1,y2):
    return (y2-y1)/(x2-x1)
def dgree(s1,s2):
    return math.degrees(math.atan((s1-s2)/1+(s1*s2)))
def sor (img):
    img[img==0]=255
    return img
face_cascade = cv2.CascadeClassifier('/home/atefeh/.local/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('/home/atefeh/.local/lib/python3.9/site-packages/cv2/data//haarcascade_eye.xml')

image = cv2.imread('ha.jpg')
glass_img = cv2.imread('glass.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

centers = []
"""cv2.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) 
scaleFactor : Parameter specifying how much the image size is reduced at each image scale. 
minNeighbors : Parameter specifying how many neighbors each candidate rectangle should have to retain it. This parameter will affect the quality of the detected faces: higher value results in less detections but with higher quality. We're using 5 in the code.
flags : Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
minSize : Minimum possible object size. Objects smaller than that are ignored.
maxSize : Maximum possible object size. Objects larger than that are ignored.

"""
slops=[]
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# iterating over the face detected
for (x, y, w, h) in faces:

    # create two Regions of Interest.
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    cv2.rectangle(image,(x,y),(x+w,h+y),(0,255,0))

    # Store the coordinates of eyes in the image to the 'center' array
    for  i  in range (len(eyes)):
        (ex, ey, ew, eh) = eyes[i]
        centers.append((x + int(ex + 0.5 * ew), y + int(ey + 0.5 * eh)))
        cv2.rectangle(image,(x+ex,y+ey),(x+ex+ew,y+eh+ey),(0,0,255))
        if i%2 != 0 :
            slops.append(slop(eyes[i-1][0],ex,eyes[i-1][1],ey))

dd=dgree(0,slops[0])

if len(centers) > 0:
    # change the given value of 2.15 according to the size of the detected face
    glasses_width = 2.16 * abs(centers[1][0] - centers[0][0])
    overlay_img = np.ones(image.shape, np.uint8) * 255
    h, w = glass_img.shape[:2]
    scaling_factor = glasses_width / w

    #overlay_glasses = cv2.resize(glass_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)   
    x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]


    
    # rotate 
    #M = cv2.getRotationMatrix2D((w/2,h/2),6.34,0.5) 
    scaleFactor=scaling_factor
    M = cv2.getRotationMatrix2D(center=(w/2,h/2), angle=dd, scale=scaleFactor) #rotate about center of image.

    #choose a new image size.
    newX,newY = w*scaleFactor,h*scaleFactor
    #include this if you want to prevent corners being cut off
    r = np.deg2rad(dd)
    newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))

    #the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    #So I will find the translation that moves the result to the center of that region.
    (tx,ty) = ((newX-w)/2,(newY-h)/2)
    M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
    M[1,2] += ty
    glass_img_gray=cv2.cvtColor(glass_img,cv2.COLOR_BGR2GRAY)
    th, dst = cv2.threshold(glass_img_gray,170,255, cv2.THRESH_BINARY_INV); 
    glass_img_1 = cv2.bitwise_and(glass_img_gray, glass_img_gray, mask=dst)
    glass_img_b=cv2.cvtColor(glass_img_1,cv2.COLOR_GRAY2BGR)
   
    rotate_30 =cv2.warpAffine(glass_img, M, dsize=(int(newX),int(newY)))
    rotate_30[rotate_30==0]=255
    
    x -= 0.26 * rotate_30.shape[1]
    y += 0.4 * rotate_30.shape[0]
    h, w = rotate_30.shape[:2]

    #rotate_30 = cv2.warpAffine(overlay_glasses,M,(w,h))
    #rotate_30=sor(rotate_30)
    #ret,mask_30 =cv2.threshold(rotate_30, 110, 255, cv2.THRESH_BINARY_INV)
 
    #rotate_30 =cv2.bitwise_and(rotate_30, rotate_30, mask=mask_30)
    overlay_img[int(y):int(y + h), int(x):int(x + w)] = rotate_30
    #overlay_img[int(y):int(y + h), int(x):int(x + w)] = overlay_glasses  
    
    # Create a mask and generate it's inverse.
    gray_glasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    gray_glasses =sor(gray_glasses)
    ret, mask = cv2.threshold(gray_glasses, 180, 255, cv2.THRESH_BINARY)
    
    mask_inv = cv2.bitwise_not(mask)
    temp = cv2.bitwise_and(image, image, mask=mask)
    temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
    final_img = cv2.add(temp, temp2)

cv2.imshow("h",rotate_30)
cv2.waitKey()
cv2.destroyAllWindows()