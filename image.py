

import cv2
#store your image xml data file in a variable 
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img=cv2.imread("news.jpg")
#converting a colour image to gray colour image
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.2,minNeighbors=5)
#showing a rectangle on your face 
for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

resized=cv2.resize(img,(int(img.shape[1]/3),int(img.shape[1]/3)))

cv2.imshow("gray",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(faces)
