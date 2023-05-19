#OpenCV is a Python library that allows you to perform image processing and computer vision tasks
#matplotlib.pyplot is a collection of functions that make matplotlib work like MATLAB
import cv2
import matplotlib.pyplot as plt

#Haar cascade is an algorithm that can detect objects in images, irrespective of their scale in image and location

#Haar cascade classifier for face detection
fd=cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

#Haar cascade classifier for smile detection
sd=cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)

vid=cv2.VideoCapture(0) #return video from the first webcam on your computer

seq=0
captured=False

while not captured: #loop runs while the condition is true
    flag, img=vid.read()

    if flag:
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #for gray-scale video
        #_, img_binary=cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY) #for binary video
        #x1,y1,w,h=(200,200,200,200)

        faces=fd.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80,80)
        )
        #smile=fd.detectMultiScale(img_gray,1.1,5)
        
        for x1,y1,w,h in faces:
            face=img_gray[y1:y1+h,x1:x1+w].copy()
            
            smiles=sd.detectMultiScale(
                face,1.1,15,minSize=(50,50)
            )
            #print(len(smiles))
            
            if len(smiles)==1:
                seq+=1
                
                if seq==5:
                    captured=cv2.imwrite('kirtan.png',img)
                    break
                
                xs,ys,ws,hs=smiles[0]
                
                cv2.rectangle( #create rectangle in image
                    img,pt1=(xs+x1,ys+y1),pt2=(xs+x1+ws,ys+y1+hs),
                    color=(0,255,0),
                    thickness=3
                )

            else:
                seq=0
            #img_cropped=img[y1:y1+h,x1:x1+w,:] #crope the image/x
            
            cv2.rectangle( #create rectangle in image
                img,pt1=(x1,y1),pt2=(x1+w,y1+h),
                color=(255,0,0),
                thickness=3
            )

        cv2.imshow('Preview',img) #show the image/video
        key=cv2.waitKey(1)
        
        if key==ord('x'):
            break
    
    else:
        break

vid.release()
cv2.destroyAllWindows()