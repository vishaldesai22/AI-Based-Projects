#this library is used to recognisation or capturing the images or video
import cv2

#facecapture stores the standard methods and functions 
facecapture=cv2.CascadeClassifier("C:/Users/visha/AppData/Local/Programs/Python/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default")

#videocapture stores the captured data in it. 
videocapture = cv2.VideoCapture(0)
while True:#loop


      #reads the data which is captured
    videodata=videocapture.read()

    
    #this is used to provide color to the border to the rectangle of caamera
    colorr= cv2.cvtColor(videodata,cv2.COLOR_BGR2GRAY)


    faces = facecapture.detectMultiScale(

        colorr,
        scaleFactor=1.1,
        minNeighbours=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )



    for(x,y,w,h)in faces:
        cv2.rectangle(videodata,(x,y),(x+w,y+h),(0,255,0),2)


    
  
    #show method is used to display the video in a frame (V_Capture is the name of the capturing frame) 
    cv2.imshow("V_Capture",videodata)
    if cv2.waitKey(5)==ord("c"):
        break
videocapture.release()
    