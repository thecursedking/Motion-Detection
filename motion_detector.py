# importing OpenCV, time and Pandas library 
import cv2, time, pandas
# importing datetime class from datetime library
from datetime import datetime

#Assigning our first_frame to none
first_frame=None
# List when any moving object appear
status_list=[None,None]
# Time of movement
times=[]
# Initializing DataFrame, one column is start 
# time and other column is end time 
df=pandas.DataFrame(columns=["Start","End"])

#Capturing Video
video=cv2.VideoCapture(0)

# Infinite while loop to treat stack of image as video 
while True:
    # Reading frame(image) from video
    check, frame = video.read()

    # Initializing status = 0(no status)
    status=0

    # Converting color image to gray_scale image
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)

    # In first iteration we assign the value 
    # of static_back to our first frame 
    if first_frame is None:
        first_frame=gray
        continue

    # Difference between first_frame background
    # and current frame(which is GaussianBlur)
    delta_frame=cv2.absdiff(first_frame,gray)

    # If change in between static background and
    # current frame is greater than 30 it will show white color(255)
    thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

    # Finding contour of moving object
    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        status=1

        (x, y, w, h)=cv2.boundingRect(contour)
        # making green rectangle arround the moving object
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)

    # Appending status of motion
    status_list.append(status)

    status_list=status_list[-2:]

    # Appending Start time of motion
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())

    # Appending End time of motion    
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

    # Displaying image in gray_scale
    cv2.imshow("Gray Frame",gray)
    # Displaying the difference in currentframe to 
    # the staticframe(very first_frame)
    cv2.imshow("Delta Frame",delta_frame)
    # Displaying the black and white image in which if 
    # intencity difference greater than 30 it will appear white 
    cv2.imshow("Threshold Frame",thresh_frame)
    # Displaying color frame with contour of motion of object
    cv2.imshow("Color Frame",frame)

    key=cv2.waitKey(1)
    # if q entered whole process will stop
    if key==ord('q'):
        # if something is movingthen it append the end time of movement
        if status==1:
            times.append(datetime.now())
        break
#Print status_list and times
print(status_list)
print(times)
# Appending time of motion in DataFrame
for i in range(0,len(times),3):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("Times.csv")

#Destroying all the windows.
video.release()
cv2.destroyAllWindows
