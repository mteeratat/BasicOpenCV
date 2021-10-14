import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def create_dataset(img,id,img_id):
    cv2.imwrite("data4/pic."+str(id)+"."+str(img_id)+".jpg",img) 


def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,clf):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray,scaleFactor,minNeighbors)
    coords = []
    for (x,y,w,h) in features:
        id,con = clf.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)

        if id == 1:
            if con <= 50 :
                cv2.putText(img,"Steve",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
            else :
                cv2.putText(img,"Unknown",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
            if con < 100 :
                con = "  {0}%".format(round(100 - con ))
            else :
                con = "  {0}%".format(round(100 - con ))

        elif id == 2:
            if con <= 50 :
                cv2.putText(img,"Bill",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
            else :
                cv2.putText(img,"Unknown",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
            if con < 100 :
                con = "  {0}%".format(round(100 - con ))
            else :
                con = "  {0}%".format(round(100 - con ))

        elif id == 3:
            if con <= 50 :
                cv2.putText(img,"May",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
            else :
                cv2.putText(img,"Unknown",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
            if con < 100 :
                con = "  {0}%".format(round(100 - con ))
            else :
                con = "  {0}%".format(round(100 - con ))

        elif id == 4:
            if con <= 50 :
                cv2.putText(img,"Pim",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
            else :
                cv2.putText(img,"Unknown",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
            if con < 100 :
                con = "  {0}%".format(round(100 - con ))
            else :
                con = "  {0}%".format(round(100 - con ))

        elif id == 5:
            if con <= 50 :
                cv2.putText(img,"M",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
                count.append('1')
            else :
                cv2.putText(img,"Unknown",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
                count.clear()
            if con < 100 :
                con = "  {0}%".format(round(100 - con ))
            else :
                con = "  {0}%".format(round(100 - con ))
            
        #print(str(con))
        coords = [x,y,w,h]
        #print(id)
    return img,coords

def detect(img,faceCascade,img_id,clf,count):
    if len(count) > 50:
        img,coords = draw_boundary(img,faceCascade,1.1,10,(0,255,),clf)
    else :
        img,coords = draw_boundary(img,faceCascade,1.1,10,(0,0,255),clf)
    #img,coords = draw_boundary(img,eyeCascade,1.1,10,(255,0,0),'Eye')
    if len(coords) == 4:
        id = 5
        result = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        #create_dataset(result,id,img_id)
    return img    

'''
print(cv2.__version__)
img = cv2.imread("Obama.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('result.jpg',img)
'''

img_id = 0
#cap = cv2.VideoCapture('Video.mp4')
#cap = cv2.VideoCapture('#เมอาพิมฐา แปลกกว่าขนม ก็คนรีวิวนี่แหละ! _ MayyR x Pimtha.mp4')
cap = cv2.VideoCapture(0)
clf = cv2.face.LBPHFaceRecognizer_create()
#clf.read("classifierSteve.xml")
#clf.read("classifierBill.xml")
#clf.read("classifierPim.xml")
#clf.read("classifierMay.xml")
#clf.read("classifierPM.xml")
clf.read("classifierM.xml")
count = []
while True :
    ret,frame = cap.read()
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)q
    frame = detect(frame,faceCascade,img_id,clf,count)
    cv2.imshow('frame',frame)
    print(len(count))
    img_id = img_id + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows

'''
img = cv2.imread('Obama2.jpg')
img = cv2.line(img,(0,0),(255,255),(0,0,255),10)
img = cv2.arrowedLine(img,(0,0),(400,400),(255,0,0),10)
img = cv2.rectangle(img,(384,0),(510,128),(0,0,255),10)
img = cv2.circle(img,(447,63),63,(0,255,0),10)
img = cv2.putText(img,'OpenCV',(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
cv2.imshow('Result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


