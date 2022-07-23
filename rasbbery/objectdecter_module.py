import cv2

thres = 0.45 # Threshold to detect object



classNames= []
classFile = 'coco.names.txt'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt.txt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getobjects(img,drop=True,objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=0.25)
    print(classIds,bbox)
    if len(objects)==0:
        objects=classNames
        objectInfo = []

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:


                objectInfo.append([box, className])
                if (drop):

                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    return img,objectInfo



cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)
while True:
    success,img=cap.read()
    result,objectInfo=getobjects(img)
    cv2.imshow("Output", img)
    if cv2.waitKey(1)==ord('q'):
        break

