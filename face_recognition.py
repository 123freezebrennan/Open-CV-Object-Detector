import cv2
thres = 0.5
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

className = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, live_feed = cap.read()
    classIds, confs, bbox = net.detect(live_feed, confThreshold=0.5)
    print(classIds, bbox)
    if len(classIds) != 0:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(live_feed, box, color = (255, 0, 0), thickness = 3)
            cv2.putText(live_feed, classNames[classId - 1], (box[0] + 150, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(live_feed, str(round(confidence * 100)), (box[0] + 300, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Open CV Python Project - Press q To Quit", live_feed)
    cv2.waitKey(1)
