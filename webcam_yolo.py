import torch
import cv2

#load YOLOv5 model (small version for better speed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Could not open webcam")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error:could not read frame")
        break


    #inference
    results = model(frame)


    #render results on frame
    annotated_frame = results.render()[0]

    #show results
    cv2.imshow("yolov5s webcam detection", annotated_frame)

    #exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()