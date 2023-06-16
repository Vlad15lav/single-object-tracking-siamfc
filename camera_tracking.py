import argparse
import torch
import cv2
import numpy as np

from tracker.tracker import SiamFCTracker


def get_args():
    parser = argparse.ArgumentParser('Video Tracking')
    parser.add_argument('--weight-path', type=str, default="weights/SiamFC.pth", help='path weights training')
    parser.add_argument('--scale', type=float, default=1.5, help='scale window size')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = get_args()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Камера не запустилась!")
        exit()


    cv2.namedWindow("Tracking")

    i_frame = 0
    track_mode, x1, y1, x2, y2 = False, -1, -1, -1, -1
    image_z, select_object = None, None
    def select_image_object(event, x, y, flags, params):
        global x1, y1, x2, y2, image_z, select_object, track_mode

        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1 = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            x2, y2 = x, y
            if x1 > x:
                x1, x2 = x, x1
            if y1 > y:
                y2, y1 = y1, y

            select_object = frame[y1:y2, x1:x2]

            x1, y1, x2, y2 = x1 // opt.scale, y1 // opt.scale, x2 // opt.scale, y2 // opt.scale
            cv2.imshow("Image Tracking", select_object)
            track_mode = True
            i_frame = 0
            
    cv2.setMouseCallback("Tracking", select_image_object)


    tracker = SiamFCTracker(opt.weight_path)
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Ошибка в получение следующего кадра!")
            exit()
        
        if track_mode:
            if i_frame == 0:
                tracker.init_tracker(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), x1, y1, x2 - x1, y2 - y1)
                bbox = (x1, y1, x2, y2)
            else:
                bbox = tracker.next_frame(frame)
            
            i_frame += 1
            
            frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)



        if opt.scale != 1:
            dim = frame.shape[:2]

            frame = cv2.resize(frame, (int(dim[1] * opt.scale), int(dim[0] * opt.scale)), interpolation=cv2.INTER_CUBIC)

        cv2.imshow("Tracking", frame)
        
        if cv2.waitKey(1) == ord('q'):
            exit()


    cap.release()
    cv2.destroyAllWindows()

