'''
Purpose: Easy video labeling using tracking
'''
import cv2
import os
from tracker.pascal_voc_io import XMLWriter
import argparse
from tqdm import tqdm

argparser = argparse.ArgumentParser(description='Multitracker for objects in the video')
argparser.add_argument('-i','--input',
                        help='input video (.mp4)')
argparser.add_argument('-f','--frame',
                        help='frame number to start annotation from',
                        default='1')
argparser.add_argument('-d','--dir',
                        help='relative path to output directory')

args = argparser.parse_args()

INPUT_VIDEO = args.input
OUTPUT_DIR = args.dir
FRAME_NUMBER = int(args.frame)

#############################################
# open video handle
frame = FRAME_NUMBER
cap = cv2.VideoCapture(INPUT_VIDEO)
for i in tqdm(range(frame)):
    ret, image = cap.read()

# check if output path exists
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print("Created {} directory\n".format(OUTPUT_DIR))

# initialize multitracker
tracker = cv2.MultiTracker_create()
tracker_list = []
init_once = False

# draw bboxs around objects and label them
while(True):
    bbox = cv2.selectROI('tracking', image)

    label = input("Label the object: ")
    if not label:
        break

    tracker_list.append([label, bbox])

waitkey = 0
while cap.isOpened():
    ret, image = cap.read()

    if ret:
        # press 'esc' to exit
        if cv2.waitKey(waitkey) == 27:
            # just make sure
            ans = input("\nDo you want to quit? [y/N] ")
            if ans == 'y' or ans == 'Y':
                break
            else:
                continue

        # press 'f' to print out current frame number
        if cv2.waitKey(waitkey) == 102:
            print("Frame number: ", frame)

        # press 'w' to switch waitkey to 0
        if cv2.waitKey(waitkey) == 119:
            print("\nChanging waitkey to 0\n")
            waitkey = 0

        # press 'e' to switch waitkey to 1
        if cv2.waitKey(waitkey) == 101:
            print("\nChanging waitkey to 1\n")
            waitkey = 1

        # if 'DEL' pressed, reinitialize tracker (+ bboxs and labels)
        if cv2.waitKey(waitkey) == 48:
            tracker_list = []
            tracker = cv2.MultiTracker_create()
            init_once = False

            while(True):
                bbox = cv2.selectROI('tracking', image)
                label = input("Label the object: ")
                if not label:
                    break

                tracker_list.append([label, bbox])

        # add bboxs for tracking
        if not init_once:
            for label, bbox in tracker_list:
                tracker.add(cv2.TrackerMIL_create(), image, bbox)
            init_once = True

        # update tracker
        ret, boxes = tracker.update(image)
        try:
            for idx, item in enumerate(tracker_list):
                item[1] = list(boxes[idx])
        except:
            print('tracker_list problem')

        # write .jpg and .xml to disk
        filename = str(frame).rjust(5, '0') + '.jpg'
        cv2.imwrite(OUTPUT_DIR + filename, image)
        XMLWriter(folder=OUTPUT_DIR,
                 filename=filename,
                 imgSize=image.shape,
                 localImgPath=OUTPUT_DIR + filename,
                 detection_info=tracker_list)

        # draw tracker bboxs on image
        for newbox in boxes:
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(image, p1, p2, (200,0,0))

        cv2.imshow('tracking', image)

    frame += 1
