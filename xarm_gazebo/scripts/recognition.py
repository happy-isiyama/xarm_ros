#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import sys
import rospy
import math
import random
import threading
import numpy as np
import moveit_commander
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

if sys.version_info < (3, 0):
    PY3 = False
    import Queue as queue
else:
    PY3 = True
    import queue
    

COLOR_DICT = {
    'red': {'lower': np.array([0, 43, 46]), 'upper': np.array([10, 255, 255])},
    'blue': {'lower': np.array([90, 100, 100]), 'upper': np.array([130, 255, 255])},
    'green': {'lower': np.array([50, 60, 60]), 'upper': np.array([77, 255, 255])},
    'yellow': {'lower': np.array([20, 40, 46]), 'upper': np.array([34, 255, 255])},
}


class GazeboCamera(object):
    def __init__(self, topic_name='/camera/image_raw/compressed'):
        self._frame_que = queue.Queue(10)
        self._bridge = CvBridge()
        self._img_sub = rospy.Subscriber(topic_name, CompressedImage, self._img_callback)

    def _img_callback(self, data):
        if self._frame_que.full():
            self._frame_que.get()
        self._frame_que.put(self._bridge.compressed_imgmsg_to_cv2(data))
    
    def get_frame(self):
        if self._frame_que.empty():
            return None
        return self._frame_que.get()


def get_recognition_rect(frame, lower=COLOR_DICT['red']['lower'], upper=COLOR_DICT['red']['upper'], show=True):
    gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)
    erode_hsv = cv2.erode(hsv, None, iterations=2)
    inRange_hsv = cv2.inRange(erode_hsv, lower, upper)
    contours = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    rects = []
    for _, c in enumerate(contours):
        rect = cv2.minAreaRect(c)
        if rect[1][0] < 20 or rect[1][1] < 20:
            continue
        if PY3:
            box = cv2.boxPoints(rect)
        else:
            box = cv2.cv.BoxPoints(rect)
        cv2.drawContours(frame, [np.int0(box)], -1, (0, 255, 255), 1)
        rects.append(rect)
    
    if show:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rospy.signal_shutdown('key to exit')
    return rects


if __name__ == '__main__':
    rospy.init_node('camera_only_node', anonymous=False)
    rate = rospy.Rate(10.0)

    color = COLOR_DICT['red']
    cam = GazeboCamera(topic_name='/camera/image_raw/compressed')

    while not rospy.is_shutdown():
        rate.sleep()
        frame = cam.get_frame()
        if frame is None:
            continue
        rects = get_recognition_rect(frame, lower=color['lower'], upper=color['upper'])