#!/usr/bin/env python
### !/usr/bin/python3
### !/usr/bin/env python #for python2 Config

import rospy
from sensor_msgs.msg import CompressedImage, Image
import numpy as np
import cv2
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox


# Contains the bbox extents and object label + id information
bounds_array_left = np.array([])
bounds_array_right = np.array([])

# create a label to text dict mapping
label_to_text = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck', 9: 'Traffic Light', 11: 'Stop Sign', 14: 'Bird', 15 : 'Cat', 16: 'Dog'}

current_image_left = None
current_image_right = None

def convertBack(x, y, w, h):
    xmin = int(x)
    xmax = int(round(x + w))
    ymin = int(y)
    ymax = int(round(y + h))
    return xmin, ymin, xmax, ymax


def callback_yolo_left(bbox_array_2D):
    global bounds_array_left
    bounds_array_temp = callback_yolo(bbox_array_2D)
    bounds_array_left = bounds_array_temp.copy()


def callback_yolo_right(bbox_array_2D):
    global bounds_array_right
    bounds_array_temp = callback_yolo(bbox_array_2D)
    bounds_array_right = bounds_array_temp.copy()


def callback_yolo(bbox_array_2D):
    bounds_array_temp = np.empty((0, 6))
    if len(bbox_array_2D.boxes) == 0: 
        return bounds_array_temp
    for bbox in bbox_array_2D.boxes:
        x = bbox.pose.position.x
        y = bbox.pose.position.y
        w = bbox.dimensions.x
        h = bbox.dimensions.y
        obj_label = bbox.label
        obj_confidence = bbox.value
        xmin, ymin, xmax, ymax = convertBack(x, y, w, h)
        # Skip all non-relevant classes
        if not (obj_label in label_to_text.keys()):
            continue
        bounds_array_temp = np.vstack(
            (bounds_array_temp,
             np.array([ xmin, ymin, xmax, ymax, obj_label, obj_confidence ])))
    return bounds_array_temp


def callback_compresed_image_right(msg):
    global bounds_array_right
    global current_image_right
    # decode compressed image into cv2 image
    np_arr = np.fromstring(msg.data, np.uint8)
    if is_compresed_image:
        current_image_right = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        current_image_right = np_arr.reshape(msg.height, msg.width, 3)

    # draw the bounding box of camera detection
    for bound in bounds_array_right:
        cv2.rectangle(current_image_right,
                    (int(bound[0]), int(bound[1])),
                    (int(bound[2]), int(bound[3])),
                    (0, 255, 0), 2)
        # draw the label
        cv2.putText(current_image_right, label_to_text[int(bound[4])],
                    (int(bound[0]), int(bound[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    bounds_array_right = np.array([])

def callback_compresed_image_left(msg):
    global bounds_array_left
    global current_image_right
    # decode compressed image into cv2 image
    np_arr = np.fromstring(msg.data, np.uint8)
    if is_compresed_image:
        current_image_left = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        current_image_left = np_arr.reshape(msg.height, msg.width, 3)

    # draw the bounding box of camera detection
    for bound in bounds_array_left:
        cv2.rectangle(current_image_left,
                    (int(bound[0]), int(bound[1])),
                    (int(bound[2]), int(bound[3])),
                    (0, 255, 0), 2)
        # draw the label
        cv2.putText(current_image_left, label_to_text[int(bound[4])],
                    (int(bound[0]), int(bound[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    bounds_array_left = np.array([])
    cv2.namedWindow("image_left", cv2.WINDOW_NORMAL)
    cv2.imshow("image_left", current_image_left)
    cv2.namedWindow("image_right", cv2.WINDOW_NORMAL)
    cv2.imshow("image_right", current_image_right)
    cv2.waitKey(1)
    
    

if __name__ == '__main__':
    rospy.init_node('cam_detect_vis_node', anonymous=True)
    visualize_image = rospy.get_param('~visualize_image', True)
    is_compresed_image = rospy.get_param('~is_compresed_image', False)
    left_camera_topic = "/pylon_camera_node_left_infra/image_rect/flipped"
    right_camera_topic = "/pylon_camera_node_right_infra/image_rect/flipped"
    callback_msg_type = Image
    if is_compresed_image:
        left_camera_topic += "/compressed"
        right_camera_topic += "/compressed"
        callback_msg_type = CompressedImage
    
    if visualize_image:
        rospy.Subscriber(right_camera_topic,
                         callback_msg_type, callback_compresed_image_right, queue_size=1)
        rospy.Subscriber(left_camera_topic, 
                         callback_msg_type, callback_compresed_image_left, queue_size=1)
        
    rospy.Subscriber("/bbox_array_right", BoundingBoxArray,
                     callback_yolo_right)
    rospy.Subscriber("/bbox_array_left", BoundingBoxArray,
                     callback_yolo_left)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()