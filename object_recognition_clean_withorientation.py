import cv2
import numpy as np
import time
from typing import Any
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String
from std_msgs.msg import Float32
from rosbot_interfaces.msg import ObstacleInfo


min_area = 200  # to tune on the real object dataset
max_area = 100000
obstacle_list = [] # declaration of the list that will be used in Obstacle Class

class Obstacle:
    
    def __init__(self, name, shape, position_x, position_y, orientation, area):
        self.name = name
        self.shape = shape
        self.position_x = position_x
        self.position_y = position_y
        self.orientation = orientation
        self.area = area

#creation of the ROS2 node
class ObjectRecognitionNode(Node):

    def __init__(self):
        super().__init__("object_recognition")
        self.msg_pub = self.create_publisher(ObstacleInfo, "/obstacle_info", 10)
        self.timer = self.create_timer(1.0, self.publish_message)
        self.get_logger().info = ("Obstacle_info Node has been started")

    def publish_message(self):
        message = ObstacleInfo()
        for obstacle in obstacle_list:
            message.name = obstacle.name
            message.shape = obstacle.shape
            message.position.x = float(obstacle.position_x)
            message.position.y = float(obstacle.position_y)
            message.position.theta = float(obstacle.orientation)
            message.area = int(obstacle.area) #**TO SET FLOAT IN INTERFACE**
            self.msg_pub.publish(message)
            #self.get_logger().info("Published message %s")


def calculate_rotation_angle(poly_approx):
    if len(poly_approx) < 3:
        raise ValueError("The polygon must have at least three vertices.")

    # Take the first two points of the polygon to calculate the vector
    x1, y1 = poly_approx[0][0]  # Access the x and y coordinates of the first point
    x2, y2 = poly_approx[1][0]  # Access the x and y coordinates of the second point

    # Calculate the vector between the first two points using numpy arrays
    vector = np.array([x2 - x1, y2 - y1])

    angle_radians = np.arctan2(vector[1], vector[0])
    angle_degrees = np.degrees(angle_radians)

    # Return the angle of rotation
    return angle_degrees


def shape_recognition(img):
    img_copy = img.copy()  # save a copy of the image to display the result later
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert in grayscale
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)  # convert in binary color
    """
    this is needed because in the map:
     1) unexplored zones of the map are gray (127 on grayscale);
     2) clear explored zones are white (255 on grayscale);
     3) the contours of the object detected are black (0 on grayscale)

     the binary result image uniform the "background" of the map to achive
     better detection 
    """
    edges = cv2.Canny(binary, 30, 100)  # edge detection (https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)

    # Image processing to clean the edge detected
    # (https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html)
    dilated = cv2.dilate(edges, None, iterations=2)
    eroded = cv2.erode(dilated, None, iterations=2)

    # find contours of eroded image with these specifications:
    #   1) cv2.RETR_EXTERNAL: consider only the most external contours of each detection (better performance ?)
    #      [https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html]
    #   2) cv2.CHAIN_APPROX_SIMPLE : assuming that the countours are almost straight line, save only the 2 end-points for each line
    #      [https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html]
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Introduce a variable to count the number of real objects
    j = 0

    for i, contour in enumerate(contours):

        area = cv2.contourArea(contour)  # calculate area of contour

        if area < min_area:  # ignore small areas (e.g. one pixel, false detection, ...)
            continue
        if area > max_area:  # ignore the room walls detections
            continue

        # A Real Object is detected
        j = j + 1

        # Approximate the contour of the object by a polygon
        # [https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html]
        epsilon = 0.03 * cv2.arcLength(contour, True)
        poly_approx = cv2.approxPolyDP(contour, epsilon, True)

        print("CONTOUR : ", j)
        print("\n1) Number of sides of the polygon approximating contour ", " : ", len(poly_approx))

        # Identify shape by aspectRatio and number of sides
        if len(poly_approx) == 3:
            shape = "triangle"
        elif len(poly_approx) == 4:
            # it could be a square or a rectangle
            (x, y, w, h) = cv2.boundingRect(poly_approx)  # approx. the polygon with a bounding box
            aspect_ratio = float(w) / float(h)

            if aspect_ratio >= 0.95 or aspect_ratio <= 1.05:
                shape = "square"
            else:
                shape = "rectangle"
        else:  # the circle is approx. as a polygon so it could have a number of sides larger than four
            shape = "circle"

        print("\n2) Shape : ", shape)
        print("\n")

        # Visual output as windows with masked object
        mask = np.zeros_like(gray)  # make a total obscuring mask
        cv2.drawContours(mask, [contour], -1, 255, 1)

        single_object = cv2.bitwise_and(img_copy, img_copy,
                                        mask=mask)  # masking the original image to show only the desired object

        # Detect the center of the object
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        print("The coordinates of the center are:\n", "cX:\t", cX, "\t", "cY:\t", cY, "\n")

        # Detect the orientation of polygonal objects
        if len(poly_approx) <= 4:
            orientation_deg = calculate_rotation_angle(poly_approx)

        else:
            orientation_deg = 0

        print("The orientation is:\t", orientation_deg, "\n")

        cv2.circle(single_object, (cX, cY), 1, (255, 255, 255),
                   -1)  # draw the center of the shape on the single object image

        # Legenda for single object
        text = f"Center=({cX},{cY}), Area={area}, Shape={shape}, Orientation_deg={orientation_deg}"
        cv2.putText(single_object, text, (1, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        #create the object of the Obstacle Class
        obstacle = Obstacle(f"Obstacle {i+1}", shape, cX, cY, orientation_deg, area) 
        obstacle_list.append(obstacle)

        # Display the single object image in a separate, resizable window
        cv2.namedWindow(f"Object {j}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"Object {j}", single_object)

        # Draw the contour and center of the shape on the original image
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
        cv2.circle(img, (cX, cY), 2, (0, 0, 255), -1)

    # Display the original image with all objects highlighted
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)

def main(args=None):

    start_time = time.time()

    # img = cv2.imread('/home/edo/progetto_lvm/obj_recognition/map.png')
    # img = cv2.imread('c:/Users/admin/Desktop/Uni/LabMecc/Progetto/map.png')
    img = cv2.imread('/home/nuctella/my_ros2_ws/src/rosbot_object_recognition/rosbot_object_recognition/object_recognition/map.png') # This path should work well for everyone of us
    shape_recognition(img)
    rclpy.init(args=args)
    node = ObjectRecognitionNode()
    rclpy.spin(node)
    rclpy.shutdown()
    print("Elapsed time: ", (time.time() - start_time) * 1000, " ms")

    cv2.waitKey(0)
