#!/usr/bin/env python3

from pynput import keyboard
import math
import threading
import asyncio
import websockets
import json
from camera import *
from vector2d import Vector2D
import itertools
import random
import angles
import time
from math import sqrt
from enum import Enum
import csv
import os
from datetime import datetime
import argparse

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
purple = (128, 0, 128)
magenta = (255, 0, 255)
cyan = (255, 255, 0)
yellow = (50, 255, 255)
black = (0, 0, 0)
white = (255, 255, 255)
grey = (100, 100, 100)

target_info = {}
robot_info = {}

INIT_POSITION_TIMEOUT = 2.0  # Stop drawing init positions if no update for 2 seconds
last_robot_info_update = 0  # Timestamp of last received robot_info

GAME_TIME = 5 * 60
# random.seed(1)

# Command-line arguments
save_log = False
save_video = False

# CSV logging variables
csv_file = None
csv_writer = None
experiment_config = None
csv_initialized = False
last_logged_simulation_time = -1  # Track the last logged simulation time to prevent duplicate entries
max_robots_in_target_ever_seen = 0  # Track the maximum number of robots that have been in target
arrival_times = {}  # Dictionary to store arrival times: {num_robots: simulation_time}

# Video recording variables
video_writer = None
video_initialized = False
video_lock = threading.Lock()
next_video_frame_time = 0.0

class Tag:
    def __init__(self, id, raw_tag):
        self.id = id
        self.raw_tag = raw_tag
        self.corners = raw_tag.tolist()[0]

        # Individual corners (e.g. tl = top left corner in relation to the tag, not the camera)
        self.tl = Vector2D(int(self.corners[0][0]), int(self.corners[0][1])) # Top left
        self.tr = Vector2D(int(self.corners[1][0]), int(self.corners[1][1])) # Top right
        self.br = Vector2D(int(self.corners[2][0]), int(self.corners[2][1])) # Bottom right
        self.bl = Vector2D(int(self.corners[3][0]), int(self.corners[3][1])) # Bottom left

        # Calculate centre of the tag
        self.centre = Vector2D(int((self.tl.x + self.tr.x + self.br.x + self.bl.x) / 4),
                               int((self.tl.y + self.tr.y + self.br.y + self.bl.y) / 4))

        # Calculate centre of top of tag
        self.front = Vector2D(int((self.tl.x + self.tr.x) / 2),
                              int((self.tl.y + self.tr.y) / 2))

        # Calculate orientation of tag
        self.forward = math.atan2(self.front.y - self.centre.y, self.front.x - self.centre.x) # Forward vector
        self.angle = math.degrees(self.forward) # Angle between forward vector and x-axis

class Robot:
    def __init__(self, tag, position):
        self.tag = tag
        self.id = tag.id
        self.position = position
        self.orientation = tag.angle
        self.sensor_range = 0.3 # 30cm sensing radius
        self.neighbours = {}


class SensorReading:
    def __init__(self, range, bearing, orientation=0, workers=0):
        self.range = range
        self.bearing = bearing
        self.orientation = orientation
        self.workers = workers


class TimerStatus(Enum):
    STARTED = 0
    COMPLETE = 1


class Timer:
    def __init__(self):
        self.current_time = 0  # Will be updated by robot_client
        self.status = TimerStatus.STARTED
    
    def set_time(self, time_seconds):
        """
        Updates the timer with the time value received from robot_client.
        
        time_seconds -- Time remaining in seconds (float)
        """
        self.current_time = time_seconds

    def set_complete(self, is_complete):
        if is_complete:
            self.status = TimerStatus.COMPLETE
        else:
            self.status = TimerStatus.STARTED
    
    def start(self):
        self.status = TimerStatus.STARTED
    
    def getColor(self):
        if self.status == TimerStatus.COMPLETE:
            return red
        return white

    def getString(self):
        time_string = ""
        seconds = int(self.current_time) % 60
        minutes = int(self.current_time) // 60

        seconds = str(seconds)

        if len(seconds) == 1:
            seconds = "0" + seconds
        time_string = str(minutes) + ":" + seconds
        return time_string


class Tracker(threading.Thread):


    def __init__(self, save_log=False, save_video=False):

        threading.Thread.__init__(self)
        self.daemon = True  # Make thread daemon so it stops when main program exits
        self.stop_event = threading.Event()  # Flag to signal thread to stop
        self.save_log = save_log
        self.save_video = save_video
        self.camera = Camera()
        self.calibrated = False
        self.num_corner_tags = 0
        self.min_x = 0 # In pixels
        self.min_y = 0 # In pixels
        self.max_x = 0 # In pixels
        self.max_y = 0 # In pixels
        self.centre = Vector2D(0, 0) # In metres
        self.corner_distance_metres = 2.06 # Euclidean distance between corner tags in metres
        self.corner_distance_pixels = 0
        self.scale_factor = 0
        self.robots = {}

        self.gameState = 0
        self.timer = Timer()
        self.roboteams = {}

        # listener = keyboard.Listener(
        #     on_press=self.on_press)
        # listener.start()

    # def on_press(self, key):
    #     try:
    #         if key.char == 'l':
    #             pass

    #         if key.char == 'r':
    #             self.timer = Timer()
    #             # self.timer.start()
    #             self.gameState = 1
    #             self.robots = {}

    #     except AttributeError:
    #         # Special keys (Ctrl, Shift, etc.) don't have 'char' attribute - ignore them
    #         pass
        

    """
    processes raw tags and updates self.robots to contain a dictionary of all visible robots and their IDs
    
    tag_ids
    raw_tags -- 
    List reserved_tags -- List of tags the process should skip (E.g. The corner tags and the ball)
    """
    def processArUco(self, tag_ids, raw_tags):
        for id, raw_tag in zip(tag_ids, raw_tags):

            tag = Tag(id, raw_tag)

            if self.calibrated:
                if (tag.id != 0):  # Reserved tag ID for corners
                    position = Vector2D(tag.centre.x / self.scale_factor,
                                        tag.centre.y / self.scale_factor)  # Convert pixel coordinates to metres
                    if (tag.id in self.robots.keys()):
                        self.robots[tag.id].position = position
                        self.robots[id].orientation = tag.angle
                        self.robots[id].tag = tag
                    else:
                        self.robots[id] = Robot(tag, position)

            else:  # Only calibrate the first time two corner tags are detected
                self.calibrate(tag)


    """
    Calibrates the play area ready for a match
    
    tag -- a tag of ID=0
    """
    def calibrate(self, tag):
        if tag.id == 0:  # Reserved tag ID for corners

            if self.num_corner_tags == 0:  # Record the first corner tag detected
                self.min_x = tag.centre.x
                self.max_x = tag.centre.x
                self.min_y = tag.centre.y
                self.max_y = tag.centre.y
            else:  # Set min/max boundaries of arena based on second corner tag detected

                if tag.centre.x < self.min_x:
                    self.min_x = tag.centre.x
                if tag.centre.x > self.max_x:
                    self.max_x = tag.centre.x
                if tag.centre.y < self.min_y:
                    self.min_y = tag.centre.y
                if tag.centre.y > self.max_y:
                    self.max_y = tag.centre.y

                self.corner_distance_pixels = math.dist([self.min_x, self.min_y], [self.max_x,
                                                                                   self.max_y])  # Euclidean distance between corner tags in pixels
                self.scale_factor = self.corner_distance_pixels / self.corner_distance_metres
                x = ((self.max_x - self.min_x) / 2) / self.scale_factor  # Convert to metres
                y = ((self.max_y - self.min_y) / 2) / self.scale_factor  # Convert to metres
                self.centre = Vector2D(x, y)

                self.calibrated = True


            self.num_corner_tags = self.num_corner_tags + 1

    """
        Backend processing for the robots.

        Currently: Builds a map of neighbouring robots.
    """

    def processRobots(self):
        for id, robot in self.robots.items():

            for other_id, other_robot in self.robots.items():

                if id != other_id:  # Don't check this robot against itself

                    range = robot.position.distance_to(other_robot.position)
                    
                    if range < robot.sensor_range:

                        absolute_bearing = math.degrees(math.atan2(other_robot.position.y - robot.position.y,
                                                                other_robot.position.x - robot.position.x))
                        relative_bearing = absolute_bearing - robot.orientation
                        normalised_bearing = angles.normalize(relative_bearing, -180, 180)
                        robot.neighbours[other_id] = SensorReading(range, normalised_bearing, other_robot.orientation)



    """
    Draws bounding box of the arena.
    
    image -- The camera image for the box to be drawn on to. 
    """
    def drawBoundingBox(self, image):
        cv2.rectangle(image, (self.min_x, self.min_y), (self.max_x, self.max_y), green, 1, lineType=cv2.LINE_AA)

    """
    Responsible for drawing minimal robot visualization (just borders and center point).
    
    image -- The camera image for the robots to be drawn onto
    """
    def drawRobotsMinimal(self, image):
        for id, robot in self.robots.items():
            tag = robot.tag

            # Draw border of tag
            cv2.line(image, (tag.tl.x, tag.tl.y), (tag.tr.x, tag.tr.y), green, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (tag.tr.x, tag.tr.y), (tag.br.x, tag.br.y), green, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (tag.br.x, tag.br.y), (tag.bl.x, tag.bl.y), green, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (tag.bl.x, tag.bl.y), (tag.tl.x, tag.tl.y), green, 1, lineType=cv2.LINE_AA)

            # Draw circle on centre point
            cv2.circle(image, (tag.centre.x, tag.centre.y), 5, red, -1, lineType=cv2.LINE_AA)

    """
    Responsible for drawing any UI element associated with the robots.
    
    image -- The camera image for the robots to be drawn onto
    """
    def drawRobots(self, image):
        for id, robot in self.robots.items():

            # Draw tag
            tag = robot.tag

            # Draw border of tag
            cv2.line(image, (tag.tl.x, tag.tl.y), (tag.tr.x, tag.tr.y), green, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (tag.tr.x, tag.tr.y), (tag.br.x, tag.br.y), green, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (tag.br.x, tag.br.y), (tag.bl.x, tag.bl.y), green, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (tag.bl.x, tag.bl.y), (tag.tl.x, tag.tl.y), green, 1, lineType=cv2.LINE_AA)

            # Draw sensing range as circle
            range_radius_px = int(robot.sensor_range * self.scale_factor)
            cv2.circle(image, (tag.centre.x, tag.centre.y), range_radius_px, cyan, 1, lineType=cv2.LINE_AA)

            # Draw circle on centre point
            cv2.circle(image, (tag.centre.x, tag.centre.y), 5, red, -1, lineType=cv2.LINE_AA)

            tag = robot.tag
            # print(f'robot {id} position: {robot.position.x}, {robot.position.y}')

            # Draw line from centre point to front of tag
            forward_point = ((tag.front - tag.centre) * 2) + tag.centre


            # Draw tag ID

            text3 = str(robot.tag.id)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 4
            textsize = cv2.getTextSize(text3, font, font_scale, thickness)[0]
            position2 = (int(tag.centre.x - textsize[0] / 2), int(tag.centre.y + textsize[1] / 2 - 2 * textsize[1]))

            cv2.putText(image, text3, position2, font, font_scale, white, thickness * 3, cv2.LINE_AA)
            cv2.putText(image, text3, position2, font, font_scale, black, thickness, cv2.LINE_AA)
            cv2.line(image, (tag.centre.x, tag.centre.y), (forward_point.x, forward_point.y), black, 10,
                     lineType=cv2.LINE_AA)
            cv2.line(image, (tag.centre.x, tag.centre.y), (forward_point.x, forward_point.y), green, 3,
                     lineType=cv2.LINE_AA)


    """
    Responsible for drawing any UI element associated with the initial position of the robots.
    
    image -- The camera image for the robots to be drawn onto
    """
    def drawInitRobotPositions(self, image):
        global robot_info
        for id, robot in robot_info.items():

            # Skip if robot tag not currently detected
            if robot["id"] not in self.robots:
                continue

            # Get tag
            tag = self.robots[robot["id"]].tag

            # Draw circle on centre point (more transparent)
            overlay = image.copy()
            cx = self.min_x + int(robot["initial_position"]["x"] * self.scale_factor)
            cy = self.min_y + int(robot["initial_position"]["y"] * self.scale_factor)
            cv2.circle(overlay, (cx, cy), 25, red, -1, lineType=cv2.LINE_AA)
            image[:] = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)

            # Draw line from centre point to front of tag
            centre = Vector2D(cx, cy)
            length_m = abs(tag.front - tag.centre) * 2 / self.scale_factor
            angle_rad = robot["initial_orientation"]

            front_x_m = robot["initial_position"]["x"] + length_m * math.cos(angle_rad)
            front_y_m = robot["initial_position"]["y"] + length_m * math.sin(angle_rad)

            front_x_px = int(self.min_x + front_x_m * self.scale_factor)
            front_y_px = int(self.min_y + front_y_m * self.scale_factor)

            forward_point = Vector2D(front_x_px, front_y_px)

            # Draw tag ID

            line_overlay = image.copy()
            cv2.line(line_overlay, (tag.centre.x, tag.centre.y), (centre.x, centre.y), cyan, 3,
                     lineType=cv2.LINE_AA)
            cv2.line(line_overlay, (centre.x, centre.y), (forward_point.x, forward_point.y), black, 10,
                     lineType=cv2.LINE_AA)
            cv2.line(line_overlay, (centre.x, centre.y), (forward_point.x, forward_point.y), green, 3,
                     lineType=cv2.LINE_AA)
            image[:] = cv2.addWeighted(line_overlay, 0.35, image, 0.65, 0)
            

    def drawTargets(self, image):
        global target_info
        for target in target_info:
            # Only draw target if show_target flag is True
            if not target.get("show_target", False):
                continue

            # Draw circle on centre point (more transparent)
            overlay = image.copy()
            cx = self.min_x + int(target["position"]["x"] * self.scale_factor)
            cy = self.min_y + int(target["position"]["y"] * self.scale_factor)
            
            # get radius in pixels
            radius = int(target["radius"] * self.scale_factor)
            
            cv2.circle(overlay, (cx, cy), radius, (191, 255, 191), -1, lineType=cv2.LINE_AA)
            image[:] = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)


    def getRobotsInTargetCount(self):
        global target_info

        total_robots = len(self.robots)
        robots_in_target = 0

        if total_robots > 0 and len(target_info) > 0:
            target = target_info[0]
            target_cx = self.min_x + int(target["position"]["x"] * self.scale_factor)
            target_cy = self.min_y + int(target["position"]["y"] * self.scale_factor)
            target_radius_px = int(target["radius"] * self.scale_factor)

            for robot in self.robots.values():
                tag = robot.tag
                if math.dist([tag.centre.x, tag.centre.y], [target_cx, target_cy]) <= target_radius_px:
                    robots_in_target += 1

        return robots_in_target, total_robots


    def drawRobotsInTargetCount(self, image):
        robots_in_target, total_robots = self.getRobotsInTargetCount()

        text = f"Number of robots in target: {robots_in_target}/{total_robots}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 5
        position = (520, 60)

        cv2.putText(image, text, position, font, font_scale, white, thickness * 3, cv2.LINE_AA)
        cv2.putText(image, text, position, font, font_scale, black, thickness, cv2.LINE_AA)

    def initialize_csv_file(self, config):
        """Initialize CSV file for experiment data logging."""
        global csv_file, csv_writer, csv_initialized, last_logged_simulation_time, max_robots_in_target_ever_seen, arrival_times
        global video_writer, video_initialized, next_video_frame_time
        
        try:
            # Create results directory if it doesn't exist
            results_dir = "results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                print(f"[CSV] Created results directory: {results_dir}")
            
            # Extract parameters from config
            experiment_name = config.get("experiment_name", "experiment")
            num_robots = config.get("num_robots", 0)
            robot_speed = config.get("robot_speed", 0)
            separation_distance = config.get("separation_distance", 0)
            broadcast_duration = config.get("broadcast_duration", 0)
            
            # Get current date and time
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Initialize CSV logging if enabled
            if self.save_log:
                # Create filename: <experiment_name>_R<robots>_S<speed>_D<separation>_B<broadcast>_<date>_<time>.csv
                csv_filename = f"{experiment_name}_R{num_robots}_S{robot_speed}_D{separation_distance}_B{broadcast_duration}_{now}.csv"
                csv_filepath = os.path.join(results_dir, csv_filename)
                
                # Open CSV file for writing
                csv_file = open(csv_filepath, 'w', newline='')
                csv_writer = csv.writer(csv_file)
                
                # Write header row
                header = ['timestamp', 'simulation_time']
                for robot_id in sorted(self.robots.keys()):
                    header.extend([
                        f'robot_{robot_id}_x',
                        f'robot_{robot_id}_y',
                        f'robot_{robot_id}_orientation',
                        f'robot_{robot_id}_in_target'
                    ])
                csv_writer.writerow(header)
                csv_file.flush()
                
                # Reset tracking variables (will start logging once simulation_time > 0)
                last_logged_simulation_time = -1
                max_robots_in_target_ever_seen = 0
                arrival_times = {}
                
                csv_initialized = True
                print(f"[CSV] Initialized CSV file: {csv_filepath}")
            else:
                print(f"[CSV] CSV logging disabled")
            
            # Initialize video recording if enabled
            if self.save_video:
                video_filename = f"{experiment_name}_R{num_robots}_S{robot_speed}_D{separation_distance}_B{broadcast_duration}_{now}.mp4"
                video_filepath = os.path.join(results_dir, video_filename)
                
                # Video codec and properties (1280x720 resolution at 30 fps)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30
                frame_size = (1280, 720)
                
                video_writer = cv2.VideoWriter(video_filepath, fourcc, fps, frame_size)
                
                if video_writer.isOpened():
                    video_initialized = True
                    next_video_frame_time = time.monotonic()
                    print(f"[VIDEO] Initialized video file: {video_filepath}")
                else:
                    print(f"[VIDEO] Error: Could not initialize video writer")
            else:
                print(f"[VIDEO] Video recording disabled")
            
        except Exception as e:
            print(f"[CSV] Error initializing CSV file: {type(e).__name__}: {e}")

    def log_robot_data(self, simulation_time):
        """Log current robot positions and states to CSV file."""
        global csv_file, csv_writer, last_logged_simulation_time, max_robots_in_target_ever_seen, arrival_times
        
        if not csv_initialized or csv_file is None or csv_writer is None:
            return
        
        # Only log if this is a different simulation timestep than the last logged one
        # This ensures we log once per simulated timestep, not once per camera frame
        if simulation_time == last_logged_simulation_time:
            return
        
        # Only log if simulation has actually started (simulation_time > 0)
        if simulation_time <= 0:
            return
        
        try:
            # Get number of robots in target
            robots_in_target, _ = self.getRobotsInTargetCount()
            
            # Check if this is a new maximum number of robots in target
            if robots_in_target > max_robots_in_target_ever_seen:
                max_robots_in_target_ever_seen = robots_in_target
                arrival_times[robots_in_target] = round(simulation_time, 1)
                print(f"[CSV] ARRIVAL_{robots_in_target}: {round(simulation_time, 1)}s")
            
            # Prepare row data with rounded simulation_time to 1 decimal place
            row = [time.time(), round(simulation_time, 1)]
            
            # Add robot data
            for robot_id in sorted(self.robots.keys()):
                if robot_id in self.robots:
                    robot = self.robots[robot_id]
                    
                    # Check if robot is in target
                    is_in_target = 0
                    if len(target_info) > 0:
                        target = target_info[0]
                        target_cx = self.min_x + int(target["position"]["x"] * self.scale_factor)
                        target_cy = self.min_y + int(target["position"]["y"] * self.scale_factor)
                        target_radius_px = int(target["radius"] * self.scale_factor)
                        tag = robot.tag
                        if math.dist([tag.centre.x, tag.centre.y], [target_cx, target_cy]) <= target_radius_px:
                            is_in_target = 1
                    
                    row.extend([
                        round(robot.position.x, 4),
                        round(robot.position.y, 4),
                        round(robot.orientation, 4),
                        is_in_target
                    ])
            
            # Write row to CSV
            csv_writer.writerow(row)
            csv_file.flush()
            
            # Update last logged simulation time
            last_logged_simulation_time = simulation_time
            
        except Exception as e:
            print(f"[CSV] Error logging robot data: {type(e).__name__}: {e}")

    def close_csv_file(self):
        """Close the CSV file cleanly and release video writer."""
        global csv_file, csv_writer, csv_initialized, last_logged_simulation_time, max_robots_in_target_ever_seen, arrival_times
        global video_writer, video_initialized, next_video_frame_time
        
        try:
            if csv_file is not None:
                # Write a blank row separator
                csv_writer.writerow([])
                csv_writer.writerow(['ARRIVAL_TIMES'])
                
                # Write arrival times (only the times, no labels)
                for num_robots in sorted(arrival_times.keys()):
                    sim_time = arrival_times[num_robots]
                    csv_writer.writerow([sim_time])
                
                csv_file.flush()
                csv_file.close()
                print(f"[CSV] CSV file closed successfully")
                print(f"[CSV] Arrival times: {arrival_times}")
                csv_file = None
                csv_writer = None
                csv_initialized = False
                last_logged_simulation_time = -1
                max_robots_in_target_ever_seen = 0
                arrival_times = {}
            
            # Release video writer (thread-safe)
            with video_lock:
                video_initialized = False
                next_video_frame_time = 0.0
                if video_writer is not None:
                    video_writer.release()
                    print(f"[VIDEO] Video writer released successfully")
                    video_writer = None
        except Exception as e:
            print(f"[CSV/VIDEO] Error closing files: {type(e).__name__}: {e}")


    def run(self):
        
        global robot_info, target_info, next_video_frame_time
        
        while not self.stop_event.is_set():
            image = self.camera.get_frame()
            overlay = image.copy()
            
            aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
            aruco_parameters = cv2.aruco.DetectorParameters()
            
            (raw_tags, tag_ids, rejected) = cv2.aruco.detectMarkers(image, aruco_dictionary, parameters=aruco_parameters)

            self.robots = {} # Clear dictionary every frame in case robots have disappeared

            # Check whether any tags were detected in this camera frame
            if tag_ids is not None and len(tag_ids.tolist()) > 0:

                tag_ids = list(itertools.chain(*tag_ids))
                tag_ids = [int(id) for id in tag_ids] # Convert from numpy.int32 to int

                # Process raw ArUco output
                self.processArUco(tag_ids, raw_tags)

                # Process and draw robots
                self.processRobots()

                # Log robot data to CSV (if initialized)
                self.log_robot_data(self.timer.current_time)

                # Draw boundary of virtual environment based on corner tag positions
                self.drawBoundingBox(image)

                # Draw targets first so they stay behind other overlays
                if len(target_info) > 0:
                    self.drawTargets(image)

                # Now add all UI elements (labels, orientation, sensing ranges, text)
                self.drawRobots(image)

                self.drawRobotsInTargetCount(image)

                text = f"Time: {self.timer.getString()}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2
                thickness = 5
                textsize = cv2.getTextSize(text, font, font_scale, thickness)[0]
                position = (20, 60)
                cv2.putText(image, text, position, font, font_scale, self.timer.getColor(), thickness * 3, cv2.LINE_AA)
                cv2.putText(image, text, position, font, font_scale, black, thickness, cv2.LINE_AA)
                
                # Now draw the overlays (init positions and targets)
                current_time = time.time()
                if len(robot_info) > 0 and (current_time - last_robot_info_update) < INIT_POSITION_TIMEOUT:
                    self.drawInitRobotPositions(image)
                elif len(robot_info) > 0 and (current_time - last_robot_info_update) >= INIT_POSITION_TIMEOUT:
                    # Clear robot_info if we haven't received an update in a while
                    robot_info = {}

                # Save image with overlays applied (or use current overlay blend)
                alpha = 0.3
                image_with_overlay = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                
                # Record to video file at a stable 30fps (thread-safe)
                if self.timer.current_time > 0:
                    now = time.monotonic()
                    with video_lock:
                        if video_initialized and video_writer is not None:
                            if now >= next_video_frame_time:
                                frame = cv2.resize(image_with_overlay, (1280, 720))
                                video_writer.write(frame)
                                frame_period = 1.0 / 30.0
                                if next_video_frame_time == 0.0:
                                    next_video_frame_time = now + frame_period
                                else:
                                    while next_video_frame_time <= now:
                                        next_video_frame_time += frame_period

            window_name = 'SwarmHack'

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
            cv2.imshow(window_name, image_with_overlay)

            # Check for 'q' key to quit
            if cv2.waitKey(1) == ord('q'):
                print("Quit key pressed. Shutting down...")
                self.stop_event.set()
                break
        
        cv2.destroyAllWindows()

async def handler(websocket):
    try:
        async for packet in websocket:
            message = json.loads(packet)
            
            # Process any requests received
            reply = {}
            send_reply = False
            if tracker.calibrated:
                if "check_awake" in message:
                    reply["awake"] = True
                    send_reply = True

                if "get_arena_limits" in message:
                    min_x_m = tracker.min_x / tracker.scale_factor
                    min_y_m = tracker.min_y / tracker.scale_factor
                    max_x_m = tracker.max_x / tracker.scale_factor
                    max_y_m = tracker.max_y / tracker.scale_factor
                    print(
                        "tracker arena limits (meters): "
                        f"min_x={round(min_x_m, 2)}, min_y={round(min_y_m, 2)}, "
                        f"max_x={round(max_x_m, 2)}, max_y={round(max_y_m, 2)}"
                    )
                    reply["arena_limits"] = {
                        "min_x": round(min_x_m, 2),
                        "min_y": round(min_y_m, 2),
                        "max_x": round(max_x_m, 2),
                        "max_y": round(max_y_m, 2),
                    }
                    send_reply = True

                if "get_robots" in message:
                    send_reply = True
                    for id, robot in tracker.robots.items():

                        reply[id] = {}
                        reply[id]["position"] = {"x": round(robot.position.x, 2), "y": round(robot.position.y, 2)}
                        reply[id]["orientation"] = round(robot.orientation, 2)
                        reply[id]["players"] = {}

                        for neighbour_id, neighbour in robot.neighbours.items():

                            neighbour_robot = tracker.robots[neighbour_id]
                            reply[id]["players"][neighbour_id] = {}
                            reply[id]["players"][neighbour_id]["range"] = round(neighbour.range, 2)
                            reply[id]["players"][neighbour_id]["bearing"] = round(neighbour.bearing, 2)
                            reply[id]["players"][neighbour_id]["orientation"] = round(neighbour.orientation, 2)

                if "get_in_target" in message:
                    robots_in_target, _ = tracker.getRobotsInTargetCount()
                    reply["get_in_target"] = {
                        "robots_in_target": robots_in_target,
                    }
                    send_reply = True

                if "targets" in message:
                    global target_info
                    target_info = message["targets"]

                if "robots" in message:
                    global robot_info, last_robot_info_update
                    robot_info = message["robots"]
                    last_robot_info_update = time.time()
                
                if "experiment_config" in message:
                    global experiment_config
                    experiment_config = message["experiment_config"]
                    print(f"[CSV] Received experiment config: {experiment_config}")
                    # Initialize CSV file with the experiment configuration
                    tracker.initialize_csv_file(experiment_config)
                
                if "simulation_time" in message:
                    tracker.timer.set_time(message["simulation_time"])

                if "experiment_finished" in message:
                    tracker.timer.set_complete(message["experiment_finished"])
                    # Close CSV file as soon as experiment finishes
                    if message["experiment_finished"]:
                        tracker.close_csv_file()

                # Send reply, if requested
                if send_reply:
                    await websocket.send(json.dumps(reply))
    except websockets.exceptions.ConnectionClosedError:
        # Connection closed by client - this is normal, especially after experiment_finished
        pass
    except Exception as e:
        print(f"[HANDLER] Unexpected error in connection handler: {type(e).__name__}: {e}")


async def main(save_log=False, save_video=False):
    global tracker
    print("Initializing Tracker...")
    tracker = Tracker(save_log=save_log, save_video=save_video)
    tracker.start()

    print("Starting WebSocket server on port 6001...")
    
    # In newer websockets versions, we use the server as an async context manager.
    try:
        async with websockets.serve(handler, host="0.0.0.0", port=6001):
            # Wait for tracker thread to signal stop event
            while not tracker.stop_event.is_set():
                await asyncio.sleep(0.1)
            print("Tracker stopped. Shutting down server...")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        # Ensure tracker stops
        tracker.stop_event.set()
        # Close CSV file
        tracker.close_csv_file()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Robot tracking server")
    parser.add_argument("--save-log", action="store_true", help="Enable CSV logging (default: False)")
    parser.add_argument("--save-video", action="store_true", help="Enable video recording (default: False)")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(save_log=args.save_log, save_video=args.save_video))
    except KeyboardInterrupt:
        print("\nServer shut down by user.")