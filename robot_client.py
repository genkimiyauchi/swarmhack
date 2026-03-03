#!/usr/bin/env python3

from robots import robots, server_none, server_sheffield

import asyncio
import websockets
import json
import signal
import time
import sys
import threading
import atexit
from enum import Enum
import time
import random
import inspect
from vector2d import Vector2D
import math
import pprint
import angles
import colorama
from colorama import Fore
import xml.etree.ElementTree as ET
import traceback

# Cross-platform keyboard handling
if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

from controller import Robot

ITERATION_TIME = 0.1   # Time to sleep between each iteration (default: 0.1)
ROBOT_CONFIG = None # This will be set to the <robot_controller> tag in the experiment XML file, which contains any parameters you set for your controller in that file. See the example XML files for how to set this up.
ROBOTS = [] # list of robot names
ROBOT_INIT_POS = {} # key: robot name, value: initial position vector2d(x,y)
ROBOT_INIT_ANGLE = {} # key: robot name, value: initial orientation in radians
TARGET_POS = None
TARGET_RADIUS = None
ARENA_LIMITS = {}

"""
This function is the main loop of your application. You can make any changes you want throughout this 
file, but most of your game logic will be located in here.

First, ensure that the robot_ids list below is correctly set to the robots you wish to work with.
For example:
    robot_ids = [34, 37, 39]

When run, this script will connect to the server and then repeatedly call the main_loop() function.
The script communicates with the robots using a websockets connection. Sending data requires that you 
use Python's asynchronous I/O. If you are not familiar with this, the rule is to remember that the control
function should be declared with "async" (see the simple_obstacle_avoidance() example) and called from 
main_loop() using loop.run_until_complete(async_thing_to_run(ids))
"""

robot_ids = ROBOTS

def main_loop():
    global simulation_time
    
    # This requests all virtual sensor data from the tracking server for the robots specified in robot_ids
    # This is stored in the global variable active_robots, a map of id -> instances of the Robot class (defined lower in this file) 
    if experiment_running:
        print(Fore.GREEN + "[INFO]: Requesting data from tracking server")
    loop.run_until_complete(get_server_data())

    # Request sensor data from detected robots
    # This augments the Robot instances with their battery level and the values from each robot's proximity sensors
    # You only need to do this if you care about their battery level, or are using their proximity sensors
    if experiment_running:
        print(Fore.GREEN + "[INFO]: Robots detected:", ids)
        print(Fore.GREEN + "[INFO]: Requesting data from detected robots")
    loop.run_until_complete(get_robot_data(ids))

    # Exchange messages with neighbouring robots
    if experiment_running:
        print(Fore.GREEN + "[INFO]: Collecting messages from neighbouring robots")
    local_communication()

    # Now run our behaviour
    if experiment_running:
        print(Fore.GREEN + "[INFO]: Sending commands to detected robots")
    loop.run_until_complete(send_robot_commands(ids))

    # Send experiment info to server
    if experiment_running:
        print(Fore.GREEN + "[INFO]: Sending data to tracking server")
    loop.run_until_complete(send_experiment_info())

    if experiment_running:
        print()

    # Increment simulation time if experiment is running
    if experiment_running:
        simulation_time += ITERATION_TIME

    # Sleep until next control cycle. We use 0.1 seconds by default so as to not flood the network.
    time.sleep(ITERATION_TIME)


def load_configuration(xml_path='experiments/practice1.xml'):
    # Parse the experiment configurations
    global ITERATION_TIME, ROBOT_CONFIG, ROBOTS, ROBOT_INIT_POS, TARGET_POS, TARGET_RADIUS

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for child in root:
        if child.tag == 'framework':
            ticks_per_second = float(child[1].get('ticks_per_second'))
            ITERATION_TIME = 1 / ticks_per_second
            
        elif child.tag == 'controllers':
            for controller in child:
                if controller.tag == 'robot_controller':
                    ROBOT_CONFIG = controller
           
        elif child.tag == 'arena':
            for entity in child:
                if entity.tag == 'target':
                    pos_str = entity.get("position")
                    x_str, y_str, *_ = pos_str.split(",")
                    TARGET_POS = Vector2D(float(x_str), float(y_str))
                    TARGET_RADIUS = float(entity.get('radius'))
                if entity.tag == 'e-puck':
                    name = int(entity.get('id'))
                    ROBOTS.append(name)
                    pos_str = entity.get("position")
                    x_str, y_str, *_ = pos_str.split(",")
                    ROBOT_INIT_POS[name] = Vector2D(float(x_str), float(y_str))
                    ROBOT_INIT_ANGLE[name] = float(entity.get('orientation')[2]) # Orientation is given as x,y,z euler angles, but we only care about the z angle (rotation around vertical axis)

    print(f'ROBOTS: {ROBOTS}')
    print(f'ROBOT_INIT_POS: {ROBOT_INIT_POS}')
    print(f'ROBOT_INIT_ANGLE: {ROBOT_INIT_ANGLE}')
    print(f'TARGET_POS: {TARGET_POS}')
    print(f'TARGET_RADIUS: {TARGET_RADIUS}')

"""
This is an example of a behaviour. You will want to replace this with a behaviour that implements your team
movements. It currently is an example of basic object avoidance.
This function is called for each robot that we listed in robot_ids that we are interested in.
"""
async def send_commands(robot):
    """
    robot.neighbours is a map of all the other robots (i.e. not this one) 
    with their role, team, range from this robot, and bearing from this robot
    For example:
    { '33': { 'bearing': -178.98,
              'orientation': -16.26,
              'range': 1.14,
              'role': 'NOMAD',
              'team': 'UNASSIGNED'},
      '34': { 'bearing': 177.24,
              'orientation': -25.56,
              'range': 1.47,
              'role': 'DEFENDER',
              'team': 'BLUE'},

    Print it with pprint.PrettyPrinter(indent=2).pprint(robot.neighbours)
    """

    try:
        # Turn off LEDs and motors when killed. Please do this!
        if kill_now():
            message = {"set_leds_colour": "off", "set_motor_speeds": {}}
            message["set_motor_speeds"]["left"] = 0
            message["set_motor_speeds"]["right"] = 0
            await robot.connection.send(json.dumps(message))

        message = {}

        # During initialization phase, use simple movement toward init position
        if initializing:
            try:
                robot.initialization_step()
            except Exception as e:
                print(Fore.LIGHTRED_EX + f'============= Error during initialization =============')
                traceback.print_exc()
        # During experiment, use full control_step with state machine
        elif experiment_running:
            try:
                robot.control_step()
            except Exception as e:
                print(Fore.LIGHTRED_EX + f'============= Error occurred =============')
                traceback.print_exc()

        """
        Construct a command message
        Robots are controlled by sending them a JSON dictionary which we create here using the message variable
        
        We can set the speed of the wheel motors (from -100 to 100). Setting them to the same value makes the robot go forwards
        or backwards. Setting them differently makes the robot turn. For example:
        message["set_motor_speeds"]["left"] = 100
        message["set_motor_speeds"]["right"] = 100
    
        We can also set the colour of the LED. Supported colours are "off", "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"
        message["set_leds_colour"] = "green"

        You can combine commands (i.e. setting both wheels and the LED colour in one go)
        """
        # Determine motor speeds based on initialization/experiment state and teleop mode
        if robot.teleop and experiment_running:
            # Teleop mode (only if experiment is running) - use teleop commands
            
            # Check if we should timeout and revert to forward motion
            # If no command received in 0.3 seconds, go forward automatically
            if (robot.teleop_last_command in ["left", "right"] and 
                time.time() - robot.teleop_last_command_time > 0.3):
                # Timeout - revert to forward motion
                robot.teleop_left = 800
                robot.teleop_right = 800
            
            left = robot.teleop_left
            right = robot.teleop_right
        elif initializing or experiment_running:
            # Initialization or experiment running - use autonomous control
            left = robot.left
            right = robot.right
        else:
            # Experiment not yet started - stop motors
            left = 0
            right = 0

        message["set_motor_speeds"] = {}
        message["set_motor_speeds"]["left"] = left
        message["set_motor_speeds"]["right"] = right
        message["set_leds_colour"] = robot.led_colour
        
        if experiment_running:
            print(f'Robot {robot.id}: Sending to motors - left={left:.0f}, right={right:.0f}')

        # Send command message
        try:
            await robot.connection.send(json.dumps(message))
        except Exception as e:
            print(f"send_commands: {type(e).__name__}: {e}")
            # Attempt to reconnect
            try:
                ip = robots[robot.id]
                uri = f"ws://{ip}:{robot_port}"
                print(f"Attempting to reconnect to robot {robot.id} at {uri}")
                robot.connection = await websockets.connect(uri)
                awake = await check_awake(robot.connection)
                if awake:
                    print(f"Robot {robot.id} reconnected successfully")
                    # Retry sending the command
                    await robot.connection.send(json.dumps(message))
                else:
                    print(f"Robot {robot.id} reconnection check failed")
            except Exception as reconnect_error:
                print(f"Failed to reconnect to robot {robot.id}: {type(reconnect_error).__name__}: {reconnect_error}")

    except Exception as e:
        print(f"send_commands: {type(e).__name__}: {e}")


#-----------------------------------------------------------------
# You probably don't need to change anything below here
#-----------------------------------------------------------------


active_robots = {} 
ids = []


# Server address, port details, globals
#---------------------------------------
server_address = server_sheffield
server_port = 6001
robot_port = 80

if len(server_address) == 0:
    raise Exception(f"Enter local tracking server address on line {inspect.currentframe().f_lineno - 6}, "
                    f"then re-run this script.")

server_connection = None
teleop_enabled = True  # Set to False to disable teleop integration
teleop_robot_id = None  # Currently controlled robot ID
initializing = True  # Robots move to init positions before experiment
experiment_running = False  # Set to True to start the experiment
simulation_time = 0.0  # Simulation time in seconds, starts when experiment begins
colorama.init(autoreset=True)

_stdin_fd = None
_saved_terminal_settings = None


# Cross-platform keyboard reading functions
def configure_terminal_input_mode():
    """Put terminal into cbreak mode with echo enabled for keyboard listening (Linux/WSL)."""
    global _stdin_fd, _saved_terminal_settings
    if sys.platform != 'win32' and _saved_terminal_settings is None:
        _stdin_fd = sys.stdin.fileno()
        _saved_terminal_settings = termios.tcgetattr(_stdin_fd)
        tty.setcbreak(_stdin_fd)
        # Enable echo while in cbreak mode
        attrs = termios.tcgetattr(_stdin_fd)
        attrs[3] |= termios.ECHO  # c_lflag (index 3) - enable ECHO flag
        termios.tcsetattr(_stdin_fd, termios.TCSADRAIN, attrs)


def restore_terminal_input_mode():
    """Restore terminal settings so shell echo/input behave normally after exit."""
    global _stdin_fd, _saved_terminal_settings
    if sys.platform != 'win32' and _saved_terminal_settings is not None:
        try:
            termios.tcsetattr(_stdin_fd, termios.TCSADRAIN, _saved_terminal_settings)
        except Exception:
            pass
        finally:
            _stdin_fd = None
            _saved_terminal_settings = None


# Always restore terminal when process exits
atexit.register(restore_terminal_input_mode)


def getKey():
    """Get a single keypress from the terminal"""
    if sys.platform == 'win32':
        return msvcrt.getwch()
    else:
        return sys.stdin.read(1)


# Background task to listen for keyboard input for experiment start and teleop
def start_keyboard_listener():
    """Start a background thread to listen for keyboard input"""
    
    def keyboard_thread():
        global experiment_running, teleop_robot_id, initializing
        valid_robots = sorted(active_robots.keys())

        configure_terminal_input_mode()
        
        print(Fore.CYAN + "\n" + "="*70)
        print(Fore.CYAN + "KEYBOARD CONTROLS:")
        if initializing:
            print(Fore.CYAN + "  Robots are initializing to their starting positions...")
            print(Fore.CYAN + "  Press 's' to start experiment (after robots reach init positions)")
        else:
            print(Fore.CYAN + "  Press 's' to start experiment")
        if teleop_enabled:
            print(Fore.CYAN + f"  Available robot IDs: {valid_robots}")
            print(Fore.CYAN + "  Press digits (e.g. 1, 2, 20) then Enter to select robot")
            print(Fore.CYAN + "  Press 'a' to turn left (while controlling a robot)")
            print(Fore.CYAN + "  Press 'd' to turn right (while controlling a robot)")
            print(Fore.CYAN + "  Press 'q' to release robot control")
        print(Fore.CYAN + "  Press Ctrl+C to exit")
        print(Fore.CYAN + "="*70 + "\n")

        digit_buffer = ""

        def select_robot(robot_id):
            global teleop_robot_id
            if robot_id in active_robots:
                teleop_robot_id = robot_id
                robot = active_robots[robot_id]
                robot.teleop = True
                robot.teleop_left = 800
                robot.teleop_right = 800
                state = "ACTIVE" if experiment_running else "PENDING START"
                print(Fore.GREEN + f"\n[TELEOP] Controlling robot {robot_id} ({state}, press 'q' to release)\n")
            else:
                print(Fore.YELLOW + f"\n[TELEOP] Robot {robot_id} not connected. Available: {valid_robots}\n")
        
        try:
            while not __kill_now:
                try:
                    key = getKey()

                    # In some terminals Ctrl+C may still appear as a character
                    if key == '\x03':
                        signal.raise_signal(signal.SIGINT)
                        continue

                    # Build robot ID from digits, commit on Enter
                    if key.isdigit():
                        digit_buffer += key
                        continue
                    if key in ('\r', '\n') and digit_buffer:
                        robot_id = int(digit_buffer)
                        digit_buffer = ""
                        select_robot(robot_id)
                        continue
                
                    # Start experiment
                    if key.lower() == 's' and initializing and not experiment_running:
                        initializing = False
                        experiment_running = True
                        global simulation_time
                        simulation_time = 0.0  # Reset simulation time when experiment starts
                        # Reset robots to initial state and main experiment target
                        for robot_id in active_robots:
                            active_robots[robot_id].target = TARGET_POS
                            active_robots[robot_id].target_radius = TARGET_RADIUS
                        print(Fore.YELLOW + "\n[EXPERIMENT STARTED] - Main experiment is now active\n")
                    elif key.lower() == 's' and not initializing and not experiment_running:
                        experiment_running = True
                        print(Fore.YELLOW + "\n[EXPERIMENT STARTED] - Robots are now active\n")

                    # Teleop controls
                    elif teleop_enabled:
                        # Release control
                        if key.lower() == 'q' and teleop_robot_id is not None:
                            robot = active_robots[teleop_robot_id]
                            robot.teleop = False
                            robot.teleop_left = 0
                            robot.teleop_right = 0
                            print(Fore.YELLOW + f"\n[TELEOP] Released control of robot {teleop_robot_id}\n")
                            teleop_robot_id = None
                        
                        # Turn left
                        elif key.lower() == 'a' and teleop_robot_id is not None:
                            robot = active_robots[teleop_robot_id]
                            if robot.teleop:
                                robot.teleop_last_command = "left"
                                robot.teleop_last_command_time = time.time()
                                robot.teleop_left = -600
                                robot.teleop_right = 600
                        
                        # Turn right
                        elif key.lower() == 'd' and teleop_robot_id is not None:
                            robot = active_robots[teleop_robot_id]
                            if robot.teleop:
                                robot.teleop_last_command = "right"
                                robot.teleop_last_command_time = time.time()
                                robot.teleop_left = 600
                                robot.teleop_right = -600
                    
                except Exception as e:
                    if not __kill_now:
                        print(Fore.YELLOW + f"[WARNING]: Keyboard listener error: {type(e).__name__}: {e}")
                        time.sleep(0.05)
        finally:
            restore_terminal_input_mode()
    
    thread = threading.Thread(target=keyboard_thread, daemon=True)
    thread.start()


# Handle Ctrl+C termination
# https://stackoverflow.com/questions/2148888/python-trap-all-signals
#---------------------------------------------------------------------
SIGNALS_TO_NAMES_DICT = dict((getattr(signal, n), n) \
    for n in dir(signal) if n.startswith('SIG') and '_' not in n)
# https://github.com/aaugustin/websockets/issues/124
__kill_now = False

def __set_kill_now(signum, frame):
    print('\nReceived signal:', SIGNALS_TO_NAMES_DICT[signum], str(signum))
    global __kill_now
    __kill_now = True
    restore_terminal_input_mode()

signal.signal(signal.SIGINT, __set_kill_now)
signal.signal(signal.SIGTERM, __set_kill_now)

def kill_now() -> bool:
    global __kill_now
    return __kill_now


# Connect to websocket server of tracking server
async def connect_to_server():
    uri = f"ws://{server_address}:{server_port}"
    connection = await websockets.connect(uri)

    print("Opening connection to server: " + uri)

    awake = await check_awake(connection)

    if awake:
        print("Server is awake")
        global server_connection
        server_connection = connection
    else:
        print("Server did not respond")


# Connect to websocket server running on each of the robots
async def connect_to_robots():
    for id in active_robots.keys():
        ip = robots[id]
        if ip != '':
            uri = f"ws://{ip}:{robot_port}"
            connection = await websockets.connect(uri)

            print("Opening connection to robot:", uri)

            awake = await check_awake(connection)

            if awake:
                print(f"Robot {id} is awake")
                active_robots[id].connection = connection
            else:
                print(f"Robot {id} did not respond")
        else:
            print(f"No IP defined for robot {id}")


# Check if robot is awake by sending the "check_awake" command to its websocket server
async def check_awake(connection):
    awake = False

    try:
        message = {"check_awake": True}

        # Send request for data and wait for reply
        await connection.send(json.dumps(message))
        reply_json = await connection.recv()
        reply = json.loads(reply_json)

        # Reply should contain "awake" with value True
        awake = reply["awake"]

    except Exception as e:
        print(f"{type(e).__name__}: {e}")

    return awake


# Ask a list of robot IDs for all their sensor data (proximity + battery)
async def get_robot_data(ids):
    await message_robots(ids, get_data)


# Send all commands to a list of robots IDs (motors + LEDs)
async def send_robot_commands(ids):
    await message_robots(ids, send_commands)


# Tell a list of robot IDs to stop
async def stop_robots(ids):
    await message_robots(ids, stop_robot)


# Send a message to a list of robot IDs
# Uses multiple websockets code from:
# https://stackoverflow.com/questions/49858021/listen-to-multiple-socket-with-websockets-and-asyncio
async def message_robots(ids, function):
    loop = asyncio.get_event_loop()
    tasks = []
    for id, robot in active_robots.items():
        if id in ids:
            tasks.append(loop.create_task(function(robot)))
    await asyncio.gather(*tasks)


async def get_arena_limits():
    try:
        global ARENA_LIMITS
        message = {"get_arena_limits": True}
        
        # Send request for data and wait for reply
        await server_connection.send(json.dumps(message))
        reply_json = await server_connection.recv()
        reply = json.loads(reply_json)

        ARENA_LIMITS = reply["arena_limits"]
        
    except Exception as e:
        print(f"get_server_data: {type(e).__name__}: {e}")


# Get robots' virtual sensor data from the tracking server, for our active robots
async def get_server_data():
    try:
        global ids
        message = {"get_robots": True}

        # Send request for data and wait for reply
        await server_connection.send(json.dumps(message))
        reply_json = await server_connection.recv()
        reply = json.loads(reply_json)

        # Filter reply from the server, based on our active robots of interest
        filtered_reply = {int(k): v for (k, v) in reply.items() if int(k) in active_robots.keys()}
        ids = list(filtered_reply.keys())

        #pprint.PrettyPrinter(indent=4).pprint(reply)
        #print(f"active_robots.keys() = {active_robots.keys()}")
        #print(f"filtered_reply = {filtered_reply}")
        #print(f"ids = {ids}")

        # Receive robot virtual sensor data from the server
        for id, robot in filtered_reply.items():
            #print(f"Updating robot {id}")
            active_robots[id].position = Vector2D(robot["position"]["x"], robot["position"]["y"])
            active_robots[id].orientation = robot["orientation"]
            active_robots[id].neighbours = robot["players"]
            active_robots[id].progress_through_zone = robot["progress_through_zone"]    

    except Exception as e:
        print(f"get_server_data: {type(e).__name__}: {e}")


# Stop robot from moving and turn off its LEDs
async def stop_robot(robot):
    try:
        # Turn off LEDs and motors when killed
        message = {"set_leds_colour": "off", "set_motor_speeds": {}}
        message["set_motor_speeds"]["left"] = 0
        message["set_motor_speeds"]["right"] = 0
        await robot.connection.send(json.dumps(message))

        # Send command message
        await robot.connection.send(json.dumps(message))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")


# Get IR and battery readings from robot
async def get_data(robot):
    try:
        message = {"get_battery": True}

        # Send request for data and wait for reply
        await robot.connection.send(json.dumps(message))
        reply_json = await robot.connection.recv()
        reply = json.loads(reply_json)

        robot.battery_voltage = reply["battery"]["voltage"]
        robot.battery_percentage = reply["battery"]["percentage"]

    except Exception as e:
        print(f"{type(e).__name__}: {e}")


# Send experiment info to the server to be visualised
async def send_experiment_info():

    global ROBOTS, ROBOT_INIT_POS, ROBOT_INIT_ANGLE, TARGET_POS, TARGET_RADIUS, simulation_time

    message = {"robots": {}, "targets": []}
    
    # Send init robot positions only during initialization phase
    if initializing:
        # send init robot position ROBOT_INIT_POS and orientation ROBOT_INIT_ANGLE to the server for visualisation
        for id in ROBOTS:
            robot_info = {
                "id": id,
                "initial_position": {"x": ROBOT_INIT_POS[id].x, "y": ROBOT_INIT_POS[id].y},
                "initial_orientation": ROBOT_INIT_ANGLE[id]
            }
            message["robots"][id] = robot_info

    # send target position and radius to the server for visualisation
    target_info = {
        "position": {"x": TARGET_POS.x, "y": TARGET_POS.y},
        "radius": TARGET_RADIUS
    }
    message["targets"].append(target_info)

    # Include simulation time from robot_client
    message["simulation_time"] = simulation_time

    # Send the experiment info to the server
    await server_connection.send(json.dumps(message))


# Exchange messages between robots that are within their communication ranges
def local_communication():
    for id in active_robots.keys():
        for other_id in active_robots[id].neighbours:
            active_robots[id].messages[int(other_id)] = active_robots[int(other_id)].msg


# Main entry point for robot control client sample code
if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    loop.run_until_complete(connect_to_server())

    if server_connection is None:
        print(Fore.RED + "[ERROR]: No connection to server")
        sys.exit(1)

    loop.run_until_complete(get_arena_limits())
    print(f"Arena Limits: {ARENA_LIMITS}")

    # Parse experiment configurations
    config_file = "experiments/practice1.xml"
    load_configuration(config_file)

    assert len(robot_ids) > 0

    # Load robot parameters
    Robot.load_params(ROBOT_CONFIG)

    # Create Robot objects
    print(Fore.GREEN + "[INFO]: Creating Robot objects")
    for robot_id in robot_ids:
        if robots[robot_id] != '':
            active_robots[robot_id] = Robot(robot_id)
            active_robots[robot_id].arena_limits = ARENA_LIMITS
            active_robots[robot_id].target = TARGET_POS
            active_robots[robot_id].target_radius = TARGET_RADIUS
            
            # Set initialization target to the robot's init position from XML
            if robot_id in ROBOT_INIT_POS:
                init_pos = ROBOT_INIT_POS[robot_id]
                # Keep explicit global init target so offset is handled consistently while moving
                active_robots[robot_id].init_target = Vector2D(init_pos.x, init_pos.y)
                # Set target orientation if available
                if robot_id in ROBOT_INIT_ANGLE:
                    active_robots[robot_id].init_angle = ROBOT_INIT_ANGLE[robot_id]
                    print(f"Initialised robot {robot_id} - init target: ({init_pos.x:.2f}, {init_pos.y:.2f}), angle: {math.degrees(ROBOT_INIT_ANGLE[robot_id]):.1f}°")
                else:
                    print(f"Initialised robot {robot_id} - init target: ({init_pos.x:.2f}, {init_pos.y:.2f})")
            else:
                print(f"Initialised {robot_id}")
        else:
            print(f"No IP defined for robot {robot_id}")

    # Create websockets connections to robots
    print(Fore.GREEN + "[INFO]: Connecting to robots")
    loop.run_until_complete(connect_to_robots())

    if not active_robots:
        print(Fore.RED + "[ERROR]: No connection to robots")
        sys.exit(1)

    # Set initial targets for robot initialization phase
    for robot_id in active_robots:
        if active_robots[robot_id].init_target is not None:
            active_robots[robot_id].target = active_robots[robot_id].init_target
            print(Fore.GREEN + f"[INIT] Robot {robot_id} will move to initial position")

    # Start keyboard listener for experiment control and teleop
    print(Fore.YELLOW + "\n[READY] All robots connected - Initialization phase starting")
    start_keyboard_listener()
    # Only communicate with robots that were successfully connected to
    while True:
        main_loop()

        if kill_now():
            loop.run_until_complete(stop_robots(robot_ids))  # Kill all robots, even if not visible
            break
