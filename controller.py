import math
from enum import Enum
from simple_pid import PID
from vector2d import Vector2D
from message import Message
from colorama import Fore
import random

TICK_DURATION = 0.1 # Duration of each tick in seconds

INTERWHEEL_DISTANCE = 0.053 # Distance between the two wheels of the e-puck in meters
HALF_INTERWHEEL_DISTANCE = INTERWHEEL_DISTANCE / 2

class State(Enum):
    RANDOM_WALK = 0  # Random walk
    BROADCAST_WALK = 1  # Random walk with broadcast
    BROADCAST_HOMING = 2  # Move to target with broadcast
    IN_TARGET = 3  # In target with broadcast


# Get vector from range and bearing
def get_vector(robot):
    x = math.cos(math.radians(robot['bearing'])) * robot['range']
    y = math.sin(math.radians(robot['bearing'])) * robot['range']
    return Vector2D(x, y)


# Main Robot class to keep track of robot states
class Robot:
    
    # 3.6V should give an indication that the battery is getting low, but this value can be experimented with.
    # Battery percentage might be a better
    BAT_LOW_VOLTAGE = 3.6

    # Firmware on both robots accepts wheel velocities between -1200 and 1200.
    # MAX_SPEED is specified in m/s and converted to motor units (×10000) when sending commands
    MAX_SPEED = 0.12 # Default value in m/s (0.12 m/s = 1200 motor units)
    HARD_TURN = 0
    SOFT_TURN = 1
    NO_TURN = 2
    HARD_TURN_ON_ANGLE_THRESHOLD = math.radians(90)
    SOFT_TURN_ON_ANGLE_THRESHOLD = math.radians(70)
    NO_TURN_ANGLE_THRESHOLD = math.radians(10)
    
    
    @classmethod
    def load_params(cls, config):
    
        for param in config[0]:
            print(param.tag)
            if param.tag == "wheel_turning":
                cls.MAX_SPEED = float(param.get("max_speed"))
                global MAX_SPEED
                MAX_SPEED = cls.MAX_SPEED
                cls.HARD_TURN_ON_ANGLE_THRESHOLD = math.radians(float(param.get("hard_turn_angle_threshold")))
                cls.SOFT_TURN_ON_ANGLE_THRESHOLD = math.radians(float(param.get("soft_turn_angle_threshold")))
                cls.NO_TURN_ANGLE_THRESHOLD = math.radians(float(param.get("no_turn_angle_threshold")))
            elif param.tag == "target_tracking":
                cls.KP = float(param.get("kp"))
                cls.KI = float(param.get("ki"))
                cls.KD = float(param.get("kd"))
                cls.THRES_RANGE = float(param.get("thres_range"))
            elif param.tag == "flocking":
                cls.TARGET_DISTANCE_WALK = float(param.get("target_distance_walk"))
                cls.TARGET_DISTANCE_TARGET = float(param.get("target_distance_target"))
                cls.GAIN = float(param.get("gain"))
                cls.EXPONENT = float(param.get("exponent"))
            elif param.tag == "motion":
                cls.MIN_RANDOM_WALK_ROTATION_ANGLE = float(param.get("random_walk_rotation_angle").split(",")[0])
                cls.MAX_RANDOM_WALK_ROTATION_ANGLE = float(param.get("random_walk_rotation_angle").split(",")[1])
                cls.BROADCAST_DURATION = float(param.get("broadcast_duration"))

    
    def __init__(self, robot_id, config=None, team_id=1):
        self.id = robot_id
        self.team_id = team_id
        self.connection = None
        
        self.teleop = False
        self.teleop_left = 0
        self.teleop_right = 0
        self.teleop_last_command = None  # Track last teleop command
        self.teleop_last_command_time = 0  # Timestamp of last teleop command
        self.move_to_target = False  # For teleop mode: whether to move to target
        self.share_target = False  # For teleop mode: whether to share target with others
        self.left = self.right = 0
        
        self.position = Vector2D(0,0)
        self.orientation = 0
        self.neighbours = {}
        
        # Init PID controller
        # PID output limits in motor units (MAX_SPEED is in m/s, 0.12 m/s = 1200 motor units)
        self.PID_heading = PID(
            Kp=self.KP,
            Ki=self.KI,
            Kd=self.KD,
            output_limits=(-self.MAX_SPEED * 10000, self.MAX_SPEED * 10000)
        )
        
        self.current_state = State.RANDOM_WALK
        self.turning_mechanism = self.NO_TURN
        
        self.led_colour = 'blue'
        
        # Message to send
        self.msg = Message()
        
        # Messages received
        self.messages = {}
        self.team_msgs = []
        self.other_msgs = []
        
        self.in_target = False
        self.target_found = False
        self.target_received = False
        self.broadcast_timer = 0
        self.rotation_remaining = 0.0
        self.last_orientation = None
        
        self.target = Vector2D(1000,1000) # Large default value when target position not set
        self.target_radius = 0.0
        self.dist_to_target = float('inf')
        self.in_target = False

        self.arena_limits = [] # max x and y coordinates of the arena, to be set by the user
        self.arena_margin_threshold = 0.1 # deffault 10% margin from the arena boundary to start repelling from it

    # Repulsion
    def generalized_lennard_jones_repulsion_walk(self, distance):
        f_norm_dist_exp = pow(self.TARGET_DISTANCE_WALK / distance, self.EXPONENT)
        return -self.GAIN / distance * (f_norm_dist_exp * f_norm_dist_exp)
    
    
    def generalized_lennard_jones_repulsion_target(self, distance):
        f_norm_dist_exp = pow(self.TARGET_DISTANCE_TARGET / distance, self.EXPONENT)
        return -self.GAIN / distance * (f_norm_dist_exp * f_norm_dist_exp)
    
    
    def control_step(self):
        print(Fore.LIGHTCYAN_EX + f'--- Robot {self.id} --- state: {self.current_state} ---')

        self.reset_variables()
        
        self.get_messages()
        
        if self.target.x != 1000 and self.target.y != 1000:
            # Transform target from arena-local to global coordinates
            global_target = Vector2D(
                self.target.x + self.arena_limits["min_x"],
                self.target.y + self.arena_limits["min_y"]
            )
            self.dist_to_target = self.position.distance_to(global_target)
            self.in_target = self.dist_to_target <= self.target_radius
        print(f"target {self.target}, in_target {self.in_target}")

        if self.current_state == State.RANDOM_WALK:
            self.target_received = False
            
            if not self.in_target:
                # Check if neighboring robots have found the target
                for msg in self.team_msgs:
                    if msg.target_position.x != 1000 and msg.target_position.y != 1000:
                        self.target = msg.target_position
                        self.target_received = True
                        break
        
            if self.in_target or self.target_received:
                self.current_state = State.BROADCAST_WALK
                self.number_of_blinks = 30
                self.broadcast_timer = int(self.BROADCAST_DURATION / TICK_DURATION)
                self.blink_interval = 5
                self.blink_timer = 0
                self.target_found = True
                print("State -> BROADCAST_WALK")
                print(f"in_target {self.in_target}, target_received {self.target_received}")
                
        elif self.current_state == State.BROADCAST_WALK:
            
            # Broadcast message while walking randomly for a certain duration
            self.broadcast_timer -= 1
            
            if not self.teleop and self.broadcast_timer <= 0:
                self.current_state = State.BROADCAST_HOMING
                print("State -> BROADCAST_HOMING")
            elif self.teleop and self.move_to_target:
                self.current_state = State.BROADCAST_HOMING
                print("State -> BROADCAST_HOMING")
                
        elif self.current_state == State.BROADCAST_HOMING:
            
            # Move towards target
            if self.in_target:
                self.current_state = State.IN_TARGET
                print("State -> IN_TARGET")
                
        elif self.current_state == State.IN_TARGET:
            
            # Stay in target and broadcast
            pass
    
        motion_vector = Vector2D(0,0)
        if self.teleop and not self.move_to_target:
            pass # skip
        else:
            if self.current_state == State.RANDOM_WALK or self.current_state == State.BROADCAST_WALK:
                # Random walk
                motion_vector = self.random_walk()
            elif self.current_state == State.BROADCAST_HOMING or self.current_state == State.IN_TARGET:
                
                # Move towards target
                target_force = self.get_attraction_vector()
                
                if self.current_state == State.IN_TARGET:
                    # Reduce attraction from the target center as it gets closer to it
                    target_force *= self.dist_to_target / self.target_radius
                    
                # Flocking or Repulsion force
                all_msgs = self.team_msgs + self.other_msgs
                repulsion_force = self.get_robot_repulsion_vector(all_msgs)
                
                motion_vector = target_force + repulsion_force
        
        # Set LED color according to its state
        if self.current_state == State.RANDOM_WALK:
            self.led_colour = 'blue'
        elif self.current_state == State.BROADCAST_WALK:
            if self.teleop:
                # Controlled by a user
                
                if self.share_target:
                    if self.blink_timer <= 0:
                        if self.led_colour == 'red':
                            self.led_colour = 'off'
                        else:
                            self.led_colour = 'red'
                    else:
                        self.blink_timer -= 1
    
            else:
                # Not controlled by a user
                if self.blink_timer <= 0:
                    if self.led_colour == 'red':
                        self.led_colour = 'off'
                    else:
                        self.led_colour = 'red'
                else:
                    self.blink_timer -= 1
                    
        elif self.current_state == State.BROADCAST_HOMING:
            self.led_colour = 'green'
        elif self.current_state == State.IN_TARGET:
            self.led_colour = 'green'
            
        # Set wheel speed
        if self.teleop and not self.move_to_target:
            # Follow the control vector
            # TODO: eight directions
            pass
        elif abs(motion_vector) > self.MAX_SPEED / 10:
            self.left, self.right = self.set_wheel_speeds_from_vector(motion_vector)
        else:
            self.left, self.right = 0, 0
            
        # Message to broadcast
        msg = Message()
        msg.id = self.id
        msg.team_id = self.team_id
        if self.current_state == State.BROADCAST_WALK or \
            self.current_state == State.BROADCAST_HOMING or \
            self.current_state == State.IN_TARGET:
                
            if self.teleop:
                if self.share_target:
                    msg.target_position = self.target
            else:
                msg.target_position = self.target
        
        self.msg = msg
    

    def reset_variables(self):
        self.team_msgs.clear()
        self.other_msgs.clear()
        
        
    def get_messages(self):
        print(f'messages: {self.messages}')

        for id, msg in self.messages.items():
            # Set vector pointing at neighbor (with respect to its own local frame)
            if str(id) in self.neighbours:
                msg.direction = get_vector(self.neighbours[str(id)])
                
            if msg.team_id == self.team_id:
                self.team_msgs.append(msg)
            else:
                self.other_msgs.append(msg)
                
                
    def get_attraction_vector(self):
        
        res_vec = Vector2D(0,0)
        
        # Transform target from arena-local to global coordinates
        global_target = Vector2D(
            self.target.x + self.arena_limits["min_x"],
            self.target.y + self.arena_limits["min_y"]
        )
    
        # Calculate a normalized vector that points to the next target
        res_vec = global_target - self.position
        
        # Transform to robot's local frame by rotating by negative heading
        orientation_rad = math.radians(self.orientation)
        res_vec = res_vec.rotate(-orientation_rad)
    
        if abs(res_vec) > 0:
            res_vec = res_vec.normalize() * self.MAX_SPEED
        
        return res_vec
    
    
    def get_robot_repulsion_vector(self, msgs):
        res_vec = Vector2D(0,0)
        counter = 0
        
        ids = []
        for msg in msgs:
            ids.append(str(msg.id))
        
        for key, value in self.neighbours.items():
            if key in ids:
                distance = value['range']
                if self.current_state == State.RANDOM_WALK or self.current_state == State.BROADCAST_WALK:
                    lf_force = self.generalized_lennard_jones_repulsion_walk(distance)
                elif self.current_state == State.BROADCAST_HOMING or self.current_state == State.IN_TARGET:
                    lf_force = self.generalized_lennard_jones_repulsion_target(distance)
                x = math.cos(math.radians(value['bearing'])) * lf_force
                y = math.sin(math.radians(value['bearing'])) * lf_force
                res_vec += Vector2D(x, y)
                counter += 1
                
        if counter > 0:
            res_vec /= counter
        
        # Normalize to MAX_SPEED if magnitude exceeds it
        magnitude = abs(res_vec)
        if magnitude > self.MAX_SPEED:
            res_vec = res_vec.normalize() * self.MAX_SPEED
            
        return res_vec


    def _boundary_local_components(self):
        curr_x = self.position.x
        curr_y = self.position.y

        print(f"curr_x: {curr_x}, curr_y: {curr_y}")

        min_x = self.arena_limits["min_x"]
        min_y = self.arena_limits["min_y"]
        max_x = self.arena_limits["max_x"]
        max_y = self.arena_limits["max_y"]
        margin_x = self.arena_margin_threshold * (max_x - min_x)
        margin_y = self.arena_margin_threshold * (max_y - min_y)

        boundary_collision = (
            curr_x < min_x + margin_x
            or curr_x > max_x - margin_x
            or curr_y < min_y + margin_y
            or curr_y > max_y - margin_y
        )

        if not boundary_collision:
            return False, 0, 0

        # Calculate repulsion vector in global coordinates (weighted by proximity)
        global_repulsion = Vector2D(0, 0)

        dist_left = curr_x - min_x
        dist_right = max_x - curr_x
        dist_bottom = curr_y - min_y
        dist_top = max_y - curr_y

        if dist_left < margin_x:
            global_repulsion.x += (margin_x - dist_left) / margin_x
        if dist_right < margin_x:
            global_repulsion.x -= (margin_x - dist_right) / margin_x
        if dist_bottom < margin_y:
            global_repulsion.y += (margin_y - dist_bottom) / margin_y
        if dist_top < margin_y:
            global_repulsion.y -= (margin_y - dist_top) / margin_y

        # Transform global repulsion vector to robot's local frame
        orientation_rad = math.radians(self.orientation)
        local_x = global_repulsion.x * math.cos(-orientation_rad) - global_repulsion.y * math.sin(-orientation_rad)
        local_y = global_repulsion.x * math.sin(-orientation_rad) + global_repulsion.y * math.cos(-orientation_rad)

        return True, local_x, local_y
    
    
    def random_walk(self):
        res_vec = Vector2D(0,0)
        
        all_msgs = self.team_msgs + self.other_msgs
        
        for msg in all_msgs:
            if abs(msg.direction) < self.TARGET_DISTANCE_WALK:
                return self.get_robot_repulsion_vector(all_msgs)
            
        def _normalize_angle(angle):
            return math.atan2(math.sin(angle), math.cos(angle))

        current_heading = math.radians(self.orientation)
        if self.last_orientation is None:
            self.last_orientation = current_heading

        if abs(self.rotation_remaining) > 1e-6:
            delta = _normalize_angle(current_heading - self.last_orientation)
            # Always reduce the absolute value of rotation_remaining
            self.rotation_remaining -= math.copysign(abs(delta), self.rotation_remaining)
            self.last_orientation = current_heading

            if abs(self.rotation_remaining) <= math.radians(2):
                self.rotation_remaining = 0.0
            else:
                turn_sign = 1.0 if self.rotation_remaining > 0 else -1.0
                res_vec = Vector2D(0, self.MAX_SPEED if turn_sign > 0 else -self.MAX_SPEED)
                print(f"Rotating: remaining={math.degrees(self.rotation_remaining)}")
                print(f"Rotation delta={math.degrees(delta)} heading={math.degrees(current_heading)}")
                print(f"res_vec: {res_vec}")
                return res_vec

        # Identify if any arena is violated
        boundary_collision, local_x, local_y = self._boundary_local_components()
        print(f"boundary_collision={boundary_collision} local_x={local_x} local_y={local_y}")

        if boundary_collision and local_x < 0:
            # local_y > 0 means boundary is on the left, local_y < 0 means boundary is on the right
            random_angle = random.uniform(self.MIN_RANDOM_WALK_ROTATION_ANGLE, self.MAX_RANDOM_WALK_ROTATION_ANGLE)
            random_angle_rad = math.radians(random_angle)

            if local_y > 0:
                print(f"Boundary detected on LEFT side of robot")
                turn_sign = 1.0
            elif local_y < 0:
                print(f"Boundary detected on RIGHT side of robot")
                turn_sign = -1.0
            else:
                turn_sign = 1.0

            self.rotation_remaining = turn_sign * abs(random_angle_rad)
            self.last_orientation = current_heading
            res_vec = Vector2D(0, self.MAX_SPEED if turn_sign > 0 else -self.MAX_SPEED)
        else:
            # Move forward when not near any boundary
            res_vec = Vector2D(self.MAX_SPEED, 0)
    
        print(f"res_vec: {res_vec}")
        return res_vec

    
    def set_wheel_speeds_from_vector(self, vector):
        heading_angle = math.atan2(vector.y, vector.x)
        heading_length = abs(vector)
        
        print('angle: {}'.format(heading_angle))
        print('length: {}'.format(heading_length))
        
        base_speed = min(heading_length, self.MAX_SPEED)

        # Smooth turning: scale wheel speeds continuously by heading angle
        max_angle = math.pi / 2
        speed_factor = (max_angle - abs(heading_angle)) / max_angle
        speed1 = base_speed + base_speed * (1.0 - speed_factor)
        speed2 = base_speed - base_speed * (1.0 - speed_factor)

        # Clamp to valid speed range
        speed1 = max(-self.MAX_SPEED, min(self.MAX_SPEED, speed1))
        speed2 = max(-self.MAX_SPEED, min(self.MAX_SPEED, speed2))
        
        if(heading_angle > 0):
            # Turn left
            left  = speed1
            right = speed2
        else:
            # Turn right
            left  = speed2
            right = speed1
        
        # print(f'Robot {self.id}: Setting speeds - left={left:.2f}, right={right:.2f} (base={base_speed:.2f}, mechanism={self.turning_mechanism})')
        
        # Convert from m/s to motor units (0.12 m/s = 1200 motor units)
        left = left * 10000
        right = right * 10000
        
        # print(f'Robot {self.id}: Motor units - left={left:.0f}, right={right:.0f}')
            
        return left, right
    
    
    def set_wheel_speeds_from_vector_homing(self, vector):
        pass
    
    
    def set_wheel_speeds_from_eight_directions(self, vector):
        pass
