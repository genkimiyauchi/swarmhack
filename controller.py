import math
from enum import Enum
from simple_pid import PID
from vector2d import Vector2D
from message import Message

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

    # Firmware on both robots accepts wheel velocities between -100 and 100.
    # This limits the controller to fit within that.
    MAX_SPEED = 100
    
    
    @classmethod
    def load_params(cls, config):
    
        for param in config[0]:
            if param.tag == 'target_tracking':
                cls.KP = 10 # TODO: Read from xml file
                cls.KI = 0 # TODO: Read from xml file
                cls.KD = 0 # TODO: Read from xml file
                cls.THRES_RANGE = 5 # TODO: Read from xml file
            elif param.tag == 'flocking':
                cls.TARGET_DISTANCE_WALK = 20 # TODO: Read from xml file
                cls.TARGET_DISTANCE_TARGET = 8 # TODO: Read from xml file
                cls.GAIN = 1000 # TODO: Read from xml file
                cls.EXPONENT = 6 # TODO: Read from xml file
            elif param.tag == 'motion':
                cls.MIN_RANDOM_WALK_ROTATION_ANGLE = 15 # TODO: Read from xml file
                cls.MAX_RANDOM_WALK_ROTATION_ANGLE = 90 # TODO: Read from xml file
                cls.BROADCAST_DURATION = 4 # TODO: Read from xml file

    
    def __init__(self, robot_id, config=None, team_id=1):
        self.id = robot_id
        
        # Init PID controller

        self.PID_heading = PID(
            kp=self.KP,
            ki=self.KI,
            kd=self.KD,
            output_limits=(-self.MAX_SPEED, self.MAX_SPEED)
        )
        
        self.current_state = State.RANDOM_WALK
        
        self.led_colour = 'blue'
        
        self.in_target = False
        self.target_found = False
        self.target_received = False
        self.random_walk_timer = 0
        self.broadcast_timer = 0
        
        self.target = Vector2D(1000,1000) # Large default value when target position not set


    # Repulsion
    def generalized_lennard_jones_repulsion_walk(self, distance):
        f_norm_dist_exp = pow(self.TARGET_DISTANCE_WALK / distance, self.EXPONENT)
        return -self.GAIN / distance * (f_norm_dist_exp * f_norm_dist_exp)
    
    
    def generalized_lennard_jones_repulsion_target(self, distance):
        f_norm_dist_exp = pow(self.TARGET_DISTANCE_TARGET / distance, self.EXPONENT)
        return -self.GAIN / distance * (f_norm_dist_exp * f_norm_dist_exp)
    
    
    def control_step(self):
        self.reset_variables()
        
        self.get_messages()
        
        # TODO: Get global position and orientation
        
        if self.target != Vector2D(1000,1000):
            # TODO: self.dist_to_target = 
            pass

        if self.current_state == State.RANDOM_WALK:
            self.target_received = False
            
            if not self.in_target:
                # Check if neighboring robots have found the target
                for msg in self.team_msgs:
                    if msg.target_position != Vector2D(1000,1000):
                        self.target = msg.target_position
                        self.target_received = True
                        break
        
            if self.in_target or self.target_received:
                self.current_state = State.BROADCAST_WALK
                self.number_of_blinks = 30
                self.broadcast_timer = int(self.BROADCAST_DURATION / 0.1) # 0.1 seconds per tick
                self.blink_interval = 5
                self.blink_timer = 0
                self.target_found = True
                
        elif self.current_state == State.BROADCAST_WALK:
            
            # Broadcast message while walking randomly for a certain duration
            self.broadcast_timer -= 1
            
            if not self.teleop and self.broadcast_timer <= 0:
                self.current_state = State.BROADCAST_HOMING
            elif self.teleop and self.move_to_target:
                self.current_state = State.BROADCAST_HOMING
                
        elif self.current_state == State.BROADCAST_HOMING:
            
            # Move towards target
            if self.in_target:
                self.current_state = State.IN_TARGET
                
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
        elif abs(motion_vector) > self.MAX_SPEED / 100:
            self.left, self.right = self.set_wheel_speeds(motion_vector) # TODO: pick set_wheel_speeds() that returns left and right
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
                self.team_msgs[id] = msg
            else:
                self.other_msgs[id] = msg
                
                
    def get_attraction_vector(self):
        # TODO:
        
        # Get current global position
        
        res_vec = Vector2D(0,0)
        
        # TODO
        
        return res_vec
    
    
    def get_robot_repulsion_vector(self, ids):
        res_vec = Vector2D(0,0)
        counter = 0
        
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
            
        return res_vec
    
    def get_obstacle_repulsion_vector(self):
        res_vec = Vector2D(0,0)

        # TODO: Repel from the boundary of the arena

        return res_vec
    
    
    def random_walk(self):
        
        # Decrement timer
        self.random_walk_timer -= 1
        
        all_msgs = self.team_msgs + self.other_msgs
        
        for msg in all_msgs:
            if abs(msg.direction) < self.TARGET_DISTANCE_WALK:
                return self.get_robot_repulsion_vector(all_msgs)
            
        # TODO: Get proximity sensor readings -> rely on distance to neighbor

        if self.random_walk_timer <= 0:
            length_left = length_right = -1
            index = 0
            
            # TODO
    
        return self.current_rotation
    
    
    def set_wheel_speeds_from_vector(self, vector):
        heading_angle = math.atan2(vector.y, vector.x)
        heading_length = abs(vector)
        
        print('angle: {}'.format(heading_angle))
        print('length: {}'.format(heading_length))
        
        speed_factor = (1.57 - abs(heading_angle)) / 1.57
        speed1 = self.MAX_SPEED + self.MAX_SPEED * (1.0 - speed_factor)
        speed2 = self.MAX_SPEED - self.MAX_SPEED * (1.0 - speed_factor)
        
        if(heading_angle > 0):
            # Turn left
            left  = speed1
            right = speed2
        else:
            # Turn right
            left  = speed2
            right = speed1
            
        return left, right
    
    
    def set_wheel_speeds_from_vector_homing(self, vector):
        pass
    
    
    def set_wheel_speeds_from_eight_directions(self, vector):
        pass
