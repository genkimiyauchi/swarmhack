from enum import Enum
from vector2d import Vector2D

# Structure to store incoming data received from other robots
class Message:
    def __init__(self) -> None:
        # Core
        self.direction = Vector2D(0,0) # Vector2D
        self.id = None # str
        self.team_id = None # int
        
        self.in_target = False # bool
        
        self.target_position = Vector2D(1000,1000) # Vector2D. Large default value when target position not set

    # Checks whether the Message is empty or not by checking the direction it was received from
    def empty(self) -> bool:
        return self.id == None
