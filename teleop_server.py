#!/usr/bin/env python3

import asyncio
import websockets
import json
from enum import Enum

class State(Enum):
    START = 1
    SELECT = 2
    DRIVE = 3

# Track all control clients (robot_client.py connections)
control_clients = set()

async def send_message(websocket, message):
    await websocket.send(json.dumps({"prompt": message}))

async def broadcast_control(message):
    """Broadcast control messages to all robot_client connections"""
    if control_clients:
        await asyncio.gather(
            *[client.send(json.dumps(message)) for client in control_clients],
            return_exceptions=True
        )


async def handler(websocket):
    # Determine if this is a control client or keyboard client
    # Control clients will send {"register": "control"} on connection
    first_message = await websocket.recv()
    message = json.loads(first_message)
    
    if message.get("register") == "control":
        # This is a robot_client connection
        control_clients.add(websocket)
        print(f"Control client connected. Total: {len(control_clients)}")
        try:
            async for packet in websocket:
                pass  # Control clients only receive, don't send
        finally:
            control_clients.remove(websocket)
            print(f"Control client disconnected. Total: {len(control_clients)}")
        return
    
    # Otherwise, it's a keyboard client - handle keyboard input
    await handle_keyboard_client(websocket, first_message)


async def handle_keyboard_client(websocket, first_message):

    state = State.START
    robot_id = ""
    valid_robots = [1, 2, 10, 23]
    forwards = "w"
    backwards = "s"
    left = "a"
    right = "d"
    stop = " "
    release = "q"
    
    # Process first message
    message = json.loads(first_message)
    if "key" in message and message["key"] == "teleop_start":
        state = State.START
        # Send the initial prompt immediately
        await send_message(websocket, f"\r\nEnter robot ID ({valid_robots}), then press return: ")
        robot_id = ""
        state = State.SELECT

    async for packet in websocket:
        message = json.loads(packet)
        print(message)

        if "key" in message:

            key = message["key"]

            if state == State.SELECT:
                if key == "\r":
                    valid = False
                    try:
                        if int(robot_id) in valid_robots:
                            valid = True
                            # Broadcast robot selection to control clients
                            await broadcast_control({
                                "teleop_control": True,
                                "robot_id": int(robot_id),
                                "command": "select"
                            })
                            await send_message(websocket, f"\r\nControlling robot ({release} to release): " + robot_id)
                            await send_message(websocket, f"\r\nControls: Forwards = {forwards}; Backwards = {backwards}; Left = {left}; Right = {right}; Stop = SPACE")
                            state = State.DRIVE
                    except ValueError:
                        pass

                    if not valid:
                        await send_message(websocket, "\r\nInvalid robot ID, try again: ")
                        robot_id = ""
                        state = State.SELECT

                else:
                    await send_message(websocket, key)
                    robot_id = robot_id + key

            elif state == State.DRIVE:
                if key == release:
                    # Broadcast robot release to control clients
                    await broadcast_control({
                        "teleop_control": False,
                        "robot_id": int(robot_id),
                        "command": "release"
                    })
                    await send_message(websocket, "\r\nReleasing control of robot: " + robot_id)
                    state = State.START
                elif key == forwards:
                    await broadcast_control({
                        "teleop_control": True,
                        "robot_id": int(robot_id),
                        "command": "forward"
                    })
                    await send_message(websocket, "\r\nDriving forwards")
                elif key == backwards:
                    await broadcast_control({
                        "teleop_control": True,
                        "robot_id": int(robot_id),
                        "command": "backward"
                    })
                    await send_message(websocket, "\r\nDriving backwards")
                elif key == left:
                    await broadcast_control({
                        "teleop_control": True,
                        "robot_id": int(robot_id),
                        "command": "left"
                    })
                    await send_message(websocket, "\r\nTurning left")
                elif key == right:
                    await broadcast_control({
                        "teleop_control": True,
                        "robot_id": int(robot_id),
                        "command": "right"
                    })
                    await send_message(websocket, "\r\nTurning right")
                elif key == stop:
                    await broadcast_control({
                        "teleop_control": True,
                        "robot_id": int(robot_id),
                        "command": "stop"
                    })
                    await send_message(websocket, "\r\nStopping")
                else:
                    await send_message(websocket, "\r\nUnrecognised command")


async def main():
    async with websockets.serve(handler, host="0.0.0.0", port=7000, ping_interval=None, ping_timeout=None):
        print("Teleop server started on ws://0.0.0.0:7000")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())