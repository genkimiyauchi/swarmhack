import asyncio
import signal
import sys
import atexit
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosed
import json
from pipuck.pipuck import PiPuck

pipuck = PiPuck(epuck_version=2)

# Cleanup function
def cleanup():
    print("Running cleanup: setting motor speeds to 0.")
    pipuck.epuck.set_motor_speeds(0, 0)

def signal_handler(signum, frame):
    signame = signal.Signals(signum).name if signum in signal.Signals._value2member_map_ else str(signum)
    print(f"Received signal {signum} ({signame}). Shutting down.")
    cleanup()
    sys.exit(0)

# Register other signals if you like (e.g., SIGHUP, SIGTERM, etc.)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGHUP, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Also register atexit as a fallback
atexit.register(cleanup)

async def handler(websocket):
    try:
        async for packet in websocket:
            message = json.loads(packet)

            # Process any requests received
            reply = {}
            send_reply = False

            if "check_awake" in message:
                reply["awake"] = True
                send_reply = True


            if "get_battery" in message:
                charging, voltage, percentage = pipuck.get_battery_state("epuck")
                reply["battery"] = {}
                reply["battery"]["voltage"] = voltage
                reply["battery"]["percentage"] = int(percentage * 100)
                send_reply = True

            if "set_leds_colour" in message:
                try:
                    pipuck.set_leds_colour(message["set_leds_colour"])
                except (KeyError, ValueError):
                    pass

            if send_reply:
                await websocket.send(json.dumps(reply))

            if "set_motor_speeds" in message:
                try:
                    left_in = int(message["set_motor_speeds"]["left"])
                    right_in = int(message["set_motor_speeds"]["right"])
                    left_clamped = max(min(left_in, 1000), -1000)
                    right_clamped = max(min(right_in, 1000), -1000)
                    left_scaled = left_clamped * 1
                    right_scaled = right_clamped * 1
                    pipuck.epuck.set_motor_speeds(left_scaled, right_scaled)
                except (KeyError, ValueError):
                    pass

    except ConnectionClosedError as e:
        print(f"Client closed abruptly: {e}")
    except ConnectionClosed as e:
        print(f"Client disconnected: {e}")
    except Exception as e:
        print(f"Error in handler: {e}")

if __name__ == "__main__":
    try:
        server = websockets.serve(ws_handler=handler, host=None, port=80)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(server)
        print("WebSocket server started. Press Ctrl+C to stop.")
        loop.run_forever()
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Shutting down.")
        cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error in main loop: {e}")
        cleanup()
        sys.exit(1)
    finally:
        print("Main loop exited. Final cleanup.")
        cleanup()