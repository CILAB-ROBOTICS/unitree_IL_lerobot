# this file is legacy, need to fix.
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
# from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_ # OLD IDL, REMOVED
# from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_ # OLD IDL, REMOVED

# NEW IMPORTS for Inspire Hand SDK
from inspire_sdkpy import inspire_dds, inspire_hand_defaut

# from teleop.robot_control.hand_retargeting import HandRetargeting, HandType # Assuming this remains the same
import numpy as np
# from enum import IntEnum # Old Enums for joint indexing might not be needed for new DDS messages
import threading
import time
from enum import IntEnum
from multiprocessing import Process, Array, Lock # Removed shared_memory as Array is used

inspire_tip_indices = [4, 9, 14, 19, 24] # Assuming this remains relevant for hand_retargeting
Inspire_Num_Motors = 6 # Number of motors per hand

# NEW DDS TOPIC NAMES (assuming these are the correct topics based on SDK examples)
kTopicInspireCtrlLeft = "rt/inspire_hand/ctrl/l"
kTopicInspireCtrlRight = "rt/inspire_hand/ctrl/r"
kTopicInspireStateLeft = "rt/inspire_hand/state/l"
kTopicInspireStateRight = "rt/inspire_hand/state/r"

class Inspire_Controller:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock=None, dual_hand_state_array=None,
                 dual_hand_action_array=None, fps=100.0, Unit_Test=False, network_interface=""): # Added network_interface
        print("Initialize Inspire_Controller...")
        self.fps = fps
        self.Unit_Test = Unit_Test

        # Initialize DDS Channel Factory
        # This should ideally be called once per process.
        # If multiple controllers or DDS entities run in the same process, ensure this is handled.

        if not self.Unit_Test:
            pass
        else:
            try:
                ChannelFactoryInitialize(0, network_interface)
                print(f"DDS ChannelFactory initialized with interface: '{network_interface if network_interface else 'default'}'")
            except Exception as e:
                print(f"Warning: DDS ChannelFactoryInitialize failed or already initialized: {e}")

        # Initialize hand command publishers
        self.LeftHandCmd_publisher = ChannelPublisher(kTopicInspireCtrlLeft, inspire_dds.inspire_hand_ctrl)
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(kTopicInspireCtrlRight, inspire_dds.inspire_hand_ctrl)
        self.RightHandCmd_publisher.Init()

        # Initialize hand state subscribers
        self.LeftHandState_subscriber = ChannelSubscriber(kTopicInspireStateLeft, inspire_dds.inspire_hand_state)
        self.LeftHandState_subscriber.Init() # Consider using callback if preferred: Init(callback_func, period_ms)
        self.RightHandState_subscriber = ChannelSubscriber(kTopicInspireStateRight, inspire_dds.inspire_hand_state)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states ([0,1] normalized values)
        self.left_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)

        # Initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state_loop)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # Wait for initial DDS messages (optional, but good for ensuring connection)
        wait_count = 0
        while not (any(self.left_hand_state_array.get_obj()) or any(self.right_hand_state_array.get_obj())):
            if wait_count % 100 == 0: # Print every second
                 print(f"[Inspire_Controller] Waiting to subscribe to hand states from DDS (L: {any(self.left_hand_state_array.get_obj())}, R: {any(self.right_hand_state_array.get_obj())})...")
            time.sleep(0.01)
            wait_count +=1
            if wait_count > 500: # Timeout after 5 seconds
                print("[Inspire_Controller] Warning: Timeout waiting for initial hand states. Proceeding anyway.")
                break
        print("[Inspire_Controller] Initial hand states received or timeout.")

        hand_control_process = Process(target=self.control_process_loop, args=(
            left_hand_array, right_hand_array, self.left_hand_state_array, self.right_hand_state_array,
            dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array))
        hand_control_process.daemon = True
        hand_control_process.start()

        print("Initialize Inspire_Controller OK!\n")

    def _subscribe_hand_state_loop(self):
        print("[Inspire_Controller] Subscribe thread started.")
        while True:
            # Left Hand
            # Read() 메소드에서 block 및 timeout_ms 인자 제거
            left_state_msg = self.LeftHandState_subscriber.Read() 
            if left_state_msg is not None: # 메시지가 성공적으로 수신되었는지 확인
                if hasattr(left_state_msg, 'angle_act') and len(left_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.left_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            # Normalize from 0-1000 (0=closed, 1000=open) to [0,1] (0=closed, 1=open)
                            self.left_hand_state_array[i] = left_state_msg.angle_act[i] / 1000.0
                else:
                    print(f"[Inspire_Controller] Warning: Received left_state_msg but attributes are missing or incorrect. Type: {type(left_state_msg)}, Content: {str(left_state_msg)[:100]}") # 메시지 내용 일부 출력
            # else:
                # print("[Inspire_Controller] No left hand state message.") # 디버깅용, 자주 출력될 수 있음

            # Right Hand
            # Read() 메소드에서 block 및 timeout_ms 인자 제거
            right_state_msg = self.RightHandState_subscriber.Read()
            if right_state_msg is not None: # 메시지가 성공적으로 수신되었는지 확인
                if hasattr(right_state_msg, 'angle_act') and len(right_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.right_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            # Normalize from 0-1000 (0=closed, 1000=open) to [0,1] (0=closed, 1=open)
                            self.right_hand_state_array[i] = right_state_msg.angle_act[i] / 1000.0
                else:
                    print(f"[Inspire_Controller] Warning: Received right_state_msg but attributes are missing or incorrect. Type: {type(right_state_msg)}, Content: {str(right_state_msg)[:100]}") # 메시지 내용 일부 출력
            # else:
                # print("[Inspire_Controller] No right hand state message.") # 디버깅용, 자주 출력될 수 있음
            
            # Add a small sleep to prevent busy-waiting if Read is non-blocking or times out frequently
            # If Read is blocking and messages are frequent, this sleep might be less critical
            # Or, if using callback-based subscription, this loop structure changes.
            time.sleep(0.002) # Adjust as needed, matches original sleep
        
    def _send_hand_command(self, left_angle_cmd_scaled, right_angle_cmd_scaled):
        """
        Send scaled angle commands [0-1000] to both hands.
        """
        # Left Hand Command
        left_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl() # Creates a new message instance
        left_cmd_msg.angle_set = left_angle_cmd_scaled # Expects list of 6 int16
        left_cmd_msg.mode = 0b0001 # Mode 1: Angle control
        self.LeftHandCmd_publisher.Write(left_cmd_msg)

        # Right Hand Command
        right_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        right_cmd_msg.angle_set = right_angle_cmd_scaled
        right_cmd_msg.mode = 0b0001 # Mode 1: Angle control
        self.RightHandCmd_publisher.Write(right_cmd_msg)
        # print("Hand control commands published.")

    def control_process_loop(self, left_hand_input_array, right_hand_input_array, 
                             shared_left_hand_state_array, shared_right_hand_state_array,
                             dual_hand_data_lock=None, dual_hand_state_array_shm=None, dual_hand_action_array_shm=None):
        print("[Inspire_Controller] Control process started.")
        running = True

        # Default target q ([0,1] normalized, 1.0 = fully open)
        # This matches the output of the `normalize` function: 0 (closed) to 1 (open)
        current_left_q_target_norm = np.ones(Inspire_Num_Motors, dtype=float) 
        current_right_q_target_norm = np.ones(Inspire_Num_Motors, dtype=float)

        # No need to initialize self.hand_msg here as it's created per-send in _send_hand_command

        try:
            while running: # In a real application, you'd have a way to set running to False to exit
                start_time = time.time()

                # get dual hand state
                left_hand_mat  = np.array(left_hand_input_array[:]).copy()
                right_hand_mat = np.array(right_hand_input_array[:]).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(shared_left_hand_state_array[:]), np.array(shared_right_hand_state_array[:])))

                # get dual hand action
                action_data = np.concatenate((left_hand_mat, right_hand_mat))    
                if dual_hand_state_array_shm and dual_hand_action_array_shm:
                    with dual_hand_data_lock:
                        dual_hand_state_array_shm[:] = state_data
                        dual_hand_action_array_shm[:] = action_data
                
                left_q_target = left_hand_mat
                right_q_target = right_hand_mat
                
                def normalize_radians_to_01(val):
                    # (max_val - val) makes it so:
                    # if val is min_val (open), result is 1.
                    # if val is max_val (closed), result is 0.
                    return np.clip(val, 0.0, 1.0)
                
                for idx in range(Inspire_Num_Motors):
                    current_left_q_target_norm[idx] = normalize_radians_to_01(left_q_target[idx])
                    current_right_q_target_norm[idx] = normalize_radians_to_01(right_q_target[idx])
                    
                scaled_left_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in current_left_q_target_norm]
                scaled_right_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in current_right_q_target_norm]
                
                self._send_hand_command(scaled_left_cmd, scaled_right_cmd)
                
                # Get human hand input data (e.g., from VR gloves, motion capture)
                # Assuming these arrays are populated by another process/thread
                # left_hand_mat = np.array(left_hand_input_array[:]).reshape(25, 3).copy()
                # right_hand_mat = np.array(right_hand_input_array[:]).reshape(25, 3).copy()

                # # Read current robot hand state from shared arrays (normalized [0,1])
                # with shared_left_hand_state_array.get_lock():
                #     current_left_q_state_norm = np.array(shared_left_hand_state_array[:])
                # with shared_right_hand_state_array.get_lock():
                #     current_right_q_state_norm = np.array(shared_right_hand_state_array[:])
                
                # state_data_for_logging = np.concatenate((current_left_q_state_norm, current_right_q_state_norm))

                # Check if hand data has been initialized before retargeting
                # (This condition is from original code, might need adjustment)
                # human_hand_data_valid = not np.all(right_hand_mat == 0.0) and \
                #                         not np.all(left_hand_mat[4] == np.array([-1.13, 0.3, 0.15]))

                # if human_hand_data_valid:
                #     ref_left_value = left_hand_mat[inspire_tip_indices]
                #     ref_right_value = right_hand_mat[inspire_tip_indices]

                    # Retargeting returns joint angles (radians for the specific robot model in HandRetargeting)
                    # then reordered by _to_hardware mapping
                    # raw_left_q_target_rad = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[
                    #     self.hand_retargeting.left_dex_retargeting_to_hardware]
                    # raw_right_q_target_rad = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[
                    #     self.hand_retargeting.right_dex_retargeting_to_hardware]
                    
                    # Normalize retargeted radian values to [0,1] (0=closed, 1=open)
                    # This uses the same normalization logic as in the original code.
                    # Ensure min_val and max_val correspond to "open" and "closed" states in radians.
                    # def normalize_radians_to_01(val):
                    #     # (max_val - val) makes it so:
                    #     # if val is min_val (open), result is 1.
                    #     # if val is max_val (closed), result is 0.
                    #     return np.clip(val, 0.0, 1.0)

                    # Min/max values in radians from original code comments
                    # idx 0~3: 0(open) ~ 1.7(closed)
                    # idx 4:   0(open) ~ 0.5(closed)
                    # idx 5:  -0.1(open) ~ 1.3(closed)
                    # Note: The interpretation of min/max might need to be flipped if the retargeting library
                    # outputs values where smaller means more closed. The original normalize function implies
                    # larger radian value = more closed for these specific ranges.

                    # for idx in range(Inspire_Num_Motors):
                    #     if idx <= 3:
                    #         min_r, max_r = 0.0, 1.7
                    #     elif idx == 4:
                    #         min_r, max_r = 0.0, 0.5
                    #     elif idx == 5:
                    #         min_r, max_r = -0.1, 1.3
                        
                    #     current_left_q_target_norm[idx] = normalize_radians_to_01(raw_left_q_target_rad[idx], min_r, max_r)
                    #     current_right_q_target_norm[idx] = normalize_radians_to_01(raw_right_q_target_rad[idx], min_r, max_r)
                
                # else: keep previous target if human hand data is not valid

                # Scale normalized q_target [0(closed),1(open)] to Inspire Hand's angle_set [0(closed),1000(open)]
                # scaled_left_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in current_left_q_target_norm]
                # scaled_right_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in current_right_q_target_norm]

                # # For logging/external use, use the normalized [0,1] action targets
                # action_data_for_logging = np.concatenate((current_left_q_target_norm, current_right_q_target_norm))
                
                # if dual_hand_state_array_shm and dual_hand_action_array_shm and dual_hand_data_lock:
                #     with dual_hand_data_lock:
                #         dual_hand_state_array_shm[:] = state_data_for_logging
                #         dual_hand_action_array_shm[:] = action_data_for_logging

                # Send commands to the robot
                # self._send_hand_command(scaled_left_cmd, scaled_right_cmd)

                # Loop timing
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1.0 / self.fps) - time_elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # else:
                #     print(f"[Inspire_Controller] Warning: Control loop took too long: {time_elapsed*1000:.2f} ms")

        except KeyboardInterrupt:
            print("[Inspire_Controller] Control process received KeyboardInterrupt. Exiting.")
        finally:
            running = False # Ensure loop terminates
            print("[Inspire_Controller] Control process has been closed.")

class Inspire_Right_Hand_JointIndex(IntEnum):
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5

class Inspire_Left_Hand_JointIndex(IntEnum):
    kLeftHandPinky = 6
    kLeftHandRing = 7
    kLeftHandMiddle = 8
    kLeftHandIndex = 9
    kLeftHandThumbBend = 10
    kLeftHandThumbRotation = 11
    
# Example of how this controller might be used (simplified)
if __name__ == '__main__':
    print("Starting Inspire_Controller example...")

    # Mock human hand input arrays (25 landmarks * 3 coords = 75 per hand)
    # In a real scenario, these would be populated by a hand tracking system
    mock_left_hand_input = Array('d', 75, lock=True)
    mock_right_hand_input = Array('d', 75, lock=True)

    # Initialize with some default "open hand" pose data (example, adjust as needed)
    # This part is highly dependent on what your HandRetargeting expects.
    # For now, just zeroing them.
    # with mock_left_hand_input.get_lock():
    #     mock_left_hand_input[:] = [0.0] * 75
    # with mock_right_hand_input.get_lock():
    #     mock_right_hand_input[:] = [0.0] * 75
    
    # Example: Make the right hand "valid" to trigger retargeting, left hand invalid
    # This is just to satisfy the `human_hand_data_valid` condition in the example.
    # The specific values for `left_hand_mat[4]` check might need to be different
    # or this validity check needs to be more robust.
    with mock_right_hand_input.get_lock():
        # Make it not all zeros
        for i in range(len(mock_right_hand_input)):
            mock_right_hand_input[i] = (i % 10) * 0.01 
            
    with mock_left_hand_input.get_lock():
        # Set to a specific "invalid" initial pose as per original code
        temp_left_mat = np.zeros((25,3))
        temp_left_mat[4] = np.array([-1.13, 0.3, 0.15]) # This makes left hand "invalid" in the condition
        mock_left_hand_input[:] = temp_left_mat.flatten()


    # Shared arrays for policy learning/logging (optional)
    shared_lock = Lock()
    shared_state = Array('d', Inspire_Num_Motors * 2, lock=False) # 6 per hand
    shared_action = Array('d', Inspire_Num_Motors * 2, lock=False) # 6 per hand

    # Instantiate the controller
    # You might need to specify a network_interface if not using default
    # e.g., network_interface="eth0"
    try:
        controller = Inspire_Controller(
            left_hand_array=mock_left_hand_input,
            right_hand_array=mock_right_hand_input,
            dual_hand_data_lock=shared_lock,
            dual_hand_state_array=shared_state,
            dual_hand_action_array=shared_action,
            fps=50.0, # Lower FPS for example
            Unit_Test=False,
            network_interface="" # Or specify your network interface for DDS
        )

        # Simulate human hand input updates (normally from another system)
        # For this example, the control loop runs indefinitely.
        # We can just let it run and observe, or add a way to stop it.
        count = 0
        while True:
            time.sleep(1.0) # Main thread does something else or just waits
            # Simulate a slight change in human hand input to see retargeting work
            with mock_right_hand_input.get_lock():
                 # Make a noticeable change to one coordinate
                mock_right_hand_input[inspire_tip_indices[0]*3 + 1] = 0.1 + (count % 10) * 0.02
            
            with shared_lock:
                print(f"Logged State: {list(shared_state[:])}")
                print(f"Logged Action: {list(shared_action[:])}")
            count +=1
            if count > 300 : # Run for 5 minutes
                 print("Example finished.")
                 break


    except Exception as e:
        print(f"An error occurred in the example: {e}")
    finally:
        # In a real app, you'd signal the controller's processes/threads to stop cleanly.
        # Since they are daemons, they will exit when the main program exits.
        print("Exiting main program.")