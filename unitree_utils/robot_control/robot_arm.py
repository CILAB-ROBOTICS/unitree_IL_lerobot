import numpy as np
import threading
import time

from unitree_dds_wrapper.idl import unitree_hg
from unitree_dds_wrapper.publisher import Publisher
from unitree_dds_wrapper.subscription import Subscription

import struct
from enum import IntEnum
import copy

kTopicLowCommand = "rt/lowcmd"
kTopicLowState = "rt/lowstate"
G1_29_Num_Motors = 35
 

class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None

class G1_29_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]

class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data

class G1_29_ArmController:
    def __init__(self):
        print("Initialize G1_29_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)

        self.msg = unitree_hg.msg.dds_.LowCmd_()
        self.msg.mode_machine = 3 # g1 is 3, h1_2 is 4
        self.__packFmtHGLowCmd = '<2B2x' + 'B3x5fI' * 35 + '5I'
        self.msg.head = [0xFE, 0xEF]
        
        self.lowcmd_publisher = Publisher(unitree_hg.msg.dds_.LowCmd_, kTopicLowCommand)
        self.lowstate_subscriber = Subscription(unitree_hg.msg.dds_.LowState_, kTopicLowState)

        self.lowstate_buffer = DataBuffer()

        self.kp_high = 100.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 30.0
        self.kd_wrist = 1.5

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.01)
            print("Waiting to subscribe dds...")
        
        self.all_motor_q = self.get_current_motor_q()
        print(f"Current all body motor state q:\n{self.all_motor_q} \n")
        print(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        print("Lock all joints except two arms...\n")
        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id in G1_29_JointArmIndex:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        self.pre_communication()
        self.lowcmd_publisher.msg = self.msg
        self.lowcmd_publisher.write()
        print("Lock OK!")

        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.arm_lock = threading.Lock()
        self.publish_thread.start()

        print("Initialize G1_29_ArmController OK!")

    def __Trans(self, packData):
        calcData = []
        calcLen = ((len(packData)>>2)-1)

        for i in range(calcLen):
            d = ((packData[i*4+3] << 24) | (packData[i*4+2] << 16) | (packData[i*4+1] << 8) | (packData[i*4]))
            calcData.append(d)

        return calcData
    
    def __Crc32(self, data):
        bit = 0
        crc = 0xFFFFFFFF
        polynomial = 0x04c11db7

        for i in range(len(data)):
            bit = 1 << 31
            current = data[i]

            for b in range(32):
                if crc & 0x80000000:
                    crc = (crc << 1) & 0xFFFFFFFF
                    crc ^= polynomial
                else:
                    crc = (crc << 1) & 0xFFFFFFFF

                if current & bit:
                    crc ^= polynomial

                bit >>= 1
        
        return crc
    
    def __pack_crc(self):
        origData = []
        origData.append(self.msg.mode_pr)
        origData.append(self.msg.mode_machine)

        for i in range(35):
            origData.append(self.msg.motor_cmd[i].mode)
            origData.append(self.msg.motor_cmd[i].q)
            origData.append(self.msg.motor_cmd[i].dq)
            origData.append(self.msg.motor_cmd[i].tau)
            origData.append(self.msg.motor_cmd[i].kp)
            origData.append(self.msg.motor_cmd[i].kd)
            origData.append(self.msg.motor_cmd[i].reserve)

        origData.extend(self.msg.reserve)
        origData.append(self.msg.crc)
        calcdata = struct.pack(self.__packFmtHGLowCmd, *origData)
        calcdata =  self.__Trans(calcdata)
        self.msg.crc = self.__Crc32(calcdata)

    def pre_communication(self):
        self.__pack_crc()

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        while True:
            time1 = time.time()
            time.sleep(self.control_dt)
            
            with self.arm_lock:
                arm_q_target   = copy.deepcopy(self.q_target)
                arm_tauff_target = copy.deepcopy(self.tauff_target)

            cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            for idx, id in enumerate(G1_29_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            self.pre_communication()
            self.lowcmd_publisher.msg = self.msg
            self.lowcmd_publisher.write()

            if self._speed_gradual_max is True:
                elapsed_time = time1 - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, elapsed_time / 5.0))

            time2 = time.time()
            # print(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # print(f"_ctrl_motor_state fps:{1/(time2-time1)}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        self.q_target = q_target
        self.tauff_target = tauff_target
            
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_29_JointArmIndex])

    def _subscribe_motor_state(self):
        while True:
            if self.lowstate_subscriber.msg:
                lowstate = G1_29_LowState()
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q  = self.lowstate_subscriber.msg.motor_state[id].q
                    lowstate.motor_state[id].dq = self.lowstate_subscriber.msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch,
            G1_29_JointIndex.kRightAnklePitch,
            # Left arm
            G1_29_JointIndex.kLeftShoulderPitch,
            G1_29_JointIndex.kLeftShoulderRoll,
            G1_29_JointIndex.kLeftShoulderYaw,
            G1_29_JointIndex.kLeftElbow,
            # Right arm
            G1_29_JointIndex.kRightShoulderPitch,
            G1_29_JointIndex.kRightShoulderRoll,
            G1_29_JointIndex.kRightShoulderYaw,
            G1_29_JointIndex.kRightElbow,
        ]
        return motor_index in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll,
            G1_29_JointIndex.kLeftWristPitch,
            G1_29_JointIndex.kLeftWristyaw,
            G1_29_JointIndex.kRightWristRoll,
            G1_29_JointIndex.kRightWristPitch,
            G1_29_JointIndex.kRightWristYaw,
        ]
        return motor_index in wrist_motors

class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28
    
    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34

if __name__ == "__main__":
    from robot_arm_ik import G1_29_ArmIK
    import pinocchio as pin 

    arm_ik = G1_29_ArmIK(Unit_Test = True, Visualization = False)
    g1arm = G1_29_ArmController()

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.2, 0.1]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.2, 0.1]),
    )

    rotation_speed = 0.005  # Rotation speed in radians per iteration
    q_target = np.zeros(35)
    tauff_target = np.zeros(35)

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program): \n")
    if user_input.lower() == 's':
        step = 0
        g1arm.speed_gradual_max()
        while True:
            if step <= 120:
                angle = rotation_speed * step
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation += np.array([0.001,  0.001, 0.001])
                R_tf_target.translation += np.array([0.001, -0.001, 0.001])
            else:
                angle = rotation_speed * (240 - step)
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation -= np.array([0.001,  0.001, 0.001])
                R_tf_target.translation -= np.array([0.001, -0.001, 0.001])

            L_tf_target.rotation = L_quat.toRotationMatrix()
            R_tf_target.rotation = R_quat.toRotationMatrix()

            current_lr_arm_q  = g1arm.get_current_dual_arm_q()
            current_lr_arm_dq = g1arm.get_current_dual_arm_dq()

            sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, current_lr_arm_q, current_lr_arm_dq)

            g1arm.ctrl_dual_arm(sol_q, sol_tauff)

            step += 1
            if step > 240:
                step = 0
            time.sleep(0.01)