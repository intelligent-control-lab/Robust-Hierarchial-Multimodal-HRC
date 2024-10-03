#!/usr/bin/env python
import os,sys
import rospy
from std_msgs.msg import String,Float64MultiArray
from geometry_msgs.msg import Pose,PoseStamped
import time
import argparse
from pathlib import Path
FILE_DIR = Path(__file__).parent
import pdb

sys.path.append(f'{FILE_DIR}/../traj_intention')
from Dataset import INTENTION_LIST
# from speech import COMMAND_LIST
COMMAND_LIST = ["stop","short","long","get up","spin","rotate"]


RETRACT_POSITION = (0.2,0,0.19,0,-0.7,-0.7,0)
BASE = (-0.18,0.28,0.25,0,-0.7,-0.7,0)
kinova_control_pub = rospy.Publisher("/kinova_demo/pose_cmd", PoseStamped, queue_size=1)
kinova_grip_pub = rospy.Publisher("/siemens_demo/gripper_cmd", Float64MultiArray, queue_size=1)
def ComposePoseFromTransQuat(data_frame):
    # assert (len(data_frame.shape) == 1 and data_frame.shape[0] == 7)
    pose = Pose()
    pose.position.x = data_frame[0]
    pose.position.y = data_frame[1]
    pose.position.z = data_frame[2]
    pose.orientation.w = data_frame[3]
    pose.orientation.x = data_frame[4]
    pose.orientation.y = data_frame[5]
    pose.orientation.z = data_frame[6]
    return pose

class PlanGraph:
    def __init__(self,args):
        self.TUBE_SUM = {"short":8,"long":4}
        self.tube_count = {"short":0,"long":0}
        self.screw_count = {"bottom":0,"four_tubes":0,"top":0}
        self.wheels_count = 0
        self.action_history = []
        self.stage_record = {"bottom":[],"four_tubes":[],"top":[]} # tube records
        self.stage = None
        self.stage_history = [] # stages that have already been completed
        if args.stageI_done:
            self.tube_count = {"short":2,"long":2}
            self.screw_count = {"bottom":4,"four_tubes":0,"top":0}
            self.action_history = ["get_short_tubes","get_long_tubes","get_short_tubes","get_long_tubes"]
            self.stage_record = {"bottom":["get_short_tubes","get_long_tubes","get_short_tubes","get_long_tubes"],"four_tubes":[],"top":[]}
            self.stage_history = ["bottom"]
        if args.user_spin:
            self.tube_count = {"short":2,"long":2}
            self.screw_count = {"bottom":0,"four_tubes":0,"top":0}
            self.action_history = ["get_short_tubes","get_long_tubes","get_short_tubes","get_long_tubes"]
            self.stage_record = {"bottom":["get_short_tubes","get_long_tubes","get_short_tubes","get_long_tubes"],"four_tubes":[],"top":[]}
            self.stage_history = ["bottom"]
            
class Receiver:
    def __init__(self,args):
        self.intention_list = []
        self.command_list = []
        self.command = None
        self.old_command_data = None
        self.GRIP_TIME = 1.2
        self.plangraph = PlanGraph(args)
        self.REVERT_LIST = {"grip":"open","open":"grip"}
        self.executing = False
        self.hand_getting_object = False
        if args.stageI_done:
            self.intention_list = ["get_connectors"]*4

    def receive_pose(self,data):
        pos_x = data.pose.position.x
        pos_y = data.pose.position.y
        pos_z = data.pose.position.z
        ort_w = data.pose.orientation.w
        ort_x = data.pose.orientation.x
        ort_y = data.pose.orientation.y
        ort_z = data.pose.orientation.z
        current_pose = (pos_x,pos_y,pos_z,ort_w,ort_x,ort_y,ort_z)
        self.current_pose = current_pose

    def receive_data(self,data):
        print(data.data)
        split_data = data.data.split('_')[0]

        if ',' in data.data: # debug action
            action = [data.data.split(',')[0],int(data.data.split(',')[1])]
            if action[0] and not self.executing:
                print(action)
                self.execute_action(action)

        elif data.data in INTENTION_LIST:
            action = self.decide_send_action(data.data)
            if action[0] and not self.executing:
                print(action)
                self.intention_list.append(data.data)
                self.execute_action(action)
        elif split_data in COMMAND_LIST and data.data not in self.command_list:
            self.command_list.append(data.data)
            self.command = split_data
            self.old_command_data = data.data
            # print(data.data)
            if self.command == "stop" and self.executing:
                try:
                    self.intention_list.pop(-1)
                except:
                    pass
            if not self.executing and self.command:
                old_command = self.command
                if self.command == "short":
                    if self.plangraph.stage == None and len(self.plangraph.stage_record["four_tubes"]) < 4 and len(self.plangraph.stage_record["bottom"]) == 4:
                        self.plangraph.stage = "four_tubes"
                        get_num = 4 - len(self.plangraph.stage_record["four_tubes"])
                        for _ in range(get_num):
                            action = self.decide_send_action(old_command)
                            if action[0]:
                                print(action)
                                self.command = None
                                self.execute_action(action)
                    else:
                        action = self.decide_send_action(self.command)
                        if action[0]:
                            print(action)
                            self.command = None
                            self.execute_action(action)
                else:
                    action = self.decide_send_action(self.command)
                    if action[0]:
                        print(action)
                        self.command = None
                        self.execute_action(action)
        
    def execute_action(self,action,index=0):
        print(self.plangraph.stage)
        waypoints_list,target_list,unique_actions = self.generate_waypoints(action)
        kinova_control_msg = PoseStamped()
        kinova_control_msg.pose = ComposePoseFromTransQuat(waypoints_list[index])
        kinova_control_pub.publish(kinova_control_msg)
        i = index
        if len(waypoints_list) == 0: return
        self.executing = True
        if i in unique_actions.keys():
            # time.sleep(0.2)
            self.execute_unique_actions(unique_actions[i])
        while i < len(waypoints_list):
            current_pose = self.current_pose
            command = self.command
            # pdb.set_trace()
            if command == "stop":
                DELIVER_INDEX = len(waypoints_list) - 2
                if i < DELIVER_INDEX:
                    print(command)
                    self.retract(current_pose,target_list,unique_actions,i)
                    self.command = None
                    # kinova_control_msg.pose = ComposePoseFromTransQuat(current_pose)
                    # kinova_control_pub.publish(kinova_control_msg)
                    break
            if command in ["long","short"] and command not in action:
                DELIVER_INDEX = len(waypoints_list) - 2
                if i < DELIVER_INDEX:
                    print(command)
                    action = self.decide_send_action(command)
                    if action:
                        self.command = None
                        BASE_INDEX = 1
                        self.return_base(current_pose,target_list,unique_actions,i,BASE_INDEX)
                        self.execute_action(action,BASE_INDEX+1)
                        break

            if self.reached(current_pose,target_list[i]):
                i += 1
                if i in unique_actions.keys():
                    kinova_control_msg.pose = ComposePoseFromTransQuat(target_list[i-1])
                    kinova_control_pub.publish(kinova_control_msg)
                    time.sleep(0.5)
                    self.execute_unique_actions(unique_actions[i])
                if i < len(waypoints_list):
                    kinova_control_msg.pose = ComposePoseFromTransQuat(waypoints_list[i])
                    kinova_control_pub.publish(kinova_control_msg)
                else: # end
                    print("end")
                    kinova_control_msg.pose = ComposePoseFromTransQuat(target_list[i-1])
                    kinova_control_pub.publish(kinova_control_msg)
                    if "short" in action[0]:
                        self.plangraph.tube_count["short"] += 1
                    elif "long" in action[0]:
                        self.plangraph.tube_count["long"] += 1
                    self.plangraph.action_history.append(action[0])
                    if self.plangraph.stage:
                        self.plangraph.stage_record[self.plangraph.stage].append(action[0])
                        if len(self.plangraph.stage_record[self.plangraph.stage]) == 4:
                            self.plangraph.stage_history.append(self.plangraph.stage)
                            self.plangraph.stage = None
        self.executing = False
        return
    
    def execute_unique_actions(self,unique_actions):
        for i in range(len(unique_actions)):
            print(unique_actions[i])
            if unique_actions[i] == "grip":
                kinova_grip_msg = Float64MultiArray()
                kinova_grip_msg.data = [0] # 0 for close, 1 for open
                kinova_grip_pub.publish(kinova_grip_msg)
                time.sleep(self.GRIP_TIME)
            elif "wait" in unique_actions[i]:
                wait_time = int(unique_actions[i][4:])
                time.sleep(wait_time)
            elif unique_actions[i] == "open":
                kinova_grip_msg = Float64MultiArray()
                kinova_grip_msg.data = [1] # 0 for close, 1 for open
                kinova_grip_pub.publish(kinova_grip_msg) 
                time.sleep(self.GRIP_TIME)
            elif unique_actions[i] == "force_triggered":
                t1 = time.time()
                old_force = self.force
                # pdb.set_trace()
                while (time.time()-t1) < 15:
                    if (abs(self.force[0]-old_force[0])+abs(self.force[1]-old_force[1])+abs(self.force[2]-old_force[2])) > 15:
                    # if self.hand_getting_object:
                        kinova_grip_msg = Float64MultiArray()
                        kinova_grip_msg.data = [1] # 0 for close, 1 for open
                        kinova_grip_pub.publish(kinova_grip_msg) 
                        time.sleep(self.GRIP_TIME)
                        break
                kinova_grip_msg = Float64MultiArray()
                kinova_grip_msg.data = [1] # 0 for close, 1 for open
                kinova_grip_pub.publish(kinova_grip_msg) 
                time.sleep(self.GRIP_TIME)
            else:
                print(f"Invalid unique actions[{i}]!")

    def hand_get(self,data):
        force = data.data
        self.force = force
        # FORCE_DEFAULT = (6.6,-3,-13)
        # if (abs(force[0]-FORCE_DEFAULT[0])+abs(force[1]-FORCE_DEFAULT[1])+abs(force[2]-FORCE_DEFAULT[2])) > 3:
        #     self.hand_getting_object = True

    def retract(self,current_pose,ori_targets,ori_unique_actions,index):
        print("retract")
        kinova_control_msg = PoseStamped()
        kinova_control_msg.pose = ComposePoseFromTransQuat(current_pose)
        kinova_control_pub.publish(kinova_control_msg)
        # revert actions
        target_list = ori_targets[:index][::-1]
        waypoints_list = []
        for i in range(len(target_list)):
            if i == 0:
                tmp = []
                for j in range(len(target_list[i])):
                    tmp.append(target_list[i][j]+0.7*(target_list[i][j]-current_pose[j]))
                waypoints_list.append(tmp)
            if i > 0:
                tmp = []
                for j in range(len(target_list[i])):
                    tmp.append(target_list[i][j]+0.7*(target_list[i][j]-target_list[i-1][j]))
                waypoints_list.append(tmp)
        
        # revert unique actions
        unique_actions = {}
        for unique_index in ori_unique_actions.keys():
            if unique_index <= index:
                unique_actions[index-unique_index] = []
                for unique_action in ori_unique_actions[unique_index]:
                    if "wait" not in unique_action and "force_triggered" not in unique_action:
                        # pdb.set_trace()
                        unique_actions[index-unique_index].append(self.REVERT_LIST[unique_action])

        # pdb.set_trace()
        i = 0
        kinova_control_msg = PoseStamped()
        kinova_control_msg.pose = ComposePoseFromTransQuat(waypoints_list[0])
        kinova_control_pub.publish(kinova_control_msg)
        # pdb.set_trace()
        if i in unique_actions.keys():
            kinova_control_msg.pose = ComposePoseFromTransQuat(target_list[0])
            kinova_control_pub.publish(kinova_control_msg)
            time.sleep(0.5)
            self.execute_unique_actions(unique_actions[i])
        while i < len(waypoints_list):
            current_pose = self.current_pose
            if self.reached(current_pose,target_list[i]):
                i += 1
                if i in unique_actions.keys():
                    kinova_control_msg.pose = ComposePoseFromTransQuat(target_list[i-1])
                    kinova_control_pub.publish(kinova_control_msg)
                    time.sleep(0.5)
                    self.execute_unique_actions(unique_actions[i])
                if i < len(waypoints_list):
                    kinova_control_msg.pose = ComposePoseFromTransQuat(waypoints_list[i])
                    kinova_control_pub.publish(kinova_control_msg)
                else:
                    kinova_control_msg.pose = ComposePoseFromTransQuat(target_list[i-1])
                    kinova_control_pub.publish(kinova_control_msg)
        return
    
    def return_base(self,current_pose,ori_targets,ori_unique_actions,index,base_index):
        print("return base")
        kinova_control_msg = PoseStamped()
        kinova_control_msg.pose = ComposePoseFromTransQuat(current_pose)
        kinova_control_pub.publish(kinova_control_msg)
        # revert actions
        if index <= base_index: # make sure that the robot indeed get to base position
            target_list = ori_targets[index:base_index+1]
        else:
            target_list = ori_targets[base_index:index][::-1]
        waypoints_list = []
        for i in range(len(target_list)):
            if i == 0:
                tmp = []
                for j in range(len(target_list[i])):
                    tmp.append(target_list[i][j]+0.5*(target_list[i][j]-current_pose[j]))
                waypoints_list.append(tmp)
            if i > 0:
                tmp = []
                for j in range(len(target_list[i])):
                    tmp.append(target_list[i][j]+0.5*(target_list[i][j]-target_list[i-1][j]))
                waypoints_list.append(tmp)
        
        # revert unique actions
        unique_actions = {}
        for unique_index in ori_unique_actions.keys():
            if unique_index <= index:
                unique_actions[index-unique_index] = []
                for unique_action in ori_unique_actions[unique_index]:
                    if "wait" not in unique_action and "force_triggered" not in unique_action:
                        # pdb.set_trace()
                        unique_actions[index-unique_index].append(self.REVERT_LIST[unique_action])

        # pdb.set_trace()
        i = 0
        kinova_control_msg = PoseStamped()
        kinova_control_msg.pose = ComposePoseFromTransQuat(waypoints_list[0])
        kinova_control_pub.publish(kinova_control_msg)
        # pdb.set_trace()
        if i in unique_actions.keys():
            kinova_control_msg.pose = ComposePoseFromTransQuat(target_list[0])
            kinova_control_pub.publish(kinova_control_msg)
            time.sleep(0.5)
            self.execute_unique_actions(unique_actions[i])
        while i < len(waypoints_list):
            current_pose = self.current_pose
            if self.reached(current_pose,target_list[i]):
                i += 1
                if i in unique_actions.keys():
                    kinova_control_msg.pose = ComposePoseFromTransQuat(target_list[i-1])
                    kinova_control_pub.publish(kinova_control_msg)
                    time.sleep(0.5)
                    self.execute_unique_actions(unique_actions[i])
                if i < len(waypoints_list):
                    kinova_control_msg.pose = ComposePoseFromTransQuat(waypoints_list[i])
                    kinova_control_pub.publish(kinova_control_msg)
                else:
                    kinova_control_msg.pose = ComposePoseFromTransQuat(target_list[i-1])
                    kinova_control_pub.publish(kinova_control_msg)
        return

    def generate_waypoints(self,action):
        waypoints_list = []
        if action[0] == "get_short_tubes":
            retract = RETRACT_POSITION
            ready = (0.2,0.32,0.25,0,-0.7,-0.7,0)
            x_interval = 0.10
            y_interval = 0.16
            row = action[1]%2
            col = action[1]//2
            base = BASE
            get = (-0.28-x_interval*col,0.28-y_interval*row,0.2,0,-0.7,-0.7,0)
            grip = (-0.28-x_interval*col,0.28-y_interval*row,-0.048,0,-0.7,-0.7,0)
            deliver = (0.3,0.3,0.25,0,-0.7,-0.6,-0.2) # TODO: move with hand
            # target_list = [retract,ready,base,get,grip,get,base,ready,deliver,ready,retract]
            # unique_actions = {5:["grip"],9:["force_triggered"]}
            target_list = [ready,base,get,grip,get,base,ready,deliver,ready]
            unique_actions = {target_list.index(grip)+1:["grip"],target_list.index(deliver)+1:["force_triggered"]}
            speed = [0.7] * len(target_list)
            speed[-1] = 0

        if action[0] == "get_long_tubes":
            retract = RETRACT_POSITION
            ready = (0.2,0.32,0.35,0,-0.7,-0.7,0)
            # ready_way = (0.2,0.32+0.18,0.19,0,-0.7,-0.7,0)
            x_interval = 0.10
            col = action[1]
            base = BASE
            get = (-0.28-x_interval*col,-0.04,0.3,0,-0.7,-0.7,0)
            grip = (-0.28-x_interval*col,-0.04,0.15,0,-0.7,-0.7,0)
            base_long = (-0.18,-0.04,0.3,0,-0.7,-0.7,0)
            deliver = (0.3,0.3,0.35,0,-0.7,-0.6,-0.2) # TODO: move with hand
            # target_list = [retract,ready,base,get,grip,get,base_long,base,ready,deliver,ready,retract]
            # unique_actions = {5:["grip"],10:["force_triggered"]}
            target_list = [ready,base,get,grip,get,base_long,base,ready,deliver,ready]
            unique_actions = {target_list.index(grip)+1:["grip"],target_list.index(deliver)+1:["force_triggered"]}
            speed = [0.7] * len(target_list)
            speed[-1] = 0

        if action[0] == "spin_bottom":
            retract = RETRACT_POSITION
            pre_ready = (0.58,0.25,0.19,0,-0.7,-0.7,0)
            ready = (0.58,0.25,0.19,0,0,1,0)
            grip = (0.58,0.25,-0.085,0,0,1,0)
            middle_spin = (0.48,0.06,-0.085,0,1,1,0)
            spin = (0.4,0.27,-0.085,0,1,0,0)
            spin_final = (0.5,0.27,-0.085,0,1,-1,0)
            grip_final = (.48,0.06,-0.085,0,1,-1,0)
            up = (0.5,0.27,0.19,0,1,-1,0)
            back = (0.48,0.05,0.19,0,1,0,0)
            if action[1] % 4 == 0:
                target_list = [ready,grip]
                unique_actions = {target_list.index(grip)+1:["grip"]}
                speed = [0.7] * len(target_list)
            elif action[1] % 4 == 1:
                target_list = [middle_spin]
                unique_actions = {}
                speed = [0.7] * len(target_list)
                speed[-1] = 0
            elif action[1] % 4 == 2:
                target_list = [spin]
                unique_actions = {}
                speed = [0.7] * len(target_list)
                speed[-1] = 0
            elif action[1] % 4 == 3:
                target_list = [spin_final,back,retract]
                unique_actions = {target_list.index(spin_final)+1:["wait10","open"]}
                speed = [0.7] * len(target_list)  
                speed[-2] = 0
                speed[-1] = 0

        if action[0] == "spin_four_tubes":
            # self.plangraph.stage_history = ["bottom","four_tubes","top"] # for debugging
            self.plangraph.stage_history = ["bottom","top","four_tubes"] # for debugging
            assert len(self.plangraph.stage_history) > 2, "spin_four_tubes should be executed after three stages have all been done!"
            retract = RETRACT_POSITION
            pre_ready = (0.58,0.25,0.19,0,-0.7,-0.7,0)
            ready = (0.58,0.25,0.19,0,0,1,0)
            grip = (0.58,0.25,0.11,0,0,1,0)
            middle_spin = (0.48,0.06,0.11,0,1,1,0)
            spin = (0.4,0.27,0.11,0,1,0,0)
            up = (0.4,0.27,0.19,0,1,0,0)
            back = (0.48,0.05,0.19,0,1,0,0)
            if action[1] == 0:
                target_list = [ready,grip]
                unique_actions = {target_list.index(grip)+1:["grip"]}
                speed = [0.7] * len(target_list)
            elif action[1] == 1:
                target_list = [middle_spin,spin,up,back,retract]
                unique_actions = {target_list.index(spin)+1:["wait20","open"]}
                speed = [0.7] * len(target_list)
                speed[1] = 0   
                speed[-1] = 0          

        if action[0] == "spin_top":
            retract = RETRACT_POSITION
            pre_ready = (0.58,0.25,0.19,0,-0.7,-0.7,0)
            ready = (0.58,0.25,0.19,0,0,1,0)
            grip = (0.58,0.25,0.09,0,0,1,0)
            middle_spin = (0.48,0.06,0.09,0,1,1,0)
            spin = (0.4,0.27,0.09,0,1,0,0)
            spin_final = (0.5,0.27,0.09,0,1,-1,0)
            grip_final = (0.48,0.06,0.09,0,1,-1,0)
            up = (0.5,0.27,0.19,0,1,-1,0)
            back = (0.48,0.05,0.19,0,1,0,0)
            if action[1] % 4 == 0:
                target_list = [ready,grip]
                unique_actions = {target_list.index(grip)+1:["grip"]}
                speed = [0.7] * len(target_list)
            elif action[1] % 4 == 1:
                target_list = [middle_spin]
                unique_actions = {}
                speed = [0.7] * len(target_list)
                speed[-1] = 0
            elif action[1] % 4 == 2:
                target_list = [spin]
                unique_actions = {}
                speed = [0.7] * len(target_list)
                speed[-1] = 0
            elif action[1] % 4 == 3:
                target_list = [spin_final]
                unique_actions = {}
                speed = [0.7] * len(target_list) 
                speed[0] = 0

        if action[0] == "lift_up":
            self.plangraph.stage_history = ["bottom","four_tubes","top"] # for debugging
            # self.plangraph.stage_history = ["bottom","top","four_tubes"] # for debugging
            if self.plangraph.stage_history[-1] == "top":
                retract = RETRACT_POSITION
                up = (0.5,0.27,0.17,0,1,-1,0)
                spin = (0.48,0.06,0.18,0,1,1,0)
                ready = (0.5,0.27,0.30,0,1,1,0)
                grip = (0.5,0.27,0.09,0,1,1,0)
                up2 = (0.5,0.27,0.20,0,1,1,0)
                ready2 = (0.2,0,0.30,0,-0.7,-0.7,0)
                if action[1] == 0:
                    target_list = [up]
                    unique_actions = {}
                    speed = [0.7] * len(target_list)
                elif action[1] == 1:
                    target_list = [spin,ready,grip,up2,ready,ready2,retract]
                    unique_actions = {target_list.index(spin)+1:["open"],target_list.index(grip)+1:["grip"],target_list.index(up2)+1:["wait20","open"]}
                    speed = [0.7] * len(target_list)
                    speed[0] = 0
                    speed[-1] = 0
            elif self.plangraph.stage_history[-1] == "four_tubes":
                retract = RETRACT_POSITION
                init_spin = (0.5,0.27,0.09,0,1,-1,0)
                up = (0.5,0.27,0.17,0,1,-1,0)
                spin = (0.48,0.06,0.17,0,1,1,0)
                ready = (0.5,0.27,0.30,0,1,1,0)
                grip = (0.5,0.27,0.09,0,1,1,0)
                up2 = (0.5,0.27,0.20,0,1,1,0)
                ready2 = (0.2,0,0.30,0,-0.7,-0.7,0)
                if action[1] == 0:
                    target_list = [init_spin,up]
                    unique_actions = {}
                    speed = [0.7] * len(target_list)
                    speed[0] = 0
                elif action[1] == 1:
                    target_list = [spin,ready,grip,up2,ready,ready2,retract]
                    unique_actions = {target_list.index(spin)+1:["open"],target_list.index(grip)+1:["grip"],target_list.index(up2)+1:["wait20","open"]}
                    speed = [0.7] * len(target_list)
                    speed[0] = 0
                    speed[-1] = 0

        waypoints_list = []
        for i in range(len(target_list)):
            if i == 0:
                waypoints_list.append(target_list[i])
            else:
                tmp = []
                for j in range(len(target_list[i])):
                    if j < 3: # only adjust position points
                        tmp.append(target_list[i][j]+speed[i]*(target_list[i][j]-target_list[i-1][j]))
                    else: # keep augular points stay as target points
                        tmp.append(target_list[i][j])
                waypoints_list.append(tmp)
        return waypoints_list,target_list,unique_actions

    def get_command(self):
        # TODO: NOT USED
        if len(self.command_list) > 0:
            return self.command_list[-1]
        return ""

    def decide_send_action(self,data):
        # TODO
        # for key in self.plangraph.tube_count.keys(): # tube sum check
        #     assert self.plangraph.tube_count[key] < self.plangraph.TUBE_SUM[key]
        REVERT_TUBE = {"get_short_tubes":"long","get_long_tubes":"short"}
        if data == "get_connectors": # decide to get short tubes or get long tubes
            # for stage in ["bottom","top"]:
            #     if len(self.plangraph.stage_record[stage]) < 4 and self.plangraph.stage != "four_tubes":
            if len(self.plangraph.stage_record["bottom"]) < 4:
                stage = "bottom"
            elif (self.intention_list.count(data) >= 4 or "spin_bottom" in self.plangraph.action_history) and self.plangraph.stage != "four_tubes" and len(self.plangraph.stage_record["top"]) < 4:
                stage = "top"
            else:
                return None,None
            self.plangraph.stage = stage
            for key in REVERT_TUBE.keys():
                if self.plangraph.stage_record[stage].count(key) == 2:
                    return [f"get_{REVERT_TUBE[key]}_tubes",self.plangraph.tube_count[REVERT_TUBE[key]]]
            if len(self.plangraph.stage_record[stage]) > 0:
                last_tube = self.plangraph.stage_record[stage][-1]
                tube_key = REVERT_TUBE[last_tube]
            else:
                tube_key = "short"
            return [f"get_{tube_key}_tubes",self.plangraph.tube_count[tube_key]]

        if data == "get_wheels" and (self.plangraph.action_history.count("spin_bottom") + self.plangraph.action_history.count("spin_four_tubes") + self.plangraph.action_history.count("spin_top")) < 8:
            data = "get_screws"
        # self.plangraph.stage_history = ["bottom","four_tubes","top"] # for debugging

        if data == "get_screws" and self.plangraph.stage_history:
            if (self.plangraph.action_history.count("spin_bottom") + self.plangraph.action_history.count("spin_four_tubes") + self.plangraph.action_history.count("spin_top")) >= 10:
                data = "get_wheels"
            elif self.plangraph.stage_history[-1] == "bottom":
                stage = "bottom"
                if self.plangraph.screw_count[stage] < 4:
                    self.plangraph.screw_count[stage] += 1
                    return ["spin_bottom",self.plangraph.action_history.count("spin_bottom")]
            elif self.plangraph.stage_history[-1] == "four_tubes":
                stage = "four_tubes"
                if self.plangraph.screw_count[stage] < 4 and len(self.plangraph.stage_history) == 3:
                    self.plangraph.screw_count[stage] += 1
                    if self.plangraph.screw_count[stage] == 1 or self.plangraph.screw_count[stage] == 3:
                        return ["spin_four_tubes",self.plangraph.action_history.count("spin_four_tubes")]
                # elif len(self.plangraph.stage_history) == 2 and self.plangraph.screw_count[stage] == 3:
                #     self.plangraph.screw_count[stage] = 1
                #     return ["spin_four_tubes",self.plangraph.action_history.count("spin_four_tubes")]
            elif self.plangraph.stage_history[-1] == "top":
                if len(self.plangraph.stage_history) == 2:
                    stage = "top"
                    if self.plangraph.screw_count[stage] < 4:
                        self.plangraph.screw_count[stage] += 1
                        return ["spin_bottom",self.plangraph.action_history.count("spin_bottom")]
                elif len(self.plangraph.stage_history) == 3:
                    stage = "top"
                    if self.plangraph.screw_count[stage] < 8:
                        self.plangraph.screw_count[stage] += 1
                        if self.plangraph.screw_count[stage] % 4 == 1 or self.plangraph.screw_count[stage] % 4 == 3:
                            return ["spin_top",self.plangraph.action_history.count("spin_top")]

        if data == "get_wheels" and self.plangraph.stage_history:
            self.plangraph.wheels_count += 1
            if self.plangraph.wheels_count == 1 or self.plangraph.wheels_count == 3:
                return ["lift_up",self.plangraph.action_history.count("lift_up")]

        if data == "get up" and self.plangraph.stage_history:
            self.plangraph.wheels_count += 1
            return ["lift_up",self.plangraph.action_history.count("lift_up")]

        if data == "spin" and self.plangraph.stage_history:
            if self.plangraph.stage_history[-1] == "bottom":
                stage = "bottom"
                self.plangraph.screw_count[stage] = 1
                return [f"spin_{stage}",self.plangraph.action_history.count(f"spin_{stage}")]
            elif self.plangraph.stage_history[-1] == "four_tubes":
                stage = "four_tubes"
                self.plangraph.screw_count[stage] = 1
                return [f"spin_{stage}",self.plangraph.action_history.count(f"spin_{stage}")]
            elif self.plangraph.stage_history[-1] == "top":
                stage = "top"
                self.plangraph.screw_count[stage] = 1
                if len(self.plangraph.stage_history) == 2:
                    return ["spin_bottom",self.plangraph.action_history.count("spin_bottom")]
                elif len(self.plangraph.stage_history) == 3:
                    return ["spin_top",self.plangraph.action_history.count("spin_top")]

        if data == "short":
            if self.plangraph.tube_count["short"] < self.plangraph.TUBE_SUM["short"]:
                return ["get_short_tubes",self.plangraph.tube_count["short"]]
        if data == "long":
            if self.plangraph.tube_count["long"] < self.plangraph.TUBE_SUM["long"]:
                return ["get_long_tubes",self.plangraph.tube_count["long"]]
        return None,None
        # return ["",int]

    def reached(self,current_pose,target_pose):
        pos_gap = abs(current_pose[0]-target_pose[0])+abs(current_pose[1]-target_pose[1])+abs(current_pose[2]-target_pose[2])
        if pos_gap > 0.05:
            # print(f"pos_gap:{pos_gap}")
            return False
        else:
            w,x,y,z = target_pose[3],target_pose[4],target_pose[5],target_pose[6]
            sqrt_sum = (w**2+x**2+y**2+z**2)**0.5
            w,x,y,z = w/sqrt_sum,x/sqrt_sum,y/sqrt_sum,z/sqrt_sum
            quat_gap = 1 - (w*current_pose[3]+x*current_pose[4]+y*current_pose[5]+z*current_pose[6])**2
            # print(quat_gap)
            if quat_gap < 0.05:
                # print(f"quat_gap:{quat_gap}")
                return True
            else:
                # print(f"quat_gap:{quat_gap}")
                return False
            
    
def listener(args):

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    receiver = Receiver(args)
    rospy.init_node('listener', anonymous=True)

    # rospy.Subscriber("chatter", String, controller)
    rospy.Subscriber("chatter", String, receiver.receive_data)
    rospy.Subscriber("/kinova/pose_tool_in_base", PoseStamped, receiver.receive_pose, queue_size=1)
    rospy.Subscriber("/kinova/tool_force",Float64MultiArray,receiver.hand_get)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stageI_done', action="store_true",
                    help="whether to set stage I as done")
    parser.add_argument('--user_spin', action="store_true",
                    help="user study on spin")
    args = parser.parse_args()
    listener(args)
    # print("ok")