import argparse
import os
import torch
from pathlib import Path
FILE_DIR = Path(__file__).resolve().parent
import numpy as np
import cv2
from torch.nn.functional import softmax
from torch.distributions import Categorical
import pdb

from DLinear import Model_FinalIntention,Model_FinalTraj
from Dataset import INTENTION_LIST

SKELETON_LIST = {
"left_shoulder": 0,
"right_shoulder": 1,
"left_elbow": 2,
"right_elbow": 3,
"left_wrist": 4,
"right_wrist": 5,
"left_pinky": 6,
"right_pinky": 7,
"left_index": 8,
"right_index": 9,
"left_thumb": 10,
"right_thumb": 11,
"left_hip": 12,
"right_hip": 13,
"nose": 14
}

class Args:
    def __init__(self,**kwargs):
        self.default = {"seq_len":5,"pred_len":5,"class_num":4,"individual":False,"channels":15*3,"half_body":False,"epochs":40,"input_type":"pkl"}
        for key in ('seq_len','pred_len', 'class_num', 'individual', 'channels','half_body','epochs','input_type'):
            if key in kwargs and\
                kwargs[key]:
                    setattr(self, key, kwargs[key])
            else:
                setattr(self, key,self.default[key])
        

class IntentionPredictor:
    def __init__(self,ckpt_path=None,model_type="final_intention",no_mask=False,filter_type=None,**kwargs): 
        args = Args(**kwargs)
        if model_type == "final_intention":
            self.model = Model_FinalIntention(args)
        elif model_type == "final_traj":
            self.model = Model_FinalTraj(args)
        if ckpt_path:
            checkpoint = torch.load(ckpt_path)
        else: # default
            if filter_type:
                checkpoint = torch.load(f'{FILE_DIR}/checkpoints/seq{args.seq_len}_pred{args.pred_len}_epoch{args.epochs}_whole_{args.input_type}_{model_type}_nomask{no_mask}_filter{filter_type}.pth')
            else:
                checkpoint = torch.load(f'{FILE_DIR}/checkpoints/seq{args.seq_len}_pred{args.pred_len}_epoch{args.epochs}_whole_{args.input_type}_{model_type}_nomask{no_mask}.pth')
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self,poses,restrict,poses_world=None):
        assert poses.shape[0] == 1
        pred_traj,pred_intention = self.model(poses)
        batch_intention = torch.argmax(pred_intention,1)

        # count = [0,0,0]
        # sum_ = [0,0,0]
        working_area_flag = False
        ood_flag = False
        if restrict == "working_area":
            working_area_flag = True
        elif restrict == "ood":
            ood_flag = True
        elif restrict == "all":
            working_area_flag = True
            ood_flag = True
        elif restrict == "no":
            pass
        else:
            print("restrict invalid!")
            return

        if ood_flag:
            entropy = Categorical(probs = softmax(pred_intention,dim=1)[0].detach()).entropy()
            if entropy > 0.4 and torch.argmax(pred_intention) != 3:
                batch_intention[0] = INTENTION_LIST["no_action"]
            elif entropy > 0.5 and torch.argmax(pred_intention) == 3:
                batch_intention[0] = INTENTION_LIST["no_action"]

        # if working_area_flag:
        #     intention = batch_intention[0]
        #     if intention != INTENTION_LIST["no_action"]:
        #         concate_traj = torch.cat((poses,pred_traj),1).view(poses.shape[0],-1,poses.shape[-1]//3,3)
        #         # concate_traj = poses.view(poses.shape[0],-1,poses.shape[-1]//3,3)
        #         if intention == INTENTION_LIST["get_connectors"]:
        #             index = SKELETON_LIST["right_wrist"]
        #             right_wrist_traj = concate_traj[:,:,index,0]
        #             # count[0] += right_wrist_traj.mean().item()
        #             # sum_[0] += 1
        #             if max(right_wrist_traj[0]) < -1: # TODO
        #                 print("modifying connectors!")
        #                 batch_intention[0] = INTENTION_LIST["no_action"]
        #         elif intention == INTENTION_LIST["get_screws"]:
        #             index1 = SKELETON_LIST["right_wrist"]
        #             index2 = SKELETON_LIST["left_wrist"]
        #             right_wrist_traj = concate_traj[:,:,index1,0]
        #             left_wrist_traj = concate_traj[:,:,index2,0]
        #             right_wrist_traj_z = concate_traj[:,:,index1,2]
        #             # count[1] += right_wrist_traj.mean().item()
        #             # sum_[1] += 1
        #             if min(right_wrist_traj[0]) > -1.33 and min(left_wrist_traj[0]) > -1.33: # TODO
        #                 batch_intention[0] = INTENTION_LIST["no_action"]
        #                 print("modifying screws!")
        #                 # pdb.set_trace()
        #             # if max(right_wrist_traj_z[0]) < 0.65: # TODO
        #             #     batch_intention[0] = INTENTION_LIST["get_wheels"]
        #             #     print("modifying screws to wheels!")
        #         elif intention == INTENTION_LIST["get_wheels"]:
        #             index = SKELETON_LIST["right_wrist"]
        #             right_wrist_traj_x = concate_traj[:,:,index,0]
        #             right_wrist_traj_z = concate_traj[:,:,index,2]
        #             # count[2] += right_wrist_traj_y.mean().item()
        #             # sum_[2] += 1
        #             # print(right_wrist_traj_z)
        #             if min(right_wrist_traj_x[0]) > -1.33: # TODO
        #                 batch_intention[0] = INTENTION_LIST["no_action"]
        #                 print("modifying wheels!")
        #                 # pdb.set_trace()
        #             # elif min(right_wrist_traj_z[0]) > 0.8:
        #             #     batch_intention[0] = INTENTION_LIST["get_screws"]
        #             #     print("modifying wheels to screws!")

        if working_area_flag:
            intention = batch_intention[0]
            if intention != INTENTION_LIST["no_action"]:
                if intention == INTENTION_LIST["get_connectors"]:
                    index = 16
                    right_wrist_traj = poses_world[:,:,index,0]
                    if right_wrist_traj[0][-1] > -22: # TODO
                        print("modifying connectors!")
                        batch_intention[0] = INTENTION_LIST["no_action"]
                elif intention == INTENTION_LIST["get_screws"]:
                    index1 = 16
                    index2 = 15
                    right_wrist_traj = poses_world[:,:,index1,0]
                    left_wrist_traj = poses_world[:,:,index2,0]
                    right_wrist_traj_z = poses_world[:,:,index1,2]
                    # count[1] += right_wrist_traj.mean().item()
                    # sum_[1] += 1
                    if right_wrist_traj[0][-1] < 70 and left_wrist_traj[0][-1] < 70: # TODO
                        batch_intention[0] = INTENTION_LIST["no_action"]
                        print("modifying screws!")
                        # pdb.set_trace()
                    # if max(right_wrist_traj_z[0]) < 0.65: # TODO
                    #     batch_intention[0] = INTENTION_LIST["get_wheels"]
                    #     print("modifying screws to wheels!")
                elif intention == INTENTION_LIST["get_wheels"]:
                    index1 = 16
                    index2 = 15
                    right_wrist_traj = poses_world[:,:,index1,0]
                    left_wrist_traj = poses_world[:,:,index2,0]
                    right_wrist_traj_z = poses_world[:,:,index1,2]
                    # count[2] += right_wrist_traj_y.mean().item()
                    # sum_[2] += 1
                    # print(right_wrist_traj_z)
                    if right_wrist_traj[0][-1] < 65 and left_wrist_traj[0][-1] < 65: # TODO
                        batch_intention[0] = INTENTION_LIST["no_action"]
                        print("modifying wheels!")
                        # pdb.set_trace()
                    # elif min(right_wrist_traj_z[0]) > 0.8:
                    #     batch_intention[0] = INTENTION_LIST["get_screws"]
                    #     print("modifying wheels to screws!")

        return pred_traj,batch_intention

if __name__ == '__main__':
    predictor = IntentionPredictor()
    pdb.set_trace()
        
