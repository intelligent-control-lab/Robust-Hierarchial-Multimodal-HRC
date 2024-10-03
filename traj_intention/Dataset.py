import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import argparse
import pickle
from pathlib import Path
import os,sys
from Filter import Smooth_Filter
sys.path.append('./depthai_blazepose')
import mediapipe_utils as mpu
FILE_DIR = Path(__file__).resolve().parent
import pdb

INTENTION_LIST = {"no_action":0, "get_connectors":1,"get_screws":2,"get_wheels":3}

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by 四元数quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = np.cross(qvec, v, len(q.shape) - 1)
    uuv = np.cross(qvec, uv, len(q.shape) - 1)
    return (v + 2 * (q[..., :1] * uv + uuv))


def camera_to_world(X, R= np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32),\
                     t=0):
    return qrot(np.tile(R, (*X.shape[:-1], 1)), X) + t

class MyDataset(Dataset):
    def __init__(self, json_file, root_dir, args,dataset_type="train",transform=None,world_pose=False):
        self.json_file = json_file
        self.transform = transform
        self.root_dir = root_dir
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.channels = args.channels
        self.type = dataset_type
        self.test_whole = args.test_whole
        self.half_body = args.half_body
        self.intention_list = INTENTION_LIST
        self.input_type = args.input_type
        self.filter_type = args.filter_type
        self.world_pose = world_pose
        if args.filter_type:
            self.filter = Smooth_Filter(args.filter_type)
        self.data,self.data_world,self.weights = self.process_json()

    def process_json(self):
        with open(self.json_file, 'r') as file:
            data_cut_points = json.load(file)
        data_cut_points = data_cut_points[self.type] # TODO
        data = []
        data_world = []
        weights = [0]*len(self.intention_list)
        for intention in data_cut_points.keys():
            if intention != "no_action":
                intention_tasks = data_cut_points[intention]
                intention_label = self.intention_list[intention]
                for task in intention_tasks.keys():
                    if not self.test_whole and 'build_cars' in task:
                        continue
                    points = intention_tasks[task]

                    # process data
                    if self.input_type == "npy":
                        npy_file = np.load(f'{self.root_dir}/{task[:-3]}/{task}.npy')
                        # pdb.set_trace()
                        if self.type == "test_self":
                            npy_file = np.concatenate((npy_file[:,11:25,:],npy_file[:,0:1,:]),axis=1)
                        if self.half_body:
                            npy_file = np.concatenate((npy_file[:,1::2,:],npy_file[:,0:1,:],npy_file[:,-1:,:],npy_file[:,-3:-2,:]),axis=1)
                    elif self.input_type == "pkl":
                        pic = open(f'{self.root_dir}/{task[:-3]}/{task}.pkl','rb')
                        pkl_file = pickle.load(pic)
                        poses = []
                        for body in pkl_file:
                            poses.append(body.landmarks)
                        npy_file = np.array(poses)
                        if npy_file.shape[1] == 33:
                            npy_file = np.concatenate((npy_file[:,11:25,:],npy_file[:,0:1,:]),axis=1)

                        if self.world_pose:
                            pose_world = []
                            for body in pkl_file:
                                pose_world.append(body.xyz/10 + body.landmarks_world*100)
                            npy_world = np.array(pose_world)

                        # npy_file = 2*(npy_file-npy_file.min())/(npy_file.max()-npy_file.min())
                        # npy_file = camera_to_world(npy_file)
                        # npy_file[:, :, 2] -= np.min(npy_file[:, :, 2])
                        # pdb.set_trace()
                    
                    assert len(points['start']) == len(points['end']), f"The number of start points and end points doesn't match in {task}!"
                    task_data = []
                    task_world = []
                    for i in range(len(points['end'])):
                        end_index = points['end'][i] - 1
                        start_index = points['start'][i] - 1
                        for j in range(start_index,end_index-self.pred_len-self.seq_len+1):
                            posInputs = npy_file[j:j+self.seq_len]
                            posInputs = 2*(posInputs-posInputs.min())/(posInputs.max()-posInputs.min())
                            posInputs = camera_to_world(posInputs)
                            posInputs[:, :, 2] -= np.min(posInputs[:, :, 2])
                            if self.filter_type:
                                posInputs = self.filter.smooth_multi_trajectories(posInputs)
                            posInputs = torch.from_numpy(posInputs.reshape(self.seq_len,self.channels)).float()
                            posOutputs = npy_file[j+self.seq_len:j+self.seq_len+self.pred_len]
                            posOutputs = 2*(posOutputs-posOutputs.min())/(posOutputs.max()-posOutputs.min())
                            posOutputs = camera_to_world(posOutputs)
                            posOutputs[:, :, 2] -= np.min(posOutputs[:, :, 2])
                            posOutputs = torch.from_numpy(posOutputs.reshape(self.pred_len,self.channels)).float()
                            task_data.append((posInputs,posOutputs,torch.tensor(intention_label)))
                            weights[intention_label] += 1

                            if self.world_pose:
                                world_inputs = npy_world[j:j+self.seq_len]
                                world_outputs = npy_world[j+self.seq_len:j+self.seq_len+self.pred_len]
                                task_world.append((world_inputs,world_outputs,torch.tensor(intention_label)))

                    # neg_data
                    pos_data_num = len(task_data) // len(self.intention_list) // 2
                    array = [] 
                    array.extend(np.arange(0,points['start'][0]-self.seq_len-self.pred_len))
                    for i in range(1,len(points['end'])-1):
                        end_index = points['end'][i]
                        start_index = points['start'][i+1]
                        array.extend(np.arange(end_index,start_index-self.seq_len-self.pred_len))
                    array.extend(np.arange(points['end'][-1],npy_file.shape[0]-self.seq_len-self.pred_len))
                    try:
                        random_numbers = np.random.choice(array, pos_data_num, replace=False)
                    except:
                        random_numbers = np.random.choice(array, pos_data_num, replace=True)
                    for j in random_numbers:
                        negInputs = npy_file[j:j+self.seq_len]
                        negInputs = 2*(negInputs-negInputs.min())/(negInputs.max()-negInputs.min())
                        negInputs = camera_to_world(negInputs)
                        negInputs[:, :, 2] -= np.min(negInputs[:, :, 2])
                        if self.filter_type:
                            negInputs = self.filter.smooth_multi_trajectories(negInputs)
                        negInputs = torch.from_numpy(negInputs.reshape(self.seq_len,self.channels)).float()
                        negOutputs = npy_file[j+self.seq_len:j+self.seq_len+self.pred_len]
                        negOutputs = 2*(negOutputs-negOutputs.min())/(negOutputs.max()-negOutputs.min())
                        negOutputs = camera_to_world(negOutputs)
                        negOutputs[:, :, 2] -= np.min(negOutputs[:, :, 2])
                        negOutputs = torch.from_numpy(negOutputs.reshape(self.pred_len,self.channels)).float()
                        task_data.append((negInputs,negOutputs,torch.tensor(0)))   
                        weights[0] += 1 

                        if self.world_pose:
                            world_inputs = npy_world[j:j+self.seq_len]
                            world_outputs = npy_world[j+self.seq_len:j+self.seq_len+self.pred_len]
                            task_world.append((world_inputs,world_outputs,torch.tensor(0)))
                        # pdb.set_trace()

                    data.extend(task_data)
                    data_world.extend(task_world)
            else:
                intention_tasks = data_cut_points[intention]
                intention_label = self.intention_list[intention]
                for task in intention_tasks.keys():
                    if not self.test_whole and 'build_cars' in task:
                        continue
                    points = intention_tasks[task]
                    # process data
                    if self.input_type == "npy":
                        npy_file = np.load(f'{self.root_dir}/{task[:-3]}/{task}.npy')
                        # pdb.set_trace()
                        if self.type == "test_self":
                            npy_file = np.concatenate((npy_file[:,11:25,:],npy_file[:,0:1,:]),axis=1)
                        if self.half_body:
                            npy_file = np.concatenate((npy_file[:,1::2,:],npy_file[:,0:1,:],npy_file[:,-1:,:],npy_file[:,-3:-2,:]),axis=1)
                    elif self.input_type == "pkl":
                        pic = open(f'{self.root_dir}/{task[:-3]}/{task}.pkl','rb')
                        pkl_file = pickle.load(pic)
                        poses = []
                        for body in pkl_file:
                            poses.append(body.landmarks)
                        npy_file = np.array(poses)
                        if npy_file.shape[1] == 33:
                            npy_file = np.concatenate((npy_file[:,11:25,:],npy_file[:,0:1,:]),axis=1)

                        pose_world = []
                        for body in pkl_file:
                            pose_world.append(body.xyz/10 + body.landmarks_world*100)
                        npy_world = np.array(pose_world)

                        # npy_file = 2*(npy_file-npy_file.min())/(npy_file.max()-npy_file.min())
                        # npy_file = camera_to_world(npy_file)
                        # npy_file[:, :, 2] -= np.min(npy_file[:, :, 2])
                        # pdb.set_trace()
                    
                    assert len(points['start']) == len(points['end']), f"The number of start points and end points doesn't match in {task}!"
                    task_data = []
                    for i in range(len(points['end'])):
                        end_index = points['end'][i] - 1
                        start_index = points['start'][i] - 1
                        array = [] 
                        array.extend(np.arange(start_index,end_index-self.seq_len-self.pred_len+1))
                        random_numbers = np.random.choice(array, len(array)//10, replace=False)
                        for j in random_numbers:
                            posInputs = npy_file[j:j+self.seq_len]
                            posInputs = 2*(posInputs-posInputs.min())/(posInputs.max()-posInputs.min())
                            posInputs = camera_to_world(posInputs)
                            posInputs[:, :, 2] -= np.min(posInputs[:, :, 2])
                            if self.filter_type:
                                posInputs = self.filter.smooth_multi_trajectories(posInputs)
                            posInputs = torch.from_numpy(posInputs.reshape(self.seq_len,self.channels)).float()
                            posOutputs = npy_file[j+self.seq_len:j+self.seq_len+self.pred_len]
                            posOutputs = 2*(posOutputs-posOutputs.min())/(posOutputs.max()-posOutputs.min())
                            posOutputs = camera_to_world(posOutputs)
                            posOutputs[:, :, 2] -= np.min(posOutputs[:, :, 2])
                            posOutputs = torch.from_numpy(posOutputs.reshape(self.pred_len,self.channels)).float()
                            task_data.append((posInputs,posOutputs,torch.tensor(intention_label)))
                            weights[intention_label] += 1

                            if self.world_pose:
                                world_inputs = npy_world[j:j+self.seq_len]
                                world_outputs = npy_world[j+self.seq_len:j+self.seq_len+self.pred_len]
                                task_world.append((world_inputs,world_outputs,torch.tensor(intention_label)))
                    
                    data.extend(task_data)
                    data_world.extend(task_world)
        
        weights = torch.tensor([1/x for x in weights])
        return data,data_world,weights
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.data_world:
            sample_world = self.data_world[idx]
        if self.transform:
            sample = self.transform(sample)
        if self.data_world:
            return sample,sample_world
        else:
            return sample,sample
    
    def get_weights(self):
        return self.weights

if __name__ == '__main__':
    ROOT_DIR = f'{FILE_DIR}/../human_traj'
    JSON_FILE = f'{ROOT_DIR}/cut_traj.json'

    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_window', default=5,
                        help="looking back window size")
    parser.add_argument('--class_num', default=2,
                        help="number of classification categories")
    parser.add_argument('--individual', default=False, 
                        help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--channels', type=int, default=15*3, 
                        help='encoder input size, channel here is set as upper body points * 3\
                        DLinear with --individual, use this hyperparameter as the number of channels') 
    parser.add_argument('--half_body', type=int, default=False, 
                        help='whether to extract only half body keypoints') 
    args = parser.parse_args()

    dataset = MyDataset(JSON_FILE,ROOT_DIR,args)

    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        inputs, labels = batch