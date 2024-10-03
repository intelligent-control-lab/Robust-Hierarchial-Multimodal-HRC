import argparse
import os
import torch
from pathlib import Path
FILE_DIR = Path(__file__).resolve().parent
import numpy as np
import cv2
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
import pdb

from DLinear import Model_FinalIntention,Model_FinalTraj
from Dataset import MyDataset


if __name__ == '__main__':
    ROOT_DIR = f'{FILE_DIR}/../human_traj'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', default=5,
                        help="input frame window")
    parser.add_argument('--pred_len', default=5,
                        help="output predicted frame length")
    parser.add_argument('--class_num', default=4,
                        help="number of classification categories")
    parser.add_argument('--individual', default=False, 
                        help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--channels', type=int, default=15*3, 
                        help='encoder input size, channel here is set as upper body points * 3\
                        DLinear with --individual, use this hyperparameter as the number of channels') 
    parser.add_argument('--half_body', type=int, default=False, 
                        help='whether to extract only half body keypoints') 
    parser.add_argument('--batch_size', type=int, default=2) 
    parser.add_argument('--test_whole', default=True,
                        help='whether to test on build_cars tasks')
    parser.add_argument('--no_mask', action="store_true",
                        help='whether to remove mask')
    parser.add_argument('--filter_type', type=str,default=None,
                        help='Trajectory filter type, must be used together with no_mask,four options:["median","wiener","kalman","ema"]')
    parser.add_argument('--model_type',type=str,default="final_traj",
                        help='two options:[final_intention,final_traj]')
    parser.add_argument('--input_type', type=str,default="pkl",
                        help='two options:[pkl,npy]')
    parser.add_argument('--pose_world', action="store_true",
                        help='whether to use landmarks_world')
    parser.add_argument('--epochs', type=int, default=40) 
    args = parser.parse_args()

    if args.half_body:
        args.channels = 10*3
    if args.model_type == "final_intention":
        net = Model_FinalIntention(args)
    elif args.model_type == "final_traj":
        net = Model_FinalTraj(args)

    if args.no_mask:
        JSON_FILE = f'{ROOT_DIR}/cut_traj_nomask.json'
    else:
        JSON_FILE = f'{ROOT_DIR}/cut_traj_mask.json'

    batch_size = args.batch_size
    dataset = MyDataset(JSON_FILE,ROOT_DIR,args,dataset_type="train",world_pose=args.pose_world)
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
    print("data loaded!")
    # pdb.set_trace()
    class_criterion = torch.nn.CrossEntropyLoss(weight=dataset.weights)
    traj_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochs = args.epochs
    for epoch in range(epochs):
        running_loss = 0.0
        count = 0
        for batch,batch_world in tqdm(dataloader):
            if args.pose_world:
                inputs,target_traj,labels = batch_world
            else:
                inputs,target_traj,labels = batch
            pred_traj,pred_intention = net(inputs)
            traj_loss = traj_criterion(pred_traj,target_traj)
            intention_loss = class_criterion(pred_intention, labels)
            loss = traj_loss + intention_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            count += 1
                
        print(f'Epoch {epoch + 1}, Loss: {running_loss / count}')
        
    if args.filter_type:
        torch.save(net.state_dict(), f'{FILE_DIR}/checkpoints/seq{args.seq_len}_pred{args.pred_len}_epoch{epochs}_whole_{args.input_type}_{args.model_type}_nomask{args.no_mask}_filter{args.filter_type}.pth')
    else:
        torch.save(net.state_dict(), f'{FILE_DIR}/checkpoints/seq{args.seq_len}_pred{args.pred_len}_epoch{epochs}_whole_{args.input_type}_{args.model_type}_nomask{args.no_mask}.pth')


    

