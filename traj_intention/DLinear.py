import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model_FinalIntention(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, args):
        super(Model_FinalIntention, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.class_num = args.class_num

        # Decompsition Kernel Size
        kernel_size = 3
        self.decompsition = series_decomp(kernel_size)
        self.individual = args.individual
        self.channels = args.channels

        if self.individual: # TODO
            self.Traj_Seasonal = nn.ModuleList()
            self.Traj_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Traj_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Traj_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Traj_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Traj_Trend = nn.Linear(self.seq_len,self.pred_len)
        self.Intention_Predictor = nn.Linear((self.seq_len+self.pred_len)*self.channels,self.class_num)
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # pdb.set_trace()
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        # seasonal_init, trend_init = seasonal_init.view(seasonal_init.shape[0],self.channels*self.seq_len), trend_init.view(seasonal_init.shape[0],self.channels*self.seq_len)
        if self.individual: # TODO
            traj_seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.class_num],dtype=seasonal_init.dtype).to(seasonal_init.device)
            traj_trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.class_num],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                traj_seasonal_output[:,i,:] = self.Traj_Seasonal[i](seasonal_init[:,i,:])
                traj_trend_output[:,i,:] = self.Traj_Trend[i](trend_init[:,i,:])
        else:
            traj_seasonal_output = self.Traj_Seasonal(seasonal_init)
            traj_trend_output = self.Traj_Trend(trend_init)


        traj_output = traj_seasonal_output + traj_trend_output
        traj_output = traj_output.permute(0,2,1) # [Batch, Output length, Channel]
        intention_output = self.Intention_Predictor(torch.cat((x,traj_output),1).view(x.shape[0],-1))
        return traj_output, intention_output

class Model_FinalTraj(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, args):
        super(Model_FinalTraj, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.class_num = args.class_num

        # Decompsition Kernel Size
        kernel_size = 3
        self.decompsition = series_decomp(kernel_size)
        self.individual = args.individual
        self.channels = args.channels

        if self.individual: # TODO
            self.Traj_Seasonal = nn.ModuleList()
            self.Traj_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Traj_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Traj_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Traj_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Traj_Trend = nn.Linear(self.seq_len,self.pred_len)
        self.Intention_Vector = nn.Linear(self.channels,self.class_num)
        self.Intention_Predictor = nn.Linear(self.seq_len*self.class_num,self.class_num)
        self.Traj_Predictor = nn.Linear(self.channels+self.class_num,self.channels)
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # pdb.set_trace()
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        # seasonal_init, trend_init = seasonal_init.view(seasonal_init.shape[0],self.channels*self.seq_len), trend_init.view(seasonal_init.shape[0],self.channels*self.seq_len)
        if self.individual: # TODO
            traj_seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.class_num],dtype=seasonal_init.dtype).to(seasonal_init.device)
            traj_trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.class_num],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                traj_seasonal_output[:,i,:] = self.Traj_Seasonal[i](seasonal_init[:,i,:])
                traj_trend_output[:,i,:] = self.Traj_Trend[i](trend_init[:,i,:])
        else:
            intention_vector = self.Intention_Vector(x).permute(0,2,1) # [Batch,class_num,seq_len]
            seasonal_input = torch.cat((seasonal_init,intention_vector),1)
            trend_input = torch.cat((trend_init,intention_vector),1)
            traj_seasonal_output = self.Traj_Seasonal(seasonal_input)
            traj_trend_output = self.Traj_Trend(trend_input)

        traj_output = traj_seasonal_output + traj_trend_output
        traj_output = traj_output.permute(0,2,1) # [Batch, Output length, Channel+class_num]
        traj_output = self.Traj_Predictor(traj_output)
        intention_output = self.Intention_Predictor(intention_vector.reshape(x.shape[0],-1))
        return traj_output, intention_output