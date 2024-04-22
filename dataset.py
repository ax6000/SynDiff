import torch.utils.data
import numpy as np, h5py
import random

class ExpandDimDataset(torch.utils.data.TensorDataset):
    def __init__(self,*tensors):
        # self.tensors = torch.from_numpy(tensors)
        self.tensors = tensors
    def _expand_dims(self,tensor):
        length = tensor.shape[-1]
        reshaped = torch.unsqueeze(tensor, axis=2)
        reshaped = torch.repeat_interleave(reshaped, length, axis=2)
        return reshaped
    def __len__(self):
        return self.tensors[0].size(0)
    
    def __getitem__(self, index):
        return tuple(self._expand_dims(tensor[index]) for tensor in self.tensors)


def CreateDatasetSynthesis(phase, input_path, contrast1 = 'T1', contrast2 = 'T2'):

    target_file = input_path + "/data_{}_{}.mat".format(phase, contrast1)
    data_fs_s1=LoadDataSet(target_file,padding=False)
    
    target_file = input_path + "/data_{}_{}.mat".format(phase, contrast2)
    data_fs_s2=LoadDataSet(target_file,padding=False)

    dataset=ExpandDimDataset(torch.from_numpy(data_fs_s1),torch.from_numpy(data_fs_s2))  
    return dataset 



#Dataset loading from load_dir and converintg to 256x256 
def LoadDataSet(load_dir, variable = 'data_fs', padding = True, Norm = True):
    f = h5py.File(load_dir,'r') 
    if np.array(f[variable]).ndim==3:
        data=np.expand_dims(np.transpose(np.array(f[variable]),(0,2,1)),axis=1)
        # print(data.shape)
    elif np.array(f[variable]).ndim==2:
        data=np.expand_dims(np.transpose(np.array(f[variable]),(0,1)),axis=1)
        print(data.shape)
    else:
        data=np.transpose(np.array(f[variable]),(1,0,3,2))
    data=data.astype(np.float32) 
    if padding:
        pad_x=int((256-data.shape[2])/2)
        pad_y=int((256-data.shape[3])/2)
        print('padding in x-y with:'+str(pad_x)+'-'+str(pad_y))
        data=np.pad(data,((0,0),(0,0),(pad_x,pad_x),(pad_y,pad_y)))   
    if Norm:    
        data=(data-0.5)/0.5      
    return data
