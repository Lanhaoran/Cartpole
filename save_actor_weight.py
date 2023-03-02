import os
import torch, numpy as np, torch.nn as nn
from utils import *


'''
path = r'E:\Character Motion\CartPole\tianshou_cartpole\models\MyCartPole-v0-7.5_0.1_2.0\ppo\5\policy.pth'
state_dict = torch.load(path)
print(type(state_dict))

for param_tensor in state_dict:
    size =state_dict[param_tensor].size()
    print(param_tensor, "\t", size)
    if 'critic.net' in param_tensor:
        print(True)
        state_dict[param_tensor] = torch.randint(10, size=size)
    #print("Change parameters.\t")
'''

root_path = r'E:\Character Motion\CartPole\tianshou_cartpole\models_5'
folders = []
config_list = []
data_list = []
for folder in os.listdir(root_path):
    #if '-7.5' in folder or '-10.0' in folder:
    #if '-1.0_0.1_1.0' in folder:
    folders.append(folder)
    #print(folder,type(folder))
    config = folder.split('-')[-1]    # get config
    #print(config)
    label = list(map(float, config.split('_')))    # convert to float
    #print(label)   
    for (root,dirs,files) in os.walk(os.path.join(root_path,folder,'ppo'), topdown=True):
        #print(root)
        #print(dirs)
        #print(files)
        for dir in dirs[:5]:
            ## get result txt to fillter good policies
            result_txt_path = os.path.join(root, dir, 'train_result.txt')
            result = read_result(result_txt_path)
            if float(result['best_reward']) >= 475:
                config_list.append(np.array(label))
                model_path = os.path.join(root, dir, 'policy.pth')
                model_state_dict = torch.load(model_path)
                actor_weight_0 = model_state_dict['actor.net.module.preprocess.model.model.0.weight'].cpu().numpy()   #(64,4) / (16,4)
                actor_bias_0 = model_state_dict['actor.net.module.preprocess.model.model.0.bias'].cpu().numpy()   #(64,) / (16,)
                #actor_weight_1 = model_state_dict['actor.net.module.preprocess.model.model.2.weight'].cpu().numpy()   #(64,64) / (16,16)
                #actor_bias_1 = model_state_dict['actor.net.module.preprocess.model.model.2.bias'].cpu().numpy()   #(64,) / (16,)
                #actor_weight_2 = model_state_dict['actor.net.module.last.model.0.weight'].cpu().numpy()   #(2,64) / (2,16)
                #actor_bias_2 = model_state_dict['actor.net.module.last.model.0.bias'].cpu().numpy()   #(2,) / (2,)

                actor_bias_0 = np.expand_dims(actor_bias_0, axis=1)   #(64,1)    |     (16,1)
                #actor_bias_1 = np.expand_dims(actor_bias_1, axis=1)   #(64,1)    |     (16,1)
                #actor_bias_2 = np.expand_dims(actor_bias_2, axis=1)   #(2,1)     

                '''
                ## full layer data 
                para_matrix_0 = np.concatenate([actor_weight_0, actor_bias_0], axis=1)   #(64,5) | (16,5)
                para_matrix_1 = np.concatenate([actor_weight_1, actor_bias_1], axis=1)   #(64,65) | (16,17)
                trans_weight_2 = np.transpose(actor_weight_2)    #(64,2) | (16,2)
                #padding_bias_2 = np.pad(actor_bias_2,((0,62),(0,0)),'constant')    #(64,1) 
                padding_bias_2 = np.pad(actor_bias_2,((0,14),(0,0)),'constant')    #(16,1)
                para_matrix = np.concatenate([para_matrix_0,para_matrix_1,trans_weight_2,padding_bias_2], axis=1)   #(64,73) | (16,25)
                '''

                ## last layer data of [16,16]
                #para_matrix = np.concatenate([actor_weight_2, actor_bias_2], axis=1)   # (2,17)

                ## single layer policy data
                #padding_bias_2 = np.pad(actor_bias_2,((0,14),(0,0)),'constant')    #(16,1)
                #para_matrix = np.concatenate([actor_weight_0,actor_bias_0,np.transpose(actor_weight_2),padding_bias_2], axis=1)   #(16,8)

                ## first layer data (fixed last two layer)
                para_matrix = np.concatenate([actor_weight_0, actor_bias_0], axis=1)   # (16,5)

                data_list.append(para_matrix)

data_list = np.expand_dims(data_list, axis=1)    # (1252, 1, 64, 73) | (N, 1, 16, 25) | (N , 1, 2, 17) | (N, 1, 16, 8)
save_path = r'E:\Character Motion\CartPole\tianshou_cartpole\actor_data'
np.save(save_path+'\data_6.npy', data_list)
np.save(save_path+'\configs_6.npy', config_list)
#print(len(config_list)) 
#print(len(data_list))
#rint(len(folders))