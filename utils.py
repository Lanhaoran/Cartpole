
import gym, torch, numpy as np, torch.nn as nn
import random
import os

##########################################
##  Load the diffusion result
#######################################
## Load [64,64] policy network
def LoadActorWeight_64(dict, policy, path, id):
    # copy
    net_dict = dict.copy()
    para_matrix = np.load(path)[:,0,:,:]          # (20,64,73)
    para_matrix_pick = para_matrix[id]             # (64,73)
    para_0, para_1, trans_weight_2, padding_bias_2,_ = np.split(para_matrix_pick,[5,70,72,73],axis=1)
    actor_weight_0, actor_bias_0,_ = np.split(para_0, [4,5], axis=1)
    actor_weight_1, actor_bias_1,_ = np.split(para_1, [64,65], axis=1)
    actor_weight_2 = np.transpose(trans_weight_2)
    actor_bias_2 = padding_bias_2[:2,:]

    net_dict["actor.net.module.preprocess.model.model.0.weight"] = torch.from_numpy(actor_weight_0)
    net_dict['actor.net.module.preprocess.model.model.0.bias'] = torch.from_numpy(actor_bias_0.squeeze(axis=1))
    net_dict['actor.net.module.preprocess.model.model.2.weight'] = torch.from_numpy(actor_weight_1)
    net_dict['actor.net.module.preprocess.model.model.2.bias'] = torch.from_numpy(actor_bias_1.squeeze(axis=1))
    net_dict['actor.net.module.last.model.0.weight'] = torch.from_numpy(actor_weight_2)
    net_dict['actor.net.module.last.model.0.bias'] = torch.from_numpy(actor_bias_2.squeeze(axis=1))
    
    net_dict["_actor_critic.actor.net.module.preprocess.model.model.0.weight"] = torch.from_numpy(actor_weight_0)
    net_dict['_actor_critic.actor.net.module.preprocess.model.model.0.bias'] = torch.from_numpy(actor_bias_0.squeeze(axis=1))
    net_dict['_actor_critic.actor.net.module.preprocess.model.model.2.weight'] = torch.from_numpy(actor_weight_1)
    net_dict['_actor_critic.actor.net.module.preprocess.model.model.2.bias'] = torch.from_numpy(actor_bias_1.squeeze(axis=1))
    net_dict['_actor_critic.actor.net.module.last.model.0.weight'] = torch.from_numpy(actor_weight_2)
    net_dict['_actor_critic.actor.net.module.last.model.0.bias'] = torch.from_numpy(actor_bias_2.squeeze(axis=1))
    
    policy.load_state_dict(net_dict)
    print("Load 64 size diffusion sample result done!")

    return policy

## Load [16,16] policy network
def LoadActorWeight_16(dict, policy, path, id):
    # copy
    net_dict = dict.copy()
    para_matrix = np.load(path)[:,0,:,:]          # (20,16,25)
    para_matrix_pick = para_matrix[id]             # (16,25)
    para_0, para_1, trans_weight_2, padding_bias_2,_ = np.split(para_matrix_pick,[5,22,24,25],axis=1)
    actor_weight_0, actor_bias_0,_ = np.split(para_0, [4,5], axis=1)   # (16,4) (16,1)
    actor_weight_1, actor_bias_1,_ = np.split(para_1, [16,17], axis=1)  # (16,16) (16,1)
    actor_weight_2 = np.transpose(trans_weight_2)  # (2,16)
    actor_bias_2 = padding_bias_2[:2,:]  #(2,1)

    net_dict["actor.net.module.preprocess.model.model.0.weight"] = torch.from_numpy(actor_weight_0)
    net_dict['actor.net.module.preprocess.model.model.0.bias'] = torch.from_numpy(actor_bias_0.squeeze(axis=1))
    net_dict['actor.net.module.preprocess.model.model.2.weight'] = torch.from_numpy(actor_weight_1)
    net_dict['actor.net.module.preprocess.model.model.2.bias'] = torch.from_numpy(actor_bias_1.squeeze(axis=1))
    net_dict['actor.net.module.last.model.0.weight'] = torch.from_numpy(actor_weight_2)
    net_dict['actor.net.module.last.model.0.bias'] = torch.from_numpy(actor_bias_2.squeeze(axis=1))
    
    net_dict["_actor_critic.actor.net.module.preprocess.model.model.0.weight"] = torch.from_numpy(actor_weight_0)
    net_dict['_actor_critic.actor.net.module.preprocess.model.model.0.bias'] = torch.from_numpy(actor_bias_0.squeeze(axis=1))
    net_dict['_actor_critic.actor.net.module.preprocess.model.model.2.weight'] = torch.from_numpy(actor_weight_1)
    net_dict['_actor_critic.actor.net.module.preprocess.model.model.2.bias'] = torch.from_numpy(actor_bias_1.squeeze(axis=1))
    net_dict['_actor_critic.actor.net.module.last.model.0.weight'] = torch.from_numpy(actor_weight_2)
    net_dict['_actor_critic.actor.net.module.last.model.0.bias'] = torch.from_numpy(actor_bias_2.squeeze(axis=1))
    
    policy.load_state_dict(net_dict)
    print("Load 16 size diffusion sample result done!")

    return policy

## Change last layer of [16,16]
def LoadActorWeight_last_16(dict, policy, path, id):
    # copy
    net_dict = dict.copy()
    para_matrix = np.load(path)[:,0,:,:]          # (20,2,17)
    para_matrix_pick = para_matrix[id]             # (2,17)
    actor_weight_2,actor_bias_2,_ = np.split(para_matrix_pick,[16,17],axis=1)   #(2,16) (2,1)

    net_dict['actor.net.module.last.model.0.weight'] = torch.from_numpy(actor_weight_2)
    net_dict['actor.net.module.last.model.0.bias'] = torch.from_numpy(actor_bias_2.squeeze(axis=1))
    
    net_dict['_actor_critic.actor.net.module.last.model.0.weight'] = torch.from_numpy(actor_weight_2)
    net_dict['_actor_critic.actor.net.module.last.model.0.bias'] = torch.from_numpy(actor_bias_2.squeeze(axis=1))
    
    policy.load_state_dict(net_dict)
    print("Load last layer of 16 size diffusion sample result done!")

    return policy

## Load [16] policy network
def LoadActorWeight_single_layer_16(dict, policy, path, id):
    # copy
    net_dict = dict.copy()
    para_matrix = np.load(path)[:,0,:,:]          # (20,16,8)
    para_matrix_pick = para_matrix[id]             # (16,8)
    actor_weight_0, actor_bias_0, trans_weight_2, padding_bias_2,_ = np.split(para_matrix_pick,[4,5,7,8],axis=1)
    actor_weight_2 = np.transpose(trans_weight_2)  # (2,16)
    actor_bias_2 = padding_bias_2[:2,:]  #(2,1)

    net_dict["actor.net.module.preprocess.model.model.0.weight"] = torch.from_numpy(actor_weight_0)
    net_dict['actor.net.module.preprocess.model.model.0.bias'] = torch.from_numpy(actor_bias_0.squeeze(axis=1))
    net_dict['actor.net.module.last.model.0.weight'] = torch.from_numpy(actor_weight_2)
    net_dict['actor.net.module.last.model.0.bias'] = torch.from_numpy(actor_bias_2.squeeze(axis=1))
    
    net_dict["_actor_critic.actor.net.module.preprocess.model.model.0.weight"] = torch.from_numpy(actor_weight_0)
    net_dict['_actor_critic.actor.net.module.preprocess.model.model.0.bias'] = torch.from_numpy(actor_bias_0.squeeze(axis=1))
    net_dict['_actor_critic.actor.net.module.last.model.0.weight'] = torch.from_numpy(actor_weight_2)
    net_dict['_actor_critic.actor.net.module.last.model.0.bias'] = torch.from_numpy(actor_bias_2.squeeze(axis=1))
    
    policy.load_state_dict(net_dict)
    print("Load single layer 16 size diffusion sample result done!")

    return policy

## Load first layer of [16,16] policy network
def LoadActorWeight_first_layer_16(dict, policy, path, id):
    # copy
    net_dict = dict.copy()
    para_matrix = np.load(path)[:,0,:,:]    # (20,16,5)
    para_matrix_pick = para_matrix[id]      # (16,5)
    actor_weight_0, actor_bias_0,_ = np.split(para_matrix_pick,[4,5],axis=1)

    net_dict["actor.net.module.preprocess.model.model.0.weight"] = torch.from_numpy(actor_weight_0)
    net_dict['actor.net.module.preprocess.model.model.0.bias'] = torch.from_numpy(actor_bias_0.squeeze(axis=1))
    
    net_dict["_actor_critic.actor.net.module.preprocess.model.model.0.weight"] = torch.from_numpy(actor_weight_0)
    net_dict['_actor_critic.actor.net.module.preprocess.model.model.0.bias'] = torch.from_numpy(actor_bias_0.squeeze(axis=1))
    
    policy.load_state_dict(net_dict)
    print("Load first layer of [16,16] network done!")

    return policy









##################################################
## Random change actor net
###############################################
def RandomActorWeight(dict, policy):
    net_dict = dict.copy()
    for param_tensor in policy.state_dict():
        size = policy.state_dict()[param_tensor].size()
        #print(param_tensor, "\t", size)
        if '.actor.net.' in param_tensor or 'actor.net.' in param_tensor:
            print(True)
            state_before = policy.state_dict()[param_tensor]
            net_dict[param_tensor] = torch.randint(100000, size=size)
        #print("Change parameters.\t")
    policy.load_state_dict(net_dict)

    return policy


#########################################
## Add noise to the control net
#########################################
def AddNoise(dict, policy, sigma, layer_name):
    net_dict = dict.copy()
    total_noise = 0
    for param_tensor in policy.state_dict():
        size = policy.state_dict()[param_tensor].size()
        #print(param_tensor, "\t", size)
        # if '.actor.net.' in param_tensor or 'actor.net.' in param_tensor:
        if layer_name in param_tensor:
            noise = random.gauss(0, sigma)
            net_dict[param_tensor] += noise
            total_noise += np.abs(noise)
            # print("Add noise to {}.\t".format(param_tensor))
    policy.load_state_dict(net_dict)

    return policy, total_noise



###################################################################
## Visualize the distribution of the poilcy network
####################################################################
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

def t_sne(X,y):
    # Random state.
    RS = 20150101

    # We import seaborn to make nice plots.
    import seaborn as sns
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    '''
    digits = load_digits()
    # We first reorder the data points according to the handwritten numbers.
    X = np.vstack([digits.data[digits.target==i]
                for i in range(10)])
    y = np.hstack([digits.target[digits.target==i]
                for i in range(10)])'''


    digits_proj = TSNE(random_state=RS).fit_transform(X)
    
    def scatter(x, colors):
        # We choose a color palette with seaborn.
        palette = np.array(sns.color_palette("hls", 10))

        # We create a scatter plot.
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                        c=palette[colors.astype(np.int)])
        #plt.xlim(-25, 25)
        #plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')

        # We add the labels for each digit.
        txts = []
        for i in range(len(np.unique(colors))):
            # Position of each label.
            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

        return f, ax, sc, txts
    
    scatter(digits_proj, y)
    plt.savefig('digits_tsne-generated.png', dpi=120)
    plt.show()

#####################################################################
## Get statistical characteristics of data of the layers in actor net
#####################################################################
def stat_charac(path):
    # path until 'ppo'
    # init the param dict
    model_state_dict = torch.load(os.path.join(path, '0', 'policy.pth'))
    dict_list = [(k,[]) for k,_ in model_state_dict.items() if 'actor.net' in k and 'critic' not in k]
    para_dict = dict(dict_list)
    for (root,dirs,files) in os.walk(path, topdown=True):
        for dir in dirs:
            model_path = os.path.join(root, dir, 'policy.pth')
            # load the model
            model_state_dict = torch.load(model_path)
            for key,value in model_state_dict.items():
                if 'actor.net' in key and 'critic' not in key:
                    para_dict[key].append(value.cpu().flatten().numpy())
    
    # get the statistical characteristics of each layer
    stat_dict = {}
    for k,v in para_dict.items():
        mean = torch.mean(torch.from_numpy(np.array(v)), dim=0)
        std = torch.std(torch.from_numpy(np.array(v)), dim=0)
        max = torch.max(torch.from_numpy(np.array(v)), dim=0)
        min = torch.min(torch.from_numpy(np.array(v)), dim=0)

        stat_dict[k+'__mean'] = mean
        stat_dict[k+'__std'] = std
        stat_dict[k+'__max'] = max
        stat_dict[k+'__min'] = min
    return stat_dict


#######################################################################
## Read the reault files
##########################################################################
def read_result(path):
    # path until '.txt'
    with open(path, "r") as f:
        data = f.read()
    #print(data)

    ## string to dict
    lines = data.strip().split('\n')
    result = {}
    for line in lines:
        key_value = line.split(':')
        key = key_value[0].strip()
        value = key_value[1].strip()
        result[key] = value

    #print(result['best_result'])

    return result








if __name__ == '__main__':

    ####################################
    ## t-sne visualization
    ####################################
    '''
    data1 = np.load(r"E:\Character Motion\CartPole\tianshou_cartpole\actor_data\data_2.npy")
    #label = np.load(r"E:\Character Motion\CartPole\tianshou_cartpole\actor_data\configs_2.npy")
    label1 = np.zeros((1000,))
    data1 = np.squeeze(data1,axis = 1)
    data1 = data1.reshape((1000,-1))
    # random data
    data2 = np.random.normal(loc=10.0, scale=1.0, size=data1.shape)
    label2 = label1+1
    # concatenate
    data = np.concatenate([data1, data2],axis=0)
    label = np.concatenate([label1, label2], axis=0)
    t_sne(data, label)
    '''

    ###########################################
    ## 
    ####################################
    path = r'E:\Character Motion\CartPole\tianshou_cartpole\models_5\MyCartPole-v0-0.1_0.1_0.1\ppo\0\train_result.txt'
    data = read_result(path)


