# """
# Ref: http://buckets.peterbeshai.com/app/#/leagueView/2015?l_countMin=1&l_showLegend=false&p_countMin=1
# Source: view-source:http://buckets.peterbeshai.com/app/#/leagueView/2015?l_countMin=1&l_showLegend=false&p_countMin=1 
# 1. fetch all 2015-2016 season from Source above. (save it as numpy)
# 2. implement the function 'get_prob_by_loc()'
# """


# def get_prob_by_loc(x, y, mode='FQ'):
#     """ ### Get Probability By Location

#     Args
#     ----
#     x : ndarray, dtype=float32, shape=(None,), range=[0, 94.0)
#     y : ndarray, dtype=float32, shape=(None,), range=[0, 50.0)
#     mode : str,
#         - 'FQ' : Shoot Frequency
#         - 'FG' : Field Goal Rate
    
#     Raises
#     ------
#     ValueError : if input argument is not valid.

#     Return
#     ------
#     prob : ndarray, dtype=float32, shape=(None,), range=[0, 1.0)
#         given location, returns the probability of FQ/FG.
#     """
#     raise NotImplementedError()


# def main():
#     get_prob_by_loc(x=0.1, y=0.1, mode)

# if __name__ == '__main__':
#     main()


# coding: utf-8

# In[250]:


import numpy as np
import json as js
import math


# In[251]:


FG=[]
FQ=[]
total_shot=0

#Calculate the field goal

with open("data.json",'r') as f:
    load_dict = js.load(f)
    for k in range(2350):
        total_shot += load_dict[k]['COUNT']
    print("Total shot :",total_shot)
    for j in range(50):  #Column
        for i in range(47):  #Row
            #print(load_dict[i+j*47]['COL'],load_dict[i+j*47]['ROW'],load_dict[i+j*47]['MADE'],load_dict[i+j*47]['COUNT'])
            FQ.append(load_dict[i+j*47]['COUNT']/total_shot)
            #print(i,j,"FQ: ",FQ[i+j*47])
            if load_dict[i+j*47]['COUNT'] == 0 :
                FG.append(0.0)
            else:
                FG.append(round((load_dict[i+j*47]['MADE']/load_dict[i+j*47]['COUNT']),3))  #FG四捨五入到小數點第一位
            #print("FG: ",FG[i+j*47]) 
   


# In[252]:


FG_new = np.array(FG,dtype=np.float32)
FQ_new = np.array(FQ,dtype=np.float32)
FG_2D = FG_new.reshape(50,47)
FQ_2D = FQ_new.reshape(50,47)

#for i in range(50):  
#        for j in range(47):  
#            print(i,j,FG_2D[i][j])


# In[253]:


FG_rev=np.arange(4700).reshape(94,50)
FQ_rev=np.arange(4700).reshape(94,50)
FG_rev.dtype=np.float32
FQ_rev.dtype=np.float32
for j in range(47):  #Column
        for i in range(50):  #Row
            FG_rev[j][i] = FG_2D[i][j]
            FQ_rev[j][i] = FQ_2D[i][j]
            FG_rev[93-j][49-i] = FG_2D[i][j]  #全場FG data，令左下角為(0,0)
            FQ_rev[93-j][49-i] = FG_2D[i][j]


# In[254]:


# for i in range(94):  
#         for j in range(50):  
#             print("(x , y) = (",i,",",j,")")
#             print("FG:",FG_rev[i][j])
#             print("FQ:",FQ_rev[i][j])
            


# In[255]:


# print(FG_rev.shape,FG_rev.dtype)
# print(FG_rev[0][0].dtype)


# In[256]:


def get_prob_by_loc(x, y, mode):
    
    if x>93 or x<0 or y>49 or y<0:
        return("Wrong Input for x or y.")
    
    x_left = math.floor(x)
    x_right = x_left+1
    y_down = math.floor(y)
    y_up = y_down+1
    deno = (x_right-x_left)*(y_up-y_down)

    if mode ==('FG'):
        FG_final = round((FG_rev[x_left][y_down]*(x_right-x)*(y_up-y)+FG_rev[x_right][y_down]*(x-x_left)*(y_up-y)+FG_rev[x_left][y_up]*(x_right-x)*(y-y_down)+FG_rev[x_right][y_up]*(x-x_left)*(y-y_down))/deno,3)
        return FG_final
    elif mode ==('FQ'):
        FQ_final = (FQ_rev[x_left][y_down]*(x_right-x)*(y_up-y)+FQ_rev[x_right][y_down]*(x-x_left)*(y_up-y)+FQ_rev[x_left][y_up]*(x_right-x)*(y-y_down)+FQ_rev[x_right][y_up]*(x-x_left)*(y-y_down))/((x_right-x_left)*(y_up-y_down))
        return FQ_final


# In[257]:


def main():
    print(get_prob_by_loc(x=0.25, y=1, mode='FG'))


# In[258]:


if __name__ == '__main__':
    main()

