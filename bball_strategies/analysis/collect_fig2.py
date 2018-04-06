import numpy as np
import os
from vis_game import plot_data

all_model = ['cnn_wi_mul_828k_nl', 'cnn_wo_644k_vanilla', 'rnn_wo_442k']
noise_list = [37, 60, 0]
condition_idx = 3

root_path = '../data/WGAN/all_model_results/'
length = np.load(root_path+'length.npy')
real_data = np.load(root_path+'real_data.npy')

cmp_result = np.empty(shape=[4, length[condition_idx], 23])
# real
cmp_result[0] = real_data[condition_idx, :length[condition_idx]]
# fake
for i, model in enumerate(all_model):
    fake_result = np.load(root_path+model+'/results_A_fake_B.npy')
    cmp_result[i+1] = fake_result[noise_list[i],
                                  condition_idx, :length[condition_idx]]

save_path = root_path + 'compare'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# real
plot_data(cmp_result[0], length=length[condition_idx],
          file_path=save_path+'/play_real.mp4', if_save=True, vis_annotation=True)
# fake
for i in range(1, cmp_result.shape[0]):
    plot_data(cmp_result[i], length=length[condition_idx],
              file_path=save_path+'/play_' + all_model[i-1] + '.mp4', if_save=True, vis_annotation=True)

np.save(save_path+'/all_results.npy', cmp_result)