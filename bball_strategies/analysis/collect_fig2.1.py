import numpy as np
import os
from vis_game import plot_data

all_model = ['CNN_WO', 'CNN', 'RNN_WO']
condition_idx = 0

root_path = '../data/WGAN/fig2/'

save_path = root_path + 'compare'
if not os.path.exists(save_path):
    os.makedirs(save_path)

results_data = np.load('../data/FEATURES-4.npy')
results_data = np.concatenate(
    [
        # ball
        results_data[:, :, 0, :3].reshape(
            [results_data.shape[0], results_data.shape[1], 1 * 3]),
        # team A players
        results_data[:, :, 1:6, :2].reshape(
            [results_data.shape[0], results_data.shape[1], 5 * 2]),
        # team B players
        results_data[:, :, 6:11, :2].reshape(
            [results_data.shape[0], results_data.shape[1], 5 * 2])
    ], axis=-1
)
print(results_data.shape)
idx = results_data.shape[0]//10*9 + 962
plot_data(results_data[idx], length=100,
            file_path=save_path+'/play_REAL.mp4', if_save=True, vis_annotation=True)

cmp_result = np.empty(shape=[4, 100, 23])
# real
cmp_result[0] = results_data[idx]
# fake
for i, model in enumerate(all_model):
    fake_result = np.load(root_path+model+'/results_A_fake_B.npy')
    fake_critic = np.load(root_path+model+'/results_critic_scores.npy')
    n_idx = np.argmax(fake_critic[0])
    cmp_result[i+1] = fake_result[n_idx,
                                  condition_idx]


# fake
for i in range(1, cmp_result.shape[0]):
    plot_data(cmp_result[i], length=100,
              file_path=save_path+'/play_' + all_model[i-1] + '.mp4', if_save=True, vis_annotation=True)

np.save(save_path+'/all_results.npy', cmp_result)