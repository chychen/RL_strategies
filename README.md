# Backetball Strategies with GAIL and BatchPPO

[Master Thesis](http://www.airitilibrary.com/Publication/alDetailedMesh?docid=U0030-0205201911102275)

![](https://github.com/chychen/RL_strategies/blob/master/demo1.gif)

## Getting Started

- Clone this repo:

```bash
$ git clone https://github.com/chychen/RL_strategies
```

- Environment Setting
[docker](https://docs.docker.com/glossary/?term=installation)/ [nvidia-docker](https://github.com/NVIDIA/nvidia-docker):

```bash
$ docker pull jaycase/bballgail:init
$ nvidia-docker run --name {name} -it -p 127.0.0.1:6006:6006 -v {repo_path}RL_strategies/:/RL_strategies -w /RL_strategies jaycase/bballgail bash
```

- Download the dataset. (You should put dataset right under the folder "{repo_path}/RL_strategies/data/")
(QQ can't find the files)

```bash
root@7bdb7335cde0:/RL_strategies# cd bball_strategies/data/
root@7bdb7335cde0:/RL_strategies/bball_strategies/data# wget http://140.113.210.14:6006/NBA/data/FPS5.npy
root@7bdb7335cde0:/RL_strategies/bball_strategies/data# wget http://140.113.210.14:6006/NBA/data/FPS5Length.npy
```

## Data Preprocessing

- Filter out bad data.

```bash
root@7bdb7335cde0:/RL_strategies/bball_strategies/data# python3 preprocess.py
```

- Create training dataset for GAIL. (transform data into state-action pair.)

```bash
root@7bdb7335cde0:/RL_strategies/bball_strategies/data# python3 create_gail_data.py
```

- Reorder the data offense and defense positions by rule-based method.
- And, duplicate OrderedGAILTransitionData_52.hdf5 for multi process settings.

```bash
root@7bdb7335cde0:/RL_strategies/bball_strategies/data# python3 postprocess_data_order.py 
root@7bdb7335cde0:/RL_strategies/bball_strategies/data# cp OrderedGAILTransitionData_52.hdf5 OrderedGAILTransitionData_522.hdf5
```

## Training

### Configuration

All configurations are in "{repo_path}/RL_strategies/bball_strategies/scripts/gail/config.py"


### Train Model

```bash
root@7bdb7335cde0:/RL_strategies# python3 -m bball_strategies.scripts.gail.train --config=double_curiculum
```

### Monitor training

```bash
root@7bdb7335cde0:/RL_strategies# tensorboard --logdir='logdir/gail_defense/{time stamp}-double_curiculum' --port=6006
```

## Analysis

###Custimized Animation Players

- Buid upon [kivy](https://kivy.org/docs/installation/installation.html) framework
- You can drag and drop the '.npz' files (recorded while tranining) into the player.

```bash
root@7bdb7335cde0:/RL_strategies# cd gui_tool/
root@7bdb7335cde0:/RL_strategies/gui_tool# python3 player.py
```

### Quantitative Analysis

- Visualize results by [plotly](https://plot.ly/python/getting-started/) framework

```bash
root@7bdb7335cde0:/RL_strategies# cd analysis/
root@7bdb7335cde0:/RL_strategies/analysis# python3 evaluation.py
```

## Acknowledgement

### Basically, the code is built upon the [TensorFlow Agents](https://github.com/tensorflow/agents) Framework.
``` shell
@article{hafner2017agents,
  title={TensorFlow Agents: Efficient Batched Reinforcement Learning in TensorFlow},
  author={Hafner, Danijar and Davidson, James and Vanhoucke, Vincent},
  journal={arXiv preprint arXiv:1709.02878},
  year={2017}
}
```
