# Backetball Strategies with GAIL and BatchPPO

## Environment Setting

- docker:

```bash
nvidia-docker run --name jay -it -p 127.0.0.1:6006:6006 -v {repo_path}RL_strategies/:/RL_strategies -w /RL_strategies jay/bball_gail bash
```

## Getting Started

- Clone this repo:

```bash
git clone http://140.113.210.14:30000/nba/RL_strategies.git
cd RL_strategies
```

- Download the dataset. (You should put dataset right under the folder "{repo_path}/RL_strategies/data/")

```bash
cd data
wget http://140.113.210.14:6006/NBA/data/FPS5.npy
wget http://140.113.210.14:6006/NBA/data/FPS5Length.npy
```

## Data Preprocessing

- Filter out bad data.

```bash
cd bball_strategies/data/
python3 preprocess.py
```

- Create training dataset for GAIL. (transform data into state-action pair.)

```bash
python3 create_gail_data.py
```

- Reorder the data offense and defense positions by rule-based method.
- And, duplicate OrderedGAILTransitionData_52.hdf5 for multi process settings.

```bash
python3 postprocess_data_order.py 
cp OrderedGAILTransitionData_52.hdf5 OrderedGAILTransitionData_522.hdf5
```

## Training

### Configuration

All configurations are in "{repo_path}/RL_strategies/bball_strategies/scripts/gail/config.py"


### Train Model

```bash
cd {repo_path}/RL_strategies/
python -m bball_strategies.scripts.gail.train --config=double_curiculum
```

### Monitor training

```bash
tensorboard --logdir='logdir/gail_defense/{time stamp}-double_curiculum'
```

### Basically, the code is built upon the TensorFlow Agents Framework.
``` shell
@article{hafner2017agents,
  title={TensorFlow Agents: Efficient Batched Reinforcement Learning in TensorFlow},
  author={Hafner, Danijar and Davidson, James and Vanhoucke, Vincent},
  journal={arXiv preprint arXiv:1709.02878},
  year={2017}
}
```