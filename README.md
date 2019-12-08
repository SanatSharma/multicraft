# multicraft

This is our experiments on two different RL approaches in solving a multi-agent setting, in particular, a Tag scenario. 
The environment and scenario we used is from the [Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).

## Dependencies

- Python (3.5.4)
- OpenAI gym (0.10.5)
- Tensorflow (1.8.0)
- numpy (1.14.5)
- torch (1.3.1)

## Running Experiments
- under `/experiment/` directory, execute `train.py`: this executes two learning algorithms, 
SARSA and DDPG respectively, and trains 2000 episodes each. By default, the scenario is rendered every 100 episodes. 

- The motivation for our experiments is to compare the performance of learned agents against that of
 [Multi-Agent Deep Deterministic Policy Gradient (MADDPG)](https://github.com/openai/maddpg)
 and to validate MADDPG as well. 

- To run our MADDPG experiment (based on https://github.com/openai/maddpg): under `/maddpg/experiments/` directory, execute `train.py`

### Results
- SARSAR and DDPG experiments each produces a result `benchmark.csv`, stored at `/experiment/sarsa_out` and 
`/experiment/ddpg_out` espectively

- MADDPG's experiment result (`benchmark.csv`) is stored at `/maddpg/experiments/maddpg_out`
 

