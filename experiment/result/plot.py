import argparse
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def arg_parse():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--data_path', type=str, default='', help='Path to the data set')
    parser.add_argument('--save_path', type=str, default='', help='Path to save graph')

    args = parser.parse_args()
    return args

def process_data (data_path):
    results = pd.read_csv(data_path)

    average_reward = results['Average Reward'].values
    adversary_reward = results['Episode Adversary Reward'].values

    # Mean average reward and episode reward for every 100 episodes
    reshape_val = 100
    average_reward_mean = np.mean(average_reward.reshape(-1, reshape_val), axis=1)
    average_adversary_reward = np.mean(adversary_reward.reshape(-1, reshape_val), axis=1)
    episodes = [i for i in range(0, len(average_reward), reshape_val)]
    
    graph_data = [(episodes, average_reward_mean), (episodes, average_adversary_reward)]
    return graph_data

def plot_prediction(graph_data, args):
    
    plt.subplot(211, xlabel='Episodes', ylabel='Average Reward')
    plt.plot(graph_data[0][0], graph_data[0][1], 'r')
    plt.legend()
    plt.subplot(212, xlabel='Episodes', ylabel='Adversary Reward')
    plt.plot(graph_data[1][0], graph_data[1][1], 'b')
    plt.legend()
    if args.save_path != '':
        plt.savefig(args.save_path)
    else:
        plt.show()



if __name__ == '__main__':
    args = arg_parse()
    print(args)
    
    graph_data = process_data(args.data_path)
    #print(graph_data)
    plot_prediction(graph_data, args)