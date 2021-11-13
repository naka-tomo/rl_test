import numpy as np
import random
import torch
import torch.nn as nn

# 迷路を定義：（1, 1）がスタート，(1, 5)がゴール
maze = np.array([
    [-1 ,-1, -1, -1, -1, -1, -1],
    [-1,  0 , 0, -1,  0,  1, -1],
    [-1,  0 , 0, -1,  0,  0, -1],
    [-1,  0 , 0, -1,  0,  0, -1],
    [-1,  0 , 0, -1,  0,  0, -1],
    [-1,  0 , 0,  0,  0,  0, -1],
    [-1 ,-1, -1, -1, -1, -1, -1]
])

action_size = 4
gamma = 0.99    # 報酬の割引率

# actionの確率を出力するpolicyネットワーク
policy_net = nn.Sequential( 
    nn.Linear(maze.shape[0]*maze.shape[1], 32),
    nn.Sigmoid(),
    nn.Linear(32, action_size),
    nn.Softmax(dim=-1)
)
optimizer = torch.optim.Adam( policy_net.parameters(), lr=0.001 )

# 迷路で実行可能なaction
actions = [ 
    np.array([ 0,  1]), # 右
    np.array([ 0, -1]), # 左
    np.array([-1,  0]), # 上
    np.array([ 1,  0]), # 下
 ]

# policyを可視化
def show_policy():
    for y in range(maze.shape[0]):
        for x in range( maze.shape[1] ):
            s = conv_to_onehot((y,x))
            a = np.argmax( policy_net( s ).detach().numpy()  ) 

            if maze[y,x]==-1:
                print("■", end="")
            elif maze[y, x]==1:
                print("★", end="")
            elif a==0:
                print( "→", end="" )
            elif a==1:
                print( "←", end="" )
            elif a==2:
                print( "↑", end="" )
            elif a==3:
                print( "↓", end="" )

        print()
    print("-----------------")

# 迷路の中の座標をワンホット化
def conv_to_onehot( state ):
    one_hot = np.zeros( maze.shape )
    one_hot[state[0], state[1]] = 1
    return torch.Tensor(one_hot.flatten())

init_state = np.array( [1, 1] )     # 初期位置
current_state = np.array( [1, 1] )  # 現在位置

for i in range(1000):
    current_state = init_state
    episode = []
    total_reward = 0
    while 1:
        # action予測
        s = conv_to_onehot(current_state)
        action_prob = policy_net( s )
        action_idx = np.random.choice( range(4), p=action_prob.detach().numpy()  )

        # 移動
        prev_state = current_state
        current_state =  current_state + actions[ action_idx ]

        # 報酬
        reward = maze[ current_state[0], current_state[1] ]
        total_reward += reward

        # 情報を保存
        episode.append( (s, reward, action_idx) )

        # 衝突したら一つ前の状態に戻す
        if reward==-1:
            current_state = prev_state

        # ゴールしたら抜ける
        if reward==1:
            break
            

    # policy更新
    for _ in range(5):
        accum_reward = 0
        loss = 0
        for s, r, a in episode[::-1]:
            # 割引累積報酬の期待値を最大化
            action_prob = policy_net( s )
            logprob = torch.log( action_prob[a] )

            accum_reward = r + gamma * accum_reward
            loss += -accum_reward * logprob

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 可視化
    if i%10==0:
        print("loss:", loss/len(episode))
        print("reward:", total_reward)
        show_policy()



