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

# actionの確率と状態価値を出力するネットワーク
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(maze.shape[0]*maze.shape[1], 32)
        self.layer_policy = nn.Linear(32, action_size)
        self.layer_value = nn.Linear(32, 1 )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.softmax( self.layer1( x ) )
        value = self.layer_value( x )
        action_prob = self.softmax( self.layer_policy(x) )

        return action_prob, value

policy_net = PolicyNet()
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
            action_prob, value = policy_net( s )
            a = np.argmax( action_prob.detach().numpy()  ) 

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

for i in range(5000):
    current_state = init_state
    episode = []
    total_reward = 0
    while 1:
        # action予測
        s = conv_to_onehot(current_state)
        action_prob, state_value = policy_net( s )
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
            
    # 累積割引報酬を計算
    accum_reward = 0
    rewards = []
    for _ ,r ,_ in episode[::-1]:
        accum_reward = r + gamma * accum_reward
        rewards.insert(0, accum_reward )

    # policy更新
    for _ in range(5):
        # policyとvalue関数のlossを計算
        loss_policy = 0
        loss_value = 0
        for i, (s, r, a) in enumerate(episode):
            action_prob, state_value = policy_net( s )
            advantage = rewards[i] - state_value.detach()
            loss_policy += -torch.log( action_prob[a] )*advantage
            loss_value += (rewards[i] - state_value)**2

        loss = loss_policy + loss_value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 可視化
    if i%10==0:
        print("loss:", loss/len(episode))
        print("reward:", total_reward)
        show_policy()



