import numpy as np
import random
from numpy.random.mtrand import sample
import torch
import torch.nn as nn
import torch.nn.functional as F

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
alpha = 1.0     # エントロピー正則化項の係数

# actionの確率と状態価値を出力するネットワーク
class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(maze.shape[0]*maze.shape[1], 32)
        self.fc2 = nn.Linear(32, action_size)
    
    def forward(self, s):
        h = F.relu( self.fc1( s ) )
        action_prob = F.softmax( self.fc2(h) )
        return action_prob

class CriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(maze.shape[0]*maze.shape[1]+action_size, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, s, a):
        h = torch.cat([s, a], dim=-1)
        h = F.relu( self.fc1( h ) )
        value = self.fc2(h)
        return value

actor_net = ActorNet()
critic_net = CriticNet()
optimizer_actor = torch.optim.Adam( actor_net.parameters(), lr=0.001 )
optimizer_critic = torch.optim.Adam( critic_net.parameters(), lr=0.01 )

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
            action_prob = actor_net( s )
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

# reparameterization trickでサンプリング
def sample_action(prob): 
    g = -torch.log( -torch.log(torch.rand( action_size )))
    s = F.softmax(( (torch.log(prob) + g)/0.1 ))
    return s

init_state = np.array( [1, 1] )     # 初期位置
current_state = np.array( [1, 1] )  # 現在位置

for i in range(5000):
    current_state = init_state
    episode = []
    total_reward = 0

    step = 0
    while 1:
        # action予測
        s = conv_to_onehot(current_state)
        action_prob = actor_net( s ).detach().numpy() 
        action_prob += 0.2
        action_prob = action_prob/np.sum(action_prob)
        action_idx = np.random.choice( range(4), p=action_prob  )
        #action_idx = np.argmax( action_prob.detach().numpy() )

        # 移動
        prev_state = current_state
        current_state =  current_state + actions[ action_idx ]

        # 報酬
        reward = maze[ current_state[0], current_state[1] ]
        total_reward += reward

        # 情報を保存
        episode.append( (s, action_idx, conv_to_onehot(current_state), reward) )

        # 衝突したら一つ前の状態に戻す
        if reward==-1:
            current_state = prev_state

        # ゴールしたら抜ける
        if reward==1:
            break


    # actor, critic更新
    loss_critic = 0
    loas_actor = 0
    for _ in range(1):
        # Q値を学習
        loss_critic = 0
        for s, a, next_s, r in episode:
            q_value = critic_net( s, torch.eye(action_size)[a] )                    # Q(s_t, a_t)
            pi_next = actor_net( next_s ).detach()                                  # pi( : | s_t+1)
            next_a = sample_action(pi_next)                                         # a_t+1 ~ pi( : | s_t+1 )
            next_a_prob = pi_next[torch.argmax(next_a)]                             # pi( a_t+1 | s_t+1 )
            q_value_next = critic_net( next_s, next_a ).detach()                    # Q(s_t+1, a_t+1 )
            v = next_a_prob * (q_value_next - alpha*torch.log(next_a_prob))   # v(s+1) = E_pi(a|s_t+1) [ Q(s_t+1, a ) - log(pi(a|s+1)) ]
            loss_critic += (r + gamma*v - q_value)**2                               # J = E[ (r(s_t, a_t) + gamma * V(s_t+1) -Q(s_t, a_t))^2 ]
        optimizer_critic.zero_grad()
        loss_critic.backward()
        optimizer_critic.step()

        # Q値が最大となるactionを学習
        loss_actor = 0
        for s, _, _, _ in episode:
            pi = actor_net( s )
            a = sample_action( pi )         # a_t ~ pi(a | s_t )  : reparameterizaton trick
            a_prob = pi[torch.argmax(a)]    # pi( a_t | s_t )
            loss_actor += -critic_net( s, a ) + alpha*torch.log( a_prob ) # J = E[ -Q( s_t, a) +  log( pi(a|s_t) ) ]
        optimizer_actor.zero_grad()
        loss_actor.backward()
        optimizer_actor.step()

    # 可視化
    if i%10==0:
        print("critic loss:", loss_critic/len(episode))
        print("actor loss:", loss_actor/len(episode))
        print("reward:", total_reward)
        show_policy()
