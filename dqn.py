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
epsiron = 0.3           # ランダムに行動する割合
gamma = 0.9             # 割引率
batch_size = 100        # バッチサイズ
max_buffer_size = 500   # バッファに貯めるデータの最大数

# Q値を学習するネットワーク
q_net = nn.Sequential( 
    nn.Linear(maze.shape[0]*maze.shape[1], 32),
    nn.Sigmoid(),
    nn.Linear(32, action_size),
)
optimizer = torch.optim.Adam( q_net.parameters(), lr=0.001 )

# 迷路で実行可能なaction
actions = [ 
    np.array([ 0,  1]), # 右
    np.array([ 0, -1]), # 左
    np.array([-1,  0]), # 上
    np.array([ 1,  0]), # 下
 ]

# Q値の可視化
def show_q_table():
    for y in range(maze.shape[0]):
        for x in range( maze.shape[1] ):
            s = conv_to_onehot((y,x))
            a = np.argmax( q_net( s ).detach().numpy()  ) 

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
prev_state = np.array( [1, 1] )     # 1step前にいた位置
replay_buffer = []                  # データ保存用バッファ
total_reward = 0

for i in range(2000):
    if random.random()>epsiron:
        # 価値が最大の行動
        s = conv_to_onehot(current_state)
        action_idx = np.argmax( q_net( s ).detach().numpy()  )
    else:
        # ランダムに行動
        action_idx = random.randint( 0, 3 )

    # 移動
    prev_state = current_state
    current_state =  current_state + actions[ action_idx ]

    # 報酬
    reward = maze[ current_state[0], current_state[1] ]
    total_reward += reward

    # バッファに貯める
    replay_buffer.append( (prev_state, current_state, action_idx, reward) )
    if len(replay_buffer)>max_buffer_size:
        replay_buffer.pop(0)

    # Qネットワーク更新
    loss = 0
    if len(replay_buffer)>batch_size:
        for _ in range(1):
            loss = 0
            for ps, cs, a, r in random.sample( replay_buffer, batch_size ): 
                cs_onehot = conv_to_onehot(cs)
                ps_onehot = conv_to_onehot(ps)
                max_q = np.max( q_net( cs_onehot ).detach().numpy() )
                prev_q = q_net( ps_onehot )[a]
                loss += (r + gamma * max_q - prev_q)**2

            # 誤差が小さくなる方向へパラメータを調整
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 可視化
    if i%100==0:
        print("loss", loss)
        print("reward:", total_reward)
        show_q_table()

    # 衝突したら一つ前の状態に戻す
    if reward==-1:
        current_state = prev_state

    # ゴールしたら初期化
    if reward==1:
        current_state = init_state
        total_reward = 0