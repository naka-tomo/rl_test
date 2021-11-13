import numpy as np
import random

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
epsiron = 0.3   # ランダムに行動する割合
alpha = 0.1     # 学習率
gamma = 0.99    # 報酬の割引率

# 座標をQ値に変換するテーブル
q_table = np.zeros( (maze.shape[0], maze.shape[1], action_size) )

# 迷路で実行可能なaction
actions = [ 
    np.array([ 0,  1]), # 右
    np.array([ 0, -1]), # 左
    np.array([-1,  0]), # 上
    np.array([ 1,  0]), # 下
 ]

# Qテーブルの可視化
def show_q_table():
    for y in range(maze.shape[0]):
        for x in range( maze.shape[1] ):
            a = np.argmax( q_table[y, x] )

            if maze[y, x]==-1:
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

init_state = np.array( [1, 1] )     # 初期位置
current_state = np.array( [1, 1] )  # 現在位置
prev_state = np.array( [1, 1] )     # 1step前にいた位置

total_reward = 0
for i in range(1000):
    if random.random()<epsiron:
        # 価値が最大の行動
        action_idx = np.argmax( q_table[current_state[0], current_state[1]] )
    else:
        # ランダムに行動
        action_idx = random.randint( 0, 3 )

    # 移動
    prev_state[:] = current_state
    current_state += actions[ action_idx ]

    # 報酬
    reward = maze[ current_state[0], current_state[1] ]
    total_reward += reward

    # Q値更新
    max_q = np.max( q_table[current_state[0], current_state[1]] ) 
    prev_q = q_table[ prev_state[0], prev_state[1], action_idx ]
    q_table[prev_state[0], prev_state[1], action_idx] +=  alpha * (reward + gamma*max_q - prev_q )

    # 衝突したら一つ前の状態に戻す
    if reward==-1:
        current_state[:] = prev_state

    # ゴールしたら初期化
    if reward==1:
        print("reward:", total_reward)
        show_q_table()
        current_state[:] = init_state
        total_reward = 0
