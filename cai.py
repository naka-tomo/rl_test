import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.random.mtrand import standard_t


# 迷路を定義：（1, 1）がスタート，(1, 5)がゴール
maze = np.array([
    [-1 ,-1, -1, -1, -1, -1, -1],
    [-1,  0 , 0, -1,  0,  1, -1],
    [-1,  0 , 0, -1,  0,  0, -1],
    [-1,  0 , 0, -1,  0,  0, -1],
    [-1,  0 , 0,  0,  0,  0, -1],
    [-1,  0 , 0,  0,  0,  0, -1],
    [-1 ,-1, -1, -1, -1, -1, -1]
])

action_size = 4
alpha = 0.1     # 学習率

# 座標をQ値に変換するテーブル
q_table = np.zeros( (maze.shape[0], maze.shape[1], action_size) )
v_table = np.zeros( (maze.shape[0], maze.shape[1]) )

# 遷移回数と行動洗濯回数
trans_count = np.zeros( (maze.shape[0], maze.shape[1], action_size, maze.shape[0], maze.shape[1]) )
action_count = np.zeros( (maze.shape[0], maze.shape[1], action_size) )

# 迷路で実行可能なaction
actions = [ 
    np.array([ 0,  1]), # 右
    np.array([ 0, -1]), # 左
    np.array([-1,  0]), # 上
    np.array([ 1,  0]), # 下
 ]

# Qテーブルの可視化
def show_q_table(table, pos=None):
    for y in range(maze.shape[0]):
        for x in range( maze.shape[1] ):
            a = np.argmax( table[y, x] )

            if maze[y, x]==-1:
                print("■", end="")
            elif maze[y, x]==1:
                print("★", end="")
            elif pos!=None and pos[0]==y and pos[1]==x:
                print( "●", end="" )
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


def softmax(p):
    p = np.exp(p-np.max(p))
    return p/np.sum(p)

total_reward = 0
for i in range(1000):
    # 最適方策p(a|s)を計算しサンプリング
    action_prob = softmax( q_table[current_state[0], current_state[1]] - v_table[current_state[0], current_state[1]] )
    action_idx = np.random.choice( range(4), p=action_prob )

    # 移動
    prev_state[:] = current_state
    current_state += actions[ action_idx ]

    # 報酬
    reward = maze[ current_state[0], current_state[1] ]
    total_reward += reward

    # 遷移確率・行動選択確率更新
    ps = prev_state
    cs = current_state
    a = action_idx
    trans_count[ ps[0], ps[1], a, cs[0], cs[1] ] += 1
    action_count[ ps[0], ps[1], a] += 1

    # p(s_t+1 | s_t, a_t)計算
    trans_prob = trans_count[ ps[0], ps[1], action_idx ] + 0.1
    trans_prob = trans_prob / np.sum(trans_prob)

    # p(a_t | s_t )計算
    action_prob = action_count[ ps[0], ps[1] ] + 0.1
    action_prob = action_prob / np.sum(action_prob)

    # V, Q更新
    # V = log E[ exp Q[s_t, a_t] ]
    # Q = r + log E[ exp V[s_t+1] ]
    v_table[ ps[0], ps[1] ] = np.log( np.sum( action_prob * np.exp(q_table[ps[0], ps[1]]) ) )
    q_table[ ps[0], ps[1], action_idx ] = reward + 0.1*np.log( np.sum( trans_prob * np.exp(v_table) ) )

    # 衝突したら一つ前の状態に戻す
    if reward==-1:
        current_state[:] = prev_state

    # ゴールしたら初期化
    if reward==1:
        print("reward:", total_reward)
        show_q_table( q_table-v_table.reshape((maze.shape[0], maze.shape[1], 1)) )
        current_state[:] = init_state
        total_reward = 0


# 学習モデルを利用したプランニング: (2, 5)から(1, 1)への経路
init_state = np.array( [2, 5] )     # 初期位置
goal_state = np.array( [1, 1] )     # ゴール位置
current_state[:] = init_state
num_step = 10
q_table_t = np.zeros( (num_step, maze.shape[0], maze.shape[1], action_size) )
v_table_t = np.zeros( (num_step, maze.shape[0], maze.shape[1]) )
beta = np.zeros( (num_step, maze.shape[0], maze.shape[1]) )

# 初期の価値を定義(caiでは最適性定義のため報酬は負)
v_table_t[num_step-1, : , : ] = -1
v_table_t[num_step-1, goal_state[0], goal_state[1] ] = 0
beta[num_step-1] = np.exp(v_table_t[num_step-1])
beta[num_step-1] = beta[num_step-1] / np.sum(beta[num_step-1])

# backward message計算
for t in range(num_step-2,-1,-1):
    reward = 0
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            for a in range(action_size):
                trans_prob = trans_count[ y, x, a ] + 0.1
                trans_prob = trans_prob / np.sum(trans_prob)
                q_table_t[t, y, x, a] = reward + np.log( np.sum( trans_prob * np.exp(v_table_t[t+1]) ) )


    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            action_prob = action_count[ y, x ]
            if np.sum(action_prob)>0:
                action_prob = action_prob / np.sum(action_prob)
            v_table_t[t, y, x] = np.log( np.sum( action_prob * np.exp(q_table_t[t, y, x]) )+1e-100 )

    beta[t] = np.exp(v_table_t[t])
    beta[t] = beta[t] / np.sum(beta[t])

# foward message計算
alpha = np.zeros( (num_step, maze.shape[0], maze.shape[1]) )
alpha[0, init_state[0], init_state[1]] = 1.0
for t in range(1,num_step):
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            for a in range(action_size):
                trans_prob = np.copy(trans_count[ y, x, a ]) + 0.1
                if np.sum(trans_prob):
                    trans_prob = trans_prob / np.sum(trans_prob)
                alpha[t] += trans_prob * alpha[t-1, y, x]
    alpha[t] = alpha[t]/np.sum(alpha[t])


# 最適行動選択
s = list(init_state)
print(f"time={t}, state={s}")
show_q_table( q_table_t[0]-v_table_t[0].reshape((maze.shape[0], maze.shape[1], 1)), s )
for t in range(1,num_step):
    optimality = alpha[t] * beta[t]

    # 移動可能な状態の中から最適性が最大のものを選択
    movable = np.sum( trans_count[ s[0], s[1] ] , axis=0 )>0
    s = np.unravel_index(np.argmax(optimality*movable), optimality.shape)

    print(f"time={t}, state={s}")
    show_q_table( q_table_t[t]-v_table_t[t].reshape((maze.shape[0], maze.shape[1], 1)), s )



