import numpy as np

# 定义环境参数
n_states = 5  # 状态数量
actions = ["left", "right"]  # 动作集
q_table = np.zeros((n_states, len(actions)))  # Q表，初始值为0
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索率

# 定义奖励函数
def get_reward(state):
    if state == n_states - 1:
        return 1  # 目标位置奖励为1
    else:
        return 0  # 其他位置奖励为0

# 定义状态转移函数
def take_action(state, action):
    if action == "left":
        next_state = max(0, state - 1)  # 向左移动
    else:
        next_state = min(n_states - 1, state + 1)  # 向右移动
    reward = get_reward(next_state)
    return next_state, reward

# Q-learning算法
def q_learning(num_episodes):
    for episode in range(num_episodes):
        state = 0  # 每轮从起始位置开始
        done = False

        while not done:
            # 选择动作（根据ε-贪婪策略）
            if np.random.rand() < epsilon:
                action_idx = np.random.choice(len(actions))  # 探索
            else:
                action_idx = np.argmax(q_table[state, :])  # 利用

            action = actions[action_idx]
            next_state, reward = take_action(state, action)

            # 更新Q值
            best_next_action = np.argmax(q_table[next_state, :])
            td_target = reward + gamma * q_table[next_state, best_next_action]
            td_error = td_target - q_table[state, action_idx]
            q_table[state, action_idx] += alpha * td_error

            # 状态更新
            state = next_state

            # 检查是否到达终点
            if state == n_states - 1:
                done = True

        # 输出每个回合的学习情况
        print(f"Episode {episode + 1}: Q-table\n{q_table}\n")

# 运行Q-learning算法
num_episodes = 10  # 训练10轮
q_learning(num_episodes)
