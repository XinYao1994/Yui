import sys
from flyb import FlapBrid
import numpy as np

sys.path.insert(0, "play/brain")
try:
    from RL_brain import QLearningTable
    from RL_brain import SarsaTable
    from RL_brain import SarsaLambdaTable
    from DRL_brain import DeepQNetwork
    from DRL_brain import DDPG
except ImportError:
    print('No import')

RENDER = False

def update_DQN(RL, env):
    step = 0
    while True: #for episode in range(300):
        state = env.reset()
        while True:
            # fresh env
            if RENDER:
                env.render()
            action = RL.choose_action(state)
            state_, reward, done = env.step(action)
            RL.store_transition(state, action, reward, state_)
            if (step > 50) and (step % 5 == 0): # 200
                RL.learn()
            state = state_
            if done:
                print(env.score)
                break
            step += 1

if __name__ == "__main__":
    env = FlapBrid() 
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      )
    update_DQN(RL, env)

    # ddpg = DDPG(env.n_actions, env.n_features, a_bound)
    # env.after(100, update_DQN)
    # env.mainloop()
    # RL.plot_cost()



'''
def update_PPDG(ddpg, env):
    i = 0
    while True:
        s = env.reset()
        ep_reward = 0
        for j in range(2000):
            if RENDER:
                env.render()
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
            s_, r, done = env.step(a)
            ddpg.store_transition(s, a, r / 10, s_)
            if ddpg.pointer > 10000:
                var *= .9995    # decay the action randomness
                ddpg.learn()
            s = s_
            ep_reward += r
            if j == 1999 or done:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                break
'''

