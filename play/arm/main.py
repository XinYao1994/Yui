import sys

#from maze_env_rl import Maze


#sys.path.insert(0, "c:/SAVE/Project/Yui/play/brain")
sys.path.insert(0, "play/brain")
sys.path.insert(0, "play/arm")

try:
    from env import ArmEnv
    from DRL_brain import ArmDDPG
except ImportError:
    print('No import')

# global parameters
MAX_EPISODES = 1000
MAX_EP_STEPS = 200
ON_TRAIN = True
ON_TRAIN = False

# init env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# RL method
'''
连续动作 (动作是一个连续值, 比如旋转角度)
Policy gradient
DDPG
A3C
PPO
离散动作 (动作是一个离散值, 比如向前,向后走)
Q-learning
DQN
A3C
PPO
'''
rl = ArmDDPG(a_dim, s_dim, a_bound)
def train():
    succ_con = 0
    # start training
    rl.restore()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            # env.render()
            a = rl.choose_action(s)
            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            #print(i)
            #print(done)
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | steps: %i' % (i, '---' if not done else 'done', ep_r, j))
                if done:
                    succ_con = succ_con + 1
                break
    rl.save()
    print("Success Rate: "+str(succ_con/10.0)+"%")

def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done = env.step(a)


if ON_TRAIN:
    train()
else:
    eval()



                 



