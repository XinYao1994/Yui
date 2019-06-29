import sys


#from maze_env_rl import Maze
from maze_env import Maze

#sys.path.insert(0, "c:/SAVE/Project/Yui/play/brain")
sys.path.insert(0, "play/brain")
try:
    from RL_brain import QLearningTable
    from RL_brain import SarsaTable
    from RL_brain import SarsaLambdaTable
    from DRL_brain import DeepQNetwork
except ImportError:
    print('No import')

def update_Q():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

def update_Sarsa():
    for episode in range(100):
        # initial observation
        observation = env.reset()
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on observation
            action_ = RL.choose_action(str(observation_))


            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

def update_Sarsa_lambda():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        # initial all zero eligibility trace
        RL.eligibility_trace *= 0

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

def update_DQN():
    step = 0

    for episode in range(300):
        # initial observation
        observation = env.reset()
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # print(action)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # print(observation_)
            # print(reward)
            # print(action)

            
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                raw_text = input("hint >>> ")
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    # RL = QLearningTable(actions=list(range(env.n_actions)))
    # RL = SarsaTable(actions=list(range(env.n_actions)))
    # RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    env.after(100, update_DQN)
    env.mainloop()
    RL.plot_cost()
