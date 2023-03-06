import gymnasium as gym
import time


env = gym.make("Taxi-v3", render_mode="human").env

state = env.reset()

num_steps = 200
rewards = []
for _ in range(10):
    reward = 0
    for step in range(num_steps):

        print(f"step: {step}")

        # sample a random action from the list of available actions
        action = env.action_space.sample()

        # perform this action on the environment
        state = env.step(action)
        reward += state[1]

        # print the new state
        env.render()


    print(f"Reward = {reward}")
    rewards.append(reward)
    env.reset()
# end this instance of the taxi environment
env.close()
average = sum(rewards)/10
print(f"Average reward = {average}")
