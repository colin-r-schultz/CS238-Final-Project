from taxi import TaxiEnv

TEST_ICE_LOCS = [(3, 3), (1, 1), (1, 3), (1, 4)]
TEST_ICE_PROB = 0.5
TEST_NOISE = 0.5

env = TaxiEnv(None, state_uncertainty=TEST_NOISE)

state = env.reset()

num_trials = 10
num_steps = 200
rewards = []
for _ in range(num_trials):
    reward = 0
    for step in range(num_steps):

        # print(f"step: {step}")

        # sample a random action from the list of available actions
        action = env.action_space.sample()

        # perform this action on the environment
        _state, r, terminated, _truncated, _info = env.step(action)
        reward += r

        if terminated:
            break


    print(f"Reward = {reward}")
    rewards.append(reward)
    env.reset()
# end this instance of the taxi environment
env.close()
average = sum(rewards)/len(rewards)
print(f"Average reward = {average}")
