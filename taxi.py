#pylint: disable=all
from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled


# class Mock():
#     def __init__(self, name, base, children=[]) -> None:
#         self.name = name
#         self.base = base
#         self.children = children

#     def __getattr__(self, name: str):
#         print(f"{self.name}.", end="")
#         if name in self.children:
#             return Mock(name, getattr(self.base, name))
#         print(name)
#         return getattr(self.base, name)
    
# pygame = Mock("pygame", pygame, ["display"])

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]
WINDOW_SIZE = (550, 350)


class TaxiEnv(Env):
    """
    The Taxi Problem involves navigating to passengers in a grid world, picking them up and dropping them
    off at one of four locations.

    ## Description
    There are four designated pick-up and drop-off locations (Red, Green, Yellow and Blue) in the
    5x5 grid world. The taxi starts off at a random square and the passenger at one of the
    designated locations.

    The goal is move the taxi to the passenger's location, pick up the passenger,
    move to the passenger's desired destination, and
    drop off the passenger. Once the passenger is dropped off, the episode ends.

    The player receives positive rewards for successfully dropping-off the passenger at the correct
    location. Negative rewards for incorrect attempts to pick-up/drop-off passenger and
    for each step where another reward is not received.

    Map:

            +---------+
            |R: | : :G|
            | : | : : |
            | : : : : |
            | | : | : |
            |Y| : |B: |
            +---------+

    From "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich [<a href="#taxi_ref">1</a>].

    ## Action Space
    The action shape is `(1,)` in the range `{0, 5}` indicating
    which direction to move the taxi or to pickup/drop off passengers.

    - 0: Move south (down)
    - 1: Move north (up)
    - 2: Move east (right)
    - 3: Move west (left)
    - 4: Pickup passenger
    - 5: Drop off passenger

    ## Observation Space
    There are 500 discrete states since there are 25 taxi positions, 5 possible
    locations of the passenger (including the case when the passenger is in the
    taxi), and 4 destination locations.

    Destination on the map are represented with the first letter of the color.

    Passenger locations:
    - 0: Red
    - 1: Green
    - 2: Yellow
    - 3: Blue
    - 4: In taxi

    Destinations:
    - 0: Red
    - 1: Green
    - 2: Yellow
    - 3: Blue

    An observation is returned as an `int()` that encodes the corresponding state, calculated by
    `((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination`

    Note that there are 400 states that can actually be reached during an
    episode. The missing states correspond to situations in which the passenger
    is at the same location as their destination, as this typically signals the
    end of an episode. Four additional states can be observed right after a
    successful episodes, when both the passenger and the taxi are at the destination.
    This gives a total of 404 reachable discrete states.

    ## Starting State
    The episode starts with the player in a random state.

    ## Rewards
    - -1 per step unless other reward is triggered.
    - +20 delivering passenger.
    - -10  executing "pickup" and "drop-off" actions illegally.

    An action that results a noop, like moving into a wall, will incur the time step
    penalty. Noops can be avoided by sampling the `action_mask` returned in `info`.

    ## Episode End
    The episode ends if the following happens:

    - Termination:
            1. The taxi drops off the passenger.

    - Truncation (when using the time_limit wrapper):
            1. The length of the episode is 200.

    ## Information

    `step()` and `reset()` return a dict with the following keys:
    - p - transition proability for the state.
    - action_mask - if actions will cause a transition to a new state.

    As taxi is not stochastic, the transition probability is always 1.0. Implementing
    a transitional probability in line with the Dietterich paper ('The fickle taxi task')
    is a TODO.

    For some cases, taking an action will have no effect on the state of the episode.
    In v0.25.0, ``info["action_mask"]`` contains a np.ndarray for each of the actions specifying
    if the action will change the state.

    To sample a modifying action, use ``action = env.action_space.sample(info["action_mask"])``
    Or with a Q-value based algorithm ``action = np.argmax(q_values[obs, np.where(info["action_mask"] == 1)[0]])``.


    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('Taxi-v3')
    ```

    ## References
    <a id="taxi_ref"></a>[1] T. G. Dietterich, “Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition,”
    Journal of Artificial Intelligence Research, vol. 13, pp. 227–303, Nov. 2000, doi: 10.1613/jair.639.

    ## Version History
    * v3: Map Correction + Cleaner Domain Description, v0.25.0 action masking added to the reset and step information
    * v2: Disallow Taxi start location = goal location, Update Taxi observations in the rollout, Update Taxi reward threshold.
    * v1: Remove (3,2) from locs, add passidx<4 check
    * v0: Initial version release
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    NOISE_PROB = 0.2
    ICE_PROPORTION = 0.2
    SLIP_PROB = 0.5

    ICE_COLOR = (181, 218, 255)

    N_ACTIONS = 6
    N_STATES = 500

    def __init__(self, render_mode: Optional[str] = None, model_uncertainty=False, state_uncertainty=False):
        self.desc = np.asarray(MAP, dtype="c")

        self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.locs_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]
        self.ice_locs = []
        self.model_uncertainty = model_uncertainty
        self.noise_prob = self.NOISE_PROB if state_uncertainty else 0
        self.ice_locs = []

        num_rows = 5
        num_columns = 5
        self.max_row = num_rows - 1
        self.max_col = num_columns - 1
        self.state_shape = [num_rows, num_columns, len(locs) + 1, len(locs)]

        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Discrete(self.N_STATES)

        self.render_mode = render_mode

        # pygame utils
        self.window = None
        self.clock = None
        self.cell_size = (
            WINDOW_SIZE[0] / self.desc.shape[1],
            WINDOW_SIZE[1] / self.desc.shape[0],
        )
        self.taxi_imgs = None
        self.taxi_orientation = 0
        self.passenger_img = None
        self.destination_img = None
        self.median_horiz = None
        self.median_vert = None
        self.background_img = None

    def next_pos(self, row, col, action):
        new_row, new_col = row, col
        if action == 0:
            new_row = min(row + 1, self.max_row)
        elif action == 1:
            new_row = max(row - 1, 0)
        if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
            new_col = min(col + 1, self.max_col)
        elif action == 3 and self.desc[1 + row, 2 * col] == b":":
            new_col = max(col - 1, 0)
        return (new_row, new_col)

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        return np.ravel_multi_index(([taxi_row, taxi_col, pass_loc, dest_idx]), self.state_shape)

    def decode(self, i):
        """Returns row, col, pass_loc, dest_idx
        """
        return np.unravel_index(i, self.state_shape)

    def action_mask(self, state: int):
        """Computes an action mask for the action space using the state information."""
        mask = np.zeros(6, dtype=np.int8)
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)
        if taxi_row < self.max_row:
            mask[0] = 1
        if taxi_row > 0:
            mask[1] = 1
        if taxi_col < self.max_col and self.desc[taxi_row + 1, 2 * taxi_col + 2] == b":":
            mask[2] = 1
        if taxi_col > 0 and self.desc[taxi_row + 1, 2 * taxi_col] == b":":
            mask[3] = 1
        if pass_loc < 4 and (taxi_row, taxi_col) == self.locs[pass_loc]:
            mask[4] = 1
        if pass_loc == 4 and (
            (taxi_row, taxi_col) == self.locs[dest_idx]
            or (taxi_row, taxi_col) in self.locs
        ):
            mask[5] = 1
        return mask
    
    def _add_noise(self, s):
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(s)
        if pass_loc < len(self.locs) and self.np_random.random() < self.noise_prob:
            pass_loc = self.np_random.integers(0, len(self.locs))
        return self.encode(taxi_row, taxi_col, pass_loc, dest_idx)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()
        return (int(self._add_noise(s)), r, t, False, {"prob": p, "action_mask": self.action_mask(s)})
    
    def setup_transitions(self):
        self.initial_state_distrib = np.zeros(self.N_STATES)
        self.P = {
            state: {action: [] for action in range(self.N_ACTIONS)}
            for state in range(self.N_STATES)
        }
        for row in range(self.max_row + 1):
            for col in range(self.max_col + 1):
                for pass_idx in range(len(self.locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(self.locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < 4 and pass_idx != dest_idx:
                            self.initial_state_distrib[state] += 1
                        for action in range(self.N_ACTIONS):
                            # defaults
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = (
                                -1
                            )  # default reward when there is no pickup/dropoff
                            terminated = False
                            taxi_loc = (row, col)
                            if action < 4 and taxi_loc in self.ice_locs:
                                transitions = []
                                for a in range(4):
                                    new_row, new_col = self.next_pos(row, col, a)
                                    p = 1.0 - self.SLIP_PROB if a == action else self.SLIP_PROB / 3
                                    new_state = self.encode(new_row, new_col, new_pass_idx, dest_idx)
                                    transitions.append((p, new_state, reward, terminated))
                            else:
                                if action < 4:
                                    new_row, new_col = self.next_pos(row, col, action)
                                elif action == 4:  # pickup
                                    if pass_idx < 4 and taxi_loc == self.locs[pass_idx]:
                                        new_pass_idx = 4
                                    else:  # passenger not at location
                                        reward = -10
                                elif action == 5:  # dropoff
                                    if (taxi_loc == self.locs[dest_idx]) and pass_idx == 4:
                                        new_pass_idx = dest_idx
                                        terminated = True
                                        reward = 20
                                    elif (taxi_loc in self.locs) and pass_idx == 4:
                                        new_pass_idx = self.locs.index(taxi_loc)
                                    else:  # dropoff at wrong location
                                        reward = -10
                                new_state = self.encode(
                                    new_row, new_col, new_pass_idx, dest_idx
                                )
                                transitions = [(1.0, new_state, reward, terminated)]
                            self.P[state][action].extend(transitions)
        self.initial_state_distrib /= self.initial_state_distrib.sum()

    def add_ice(self):
        self.ice_locs = []
        for row in range(self.max_row + 1):
            for col in range(self.max_col + 1):
                if (row, col) not in self.locs and self.np_random.random() < self.ICE_PROPORTION:
                    self.ice_locs.append((row, col))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if self.model_uncertainty:
            self.add_ice()
        self.setup_transitions()
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)

        self.lastaction = None
        self.taxi_orientation = 0

        if self.render_mode == "human":
            self.render()
        return int(self._add_noise(self.s)), {"prob": 1.0, "action_mask": self.action_mask(self.s)}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        elif self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame  # dependency to pygame only if rendering with human
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e
        
        from gymnasium.envs.toy_text import taxi
        file = taxi.__file__

        if self.window is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Taxi")
                self.window = pygame.display.set_mode(WINDOW_SIZE)
            elif mode == "rgb_array":
                pygame.display.set_mode(WINDOW_SIZE)
                self.window = pygame.Surface(WINDOW_SIZE)

        assert (
            self.window is not None
        ), "Something went wrong with pygame. This should never happen."
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.taxi_imgs is None:
            file_names = [
                path.join(path.dirname(file), "img/cab_front.png"),
                path.join(path.dirname(file), "img/cab_rear.png"),
                path.join(path.dirname(file), "img/cab_right.png"),
                path.join(path.dirname(file), "img/cab_left.png"),
            ]
            self.taxi_imgs = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.passenger_img is None:
            file_name = path.join(path.dirname(file), "img/passenger.png")
            self.passenger_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.destination_img is None:
            file_name = path.join(path.dirname(file), "img/hotel.png")
            self.destination_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
            self.destination_img.set_alpha(170)
        if self.median_horiz is None:
            file_names = [
                path.join(path.dirname(file), "img/gridworld_median_left.png"),
                path.join(path.dirname(file), "img/gridworld_median_horiz.png"),
                path.join(path.dirname(file), "img/gridworld_median_right.png"),
            ]
            self.median_horiz = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.median_vert is None:
            file_names = [
                path.join(path.dirname(file), "img/gridworld_median_top.png"),
                path.join(path.dirname(file), "img/gridworld_median_vert.png"),
                path.join(path.dirname(file), "img/gridworld_median_bottom.png"),
            ]
            self.median_vert = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.background_img is None:
            file_name = path.join(path.dirname(file), "img/taxi_background.png")
            self.background_img = pygame.transform.scale(
                pygame.image.load(file_name).convert(), self.cell_size
            )

        desc = self.desc

        for y in range(0, desc.shape[0]):
            for x in range(0, desc.shape[1]):
                cell = (x * self.cell_size[0], y * self.cell_size[1])
                self.window.blit(self.background_img, cell)
                if desc[y][x] == b"|" and (y == 0 or desc[y - 1][x] != b"|"):
                    self.window.blit(self.median_vert[0], cell)
                elif desc[y][x] == b"|" and (
                    y == desc.shape[0] - 1 or desc[y + 1][x] != b"|"
                ):
                    self.window.blit(self.median_vert[2], cell)
                elif desc[y][x] == b"|":
                    self.window.blit(self.median_vert[1], cell)
                elif desc[y][x] == b"-" and (x == 0 or desc[y][x - 1] != b"-"):
                    self.window.blit(self.median_horiz[0], cell)
                elif desc[y][x] == b"-" and (
                    x == desc.shape[1] - 1 or desc[y][x + 1] != b"-"
                ):
                    self.window.blit(self.median_horiz[2], cell)
                elif desc[y][x] == b"-":
                    self.window.blit(self.median_horiz[1], cell)

        for cell, color in zip(self.locs, self.locs_colors):
            color_cell = pygame.Surface(self.cell_size)
            color_cell.set_alpha(128)
            color_cell.fill(color)
            loc = self.get_surf_loc(cell)
            self.window.blit(color_cell, (loc[0], loc[1] + 10))
        
        for loc in self.ice_locs:
            color_cell = pygame.Surface(self.cell_size)
            color_cell.set_alpha(225)
            color_cell.fill(self.ICE_COLOR)
            loc = self.get_surf_loc(loc)
            self.window.blit(color_cell, (loc[0], loc[1] + 10))

        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        if pass_idx < 4:
            self.window.blit(self.passenger_img, self.get_surf_loc(self.locs[pass_idx]))

        if self.lastaction in [0, 1, 2, 3]:
            self.taxi_orientation = self.lastaction
        dest_loc = self.get_surf_loc(self.locs[dest_idx])
        taxi_location = self.get_surf_loc((taxi_row, taxi_col))

        if dest_loc[1] <= taxi_location[1]:
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
        else:  # change blit order for overlapping appearance
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )

        if mode == "human":
            for _ in pygame.event.get():
                pass
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def get_surf_loc(self, map_loc):
        return (map_loc[1] * 2 + 1) * self.cell_size[0], (
            map_loc[0] + 1
        ) * self.cell_size[1]

    def _render_text(self):
        desc = self.desc.copy().tolist()
        outfile = StringIO()

        out = [[c.decode("utf-8") for c in line] for line in desc]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x
        
        for (r, c) in self.ice_locs:
            out[r + 1][2 * c + 1] = "~"

        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), "green", highlight=True
            )

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                f"  ({['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'][self.lastaction]})\n"
            )
        else:
            outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


def plot_run_data(reward_list, data_title, steps_or_runs):
    xpoints = np.arange(1, len(reward_list) + 1)
    ypoints = np.array(reward_list)

    plt.plot(xpoints, ypoints)

    plt.title(data_title)
    if steps_or_runs == "steps":
        plt.xlabel("Step #")
        plt.ylabel("Reward After Step")
    elif steps_or_runs == "runs":
        plt.xlabel("Run #")
        plt.ylabel("Reward After Run")
    plt.show()

def train_q_learning(env, learn_rate, discount_rate=0.9, decay_rate=0.0005, num_runs=2000, max_steps=100):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q_table = np.zeros((state_size, action_size))
    epsilon = 0.1
    reward_list = []
    steps_list = []
    for run in range(num_runs):
        epsilon = np.exp(-decay_rate * run)
        print(f"Run #{run}".format(run + 1))
        state = env.reset()[0]
        terminated = False
        rewards = 0
        steps = 0
        while not terminated:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            observation, reward, terminated, truncated, info = env.step(action)
            q_table[state, action] = (1 - learn_rate)*q_table[state,action] + learn_rate*(reward + discount_rate*np.max(q_table[observation]))
            state = observation
            rewards += reward
            steps += 1
            if steps > max_steps:
                break
        steps_list.append(steps)
        reward_list.append(rewards)
    print(f"Average number of steps per run: {np.mean(np.array(steps_list))}")
    plot_run_data(reward_list, "Rewards for Q-Learning Training Runs", "runs")
    return q_table

def run_q_learning(env, q_table, num_runs=100, max_steps=100, render_flag=False):
    reward_list = []
    # frames = [env.render()]
    for i in range(num_runs):
        step = 0
        state, _ = env.reset()
        rewards = 0
        terminated = False
        while not terminated:
            # print(f"Step {step}".format(step + 1))
            action = np.argmax(q_table[state])
            observation, reward, terminated, truncated, info = env.step(action)
            rewards += reward
            # if render_flag:
            #     frames.append(env.render())
            # print(f"Score: {rewards}")
            state = observation
            step += 1
            if step > max_steps:
                break
        reward_list.append(rewards)

    print(f"Avg Score: {np.mean(reward_list)}")
    print(f"Number of steps taken: {step}")
    # frames = list(map(Image.fromarray, frames))
    # frames[0].save("out.gif", save_all=True, append_images=frames[1:])
    # plot_run_data(reward_list, "Rewards for Final Trained Q-Learning Run", "steps")
    return rewards

def run_random_agent(env, num_runs=100, max_steps=100):
    reward_list = []
    for _ in range(num_runs):
        env.reset()
        rewards = 0

        for step in range(max_steps):
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)
            rewards += reward

            if terminated:
                break
        reward_list.append(rewards)
    
    
    print(f"Avg Score: {np.mean(rewards)}")

def update_models(state, a, r, obs, transition_counts, reward_sums):
    transition_counts[state,a,obs] += 1
    reward_sums[state, a] += r

def update_value_fn(value_fn, points, transition_counts, reward_sums, gamma):
    n = np.maximum(np.sum(transition_counts[points, :, :], axis=-1), 1e-6)
    r = reward_sums[points, :] / n
    t = transition_counts[points, :, :] / n[..., None]
    value_fn[points] = np.max(r + gamma * np.sum(t * value_fn, axis=2), axis=1)

def best_action(value_fn, state, transition_counts, reward_sums, gamma):
    n = np.sum(transition_counts[state, :, :], axis=-1)
    r = reward_sums[state, :] / n
    t = transition_counts[state, :, :] / n[..., None]
    return np.argmax(r + gamma * np.sum(t * value_fn, axis=1))

def train_mle_model(env, discount_rate=0.9, decay_rate=0.005, num_runs=2000, max_steps=100):
    num_random_points = 15
    state_size = env.observation_space.n
    action_size = env.action_space.n
    transition_counts = np.zeros((state_size, action_size, state_size), np.float32)
    reward_sums = np.zeros((state_size, action_size), np.float32)
    value_fn = np.zeros(state_size, np.float32)
    for run in range(num_runs):
        epsilon = np.exp(-decay_rate * run)
        print(f"Run #{run + 1}, avg_val={np.mean(value_fn)}")
        obs, _ = env.reset()
        # b = update_belief(env, np.ones(len(env.locs) + 1, float), None, obs)
        state = obs
        for step in range(max_steps):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = best_action(value_fn, state, transition_counts, reward_sums, discount_rate)
            
            obs, reward, terminated, _, _ = env.step(action)
            update_models(state, action, reward, obs, transition_counts, reward_sums)
            update_points = np.r_[state, np.random.randint(0, state_size, num_random_points)]
            update_value_fn(value_fn, update_points, transition_counts, reward_sums, discount_rate)
            state = obs

            if terminated:
                break

    return {
        "transition_counts": transition_counts,
        "reward_sums": reward_sums,
        "value_fn": value_fn
    }

def run_mle_model(env, model, discount_rate, num_runs=100, max_steps=100):
    reward_list = []
    # frames = [env.render()]
    for i in range(num_runs):
        step = 0
        state, _ = env.reset()
        rewards = 0
        terminated = False
        while not terminated:
            # print(f"Step {step}".format(step + 1))
            action = best_action(model["value_fn"],state, model["transition_counts"], model["reward_sums"], discount_rate)
            observation, reward, terminated, truncated, info = env.step(action)
            rewards += reward
            # if render_flag:
            #     frames.append(env.render())
            # print(f"Score: {rewards}")
            state = observation
            step += 1
            if step > max_steps:
                break
        reward_list.append(rewards)

    print(f"Avg Score: {np.mean(reward_list)}")
    print(f"Number of steps taken: {step}")
    # frames = list(map(Image.fromarray, frames))
    # frames[0].save("out.gif", save_all=True, append_images=frames[1:])
    # plot_run_data(reward_list, "Rewards for Final Trained Q-Learning Run", "steps")
    return rewards


# Taxi rider from https://franuka.itch.io/rpg-asset-pack
# All other assets by Mel Tillery http://www.cyaneus.com/

if __name__ == "__main__":
    # env = gym.make("Taxi-v3", render_mode="human")
    env = TaxiEnv("rgb_array", model_uncertainty=False, state_uncertainty=False)

    # uncomment next line if you want to run Random Agent
    # run_random_agent(env)

    # uncomment next 2 lines if you want to run Q-Learning

    # q_table = train_q_learning(env, 0.1, 0.9)
    # run_q_learning(env, q_table)

    model = train_mle_model(env, 0.9)
    run_mle_model(env, model, 0.9)
