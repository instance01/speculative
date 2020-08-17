import numpy as np


class Env:
    def __init__(self):
        self.steps = 0
        self.tot_reward = 0

    def step(self, action):
        self.steps += 1
        if self.steps > 100:
            return 0, True
        action -= .5
        if abs(action) <= .01:
            action = .01
        reward = 0.1 / action
        # TODO
        # reward = 1 / abs(action)
        self.tot_reward += reward
        return reward, self.tot_reward > 1000

    def reset(self):
        self.steps = 0
        self.tot_reward = 0


best_actions = []

def sample_path(eps):
    global best_actions
    best_actions = sorted(best_actions)[::-1]
    env = Env()

    actions = []
    tot_reward = 0

    idx = 0
    greedy = np.random.random() > eps

    done = False
    while not done:
        if greedy:
            action = best_actions[0][idx]
            idx += 1
        else:
            action = np.random.random()
        reward, done = env.step(action)
        tot_reward += reward
        actions.append(action)

    best_actions.append((tot_reward, np.array(actions)))


def doit(new_actions_, curr_best_reward):
    env = Env()
    done = False
    tot_reward = 0
    for action in new_actions_:
        reward, done = env.step(action)
        tot_reward += reward
        if done:
            break
    if tot_reward > curr_best_reward:
        curr_best_reward = tot_reward
        return True, curr_best_reward, tot_reward
    return False, curr_best_reward, tot_reward


def update():
    global best_actions
    best_actions = sorted(best_actions)[::-1]
    size_ = min(best_actions[0][1].shape[0], best_actions[1][1].shape[0])
    gradient = best_actions[0][1][:size_] - best_actions[1][1][:size_]
    curr_best_reward = best_actions[0][0]
    new_actions = best_actions[0][1].copy()

    print(new_actions[:5], curr_best_reward)

    # print('curr best reward', curr_best_reward)

    tot_reward = -1
    for i in range(size_):
        # print(new_actions[:5], curr_best_reward)
        results = []
        tries = [0.5, 0.3, 0.1, 0.01, -0.5, -0.3, -0.1, -0.01]
        # TODO of course here you should sort by best reward
        for try_ in tries:
            curr_new_actions = new_actions.copy()
            curr_new_actions[i] += try_
            result = doit(curr_new_actions.copy(), curr_best_reward)
            if result[0]:
                curr_best_reward = result[1]
                tot_reward = result[2]
                new_actions[i] = curr_new_actions[i]
                break

    print(new_actions[:5])

    if tot_reward != -1:
        best_actions = sorted(best_actions)[::-1]
        best_actions.pop(0)
        best_actions.append((tot_reward, new_actions))


sample_path(1.0)
sample_path(1.0)
sample_path(1.0)
sample_path(1.0)
sample_path(1.0)
for i in range(20):
    update()
