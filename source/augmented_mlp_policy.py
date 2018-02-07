from source.augmented_trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import rllab.misc.logger as logger
import copy
import numpy as np
import time
import pickle

global episodic_returns


def plot(fig, hl11, ax2, episodic_returns):
    batch_size = 1
    print ("Started plotting routine")
    import matplotlib.pyplot as plt
    count = 0
    if len(episodic_returns)>0:
        returns = np.array(episodic_returns).copy()
        # plot learning curve
        hl11.set_ydata(returns)
        hl11.set_xdata(batch_size*np.arange(len(returns)))
        ax2.set_ylim([np.min(returns), np.max(returns)+5])
        ax2.set_xlim([0, len(returns)*batch_size])
        fig.canvas.draw()
        fig.canvas.flush_events()
    fig.canvas.draw()
    fig.canvas.flush_events()
    count += 1
    time.sleep(0.01)

def augment_paths(env, paths, repeats=10):
    augmented_paths = []
    for path in paths:
        augmented_paths.append(path)
        for _ in range(repeats):
            new_path = copy.copy(path)
            env.wrapped_env.env.env.reset_model()
            fake_target = env.wrapped_env.env.env.get_body_com("target")
            for step in range(len(paths[0]["rewards"])):
                vec = path["end_effectors"][step] - fake_target
                reward_dist = - np.linalg.norm(vec)
                reward_ctrl = - np.square(path["actions"][step]).sum()
                reward = reward_dist + reward_ctrl
                new_path["rewards"][step] = reward
                new_path["observations"][step][-3:] = vec.flatten()
            augmented_paths.append(new_path)
    return augmented_paths


def augment_train(self, episodic_returns, run_num, augment, batchsize):
    self.start_worker()
    self.init_opt()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 6))
    ax2 = fig.add_subplot(111)
    plt.ion()
    hl11, = ax2.plot([], [])
    plt.grid()
    plt.pause(0.01)
    for itr in range(self.current_itr, self.n_itr):
        with logger.prefix('itr #%d | ' % itr):
            paths = self.sampler.obtain_samples(itr)
            average_return = np.mean([sum(path["rewards"]) for path in paths])
            episodic_returns.append(average_return)
            plot(fig, hl11, ax2, episodic_returns)
            if augment:
                new_paths = augment_paths(self.env, paths, repeats=10)
            else:
                new_paths = paths
            #logger.dump_tabular(with_prefix=False)
            samples_data = self.sampler.process_samples(itr, new_paths)
            print(len(paths), len(new_paths), len(samples_data))
            start = time.time()
            self.optimize_policy(itr, samples_data)
            print("Optimization time", time.time() - start)
            self.current_itr = itr + 1
    with open("../data/new_run_num_%d_augment_%d_batchsize_%d.pkl" % (run_num, int(augment), batchsize), 'wb') as f:
        pickle.dump(episodic_returns, f)
    plt.close()
    self.shutdown_worker()

def run_task(run_num, batchsize, augment=False):
    # Please note that different environments with different action spaces may require different
    # policies. For example with a Box action space, a GaussianMLPPolicy works, but for a Discrete
    # action space may need to use a CategoricalMLPPolicy (see the trpo_gym_cartpole.py example)
    env = normalize(GymEnv("Reacher-v1"))
    episodic_returns = []
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        adaptive_std=True,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batchsize,
        max_path_length=env.horizon,
        n_itr=50,
        discount=0.99,
        step_size=0.05,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    augment_train(algo, episodic_returns, run_num, augment, batchsize)

for batchsize in [500, 2000]:
    for i in range(10):
        run_task(i, batchsize, augment=False)
    for i in range(10):
        run_task(i, batchsize, augment=True)






