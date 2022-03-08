import numpy as np
from statistics import mean
import torch
from pathlib import Path
import matplotlib.pyplot as plt

from diffql.envs.fixed_wing import CCVFighterDTLinearEnv
from diffql.envs.basic import MechanicalDTEnv
from diffql.networks import LSE
from diffql.agents.random import ActionSampler
from diffql.agents.lqr import DLQR
from diffql.agents.pcq_agent import ParametrisedConvexQAgent
from diffql.sim import parsim, sim
from diffql.learning import DataBuffer, PCQL

import time


env_name = "CCVFighter"
if env_name == "CCVFighter":
    Env = CCVFighterDTLinearEnv
    bounds_dict = {
        "x_min": np.array([-np.deg2rad(10), -np.deg2rad(30), -np.deg2rad(10)]),
        "x_max": np.array([np.deg2rad(10), np.deg2rad(30), np.deg2rad(10)]),
        "u_min": np.array([-np.deg2rad(25), -np.deg2rad(20)]),
        "u_max": np.array([np.deg2rad(25), np.deg2rad(20)]),
    }
elif env_name == "mechanical":
    Env = MechanicalDTEnv
    bounds_dict = {
        "x_min": -10*np.ones(4),
        "x_max": 10*np.ones(4),
        "u_min": -1*np.ones(1),
        "u_max": 1*np.ones(1),
    }


def sample_envs(
    N: int,
    seed=None,
    is_env_bounded: bool = True,
):
    x_min, x_max = bounds_dict["x_min"], bounds_dict["x_max"]
    u_min, u_max = bounds_dict["u_min"], bounds_dict["u_max"]
    # conditions (e.g., init cond from dummy env `_env`)
    _env = Env(
        x_min=x_min, x_max=x_max,
        u_min=u_min, u_max=u_max,
    )  # dummy env
    Q = np.diag(np.ones(Env.observation_space_shape[0]))
    R = np.diag(np.ones(Env.action_space_shape[0]))
    # initial conditions
    _env.observation_space.seed(seed)  # for consistent random initial conditions
    x0s = [_env.observation_space.sample() for _ in range(N)]
    if is_env_bounded:
        _x_min, _x_max = x_min, x_max
    else:
        _x_min, _x_max = None, None
    envs = [
        Env(
            initial_state=x0s[i],
            x_min=_x_min, x_max=_x_max,
            u_min=u_min, u_max=u_max,
            Q=Q, R=R,
        ) for i in range(N)
    ]
    return envs


def sample_random_policy(N: int, seed=None):
    _envs = sample_envs(N, seed=seed)  # dummy envs for random policy
    np.random.seed(seed)
    seeds_agent = [_seed.item() for _seed in np.random.randint(1234567890, size=N)]
    agents = [
        ActionSampler(_envs[i].action_space, seed=seeds_agent[i]) for i in range(N)
    ]
    return agents


def run_sim(
    envs: list, agents: list,
    plot_fig=False, save_data=False, sim_type="sequential", dir_log=Path("data"),
    seed=None,
):
    # random seeds
    N = len(envs)
    np.random.seed(seed)
    seeds_env = [_seed.item() for _seed in np.random.randint(1234567890, size=N)]  # not numpy.int64, but native `int`
    # sim
    t0 = time.time()
    # Sequential sim
    if sim_type == "sequential":
        data_buffers = [sim(
            i+1, envs[i], agents[i], seed=seeds_env[i],
            dir_log=dir_log, plot_fig=plot_fig, save_data=save_data,
        ) for i in range(N)]
    elif sim_type == "parallel":
        data_buffers = parsim(
            N, envs, agents, dir_log=dir_log, plot_fig=plot_fig, save_data=save_data,
        )
    else:
        raise ValueError("Invalid sim_type")
    # Parallel sim. too slow; See #73
    t1 = time.time()
    print(f"Elapsed time: {t1-t0}")
    return data_buffers


def sample_training_data(N: int, seed=None):
    envs = sample_envs(N, seed=seed)
    agents = sample_random_policy(N, seed=seed)
    # data sampling
    data_buffers = run_sim(
        envs, agents,
        plot_fig=False, save_data=False,
        sim_type="sequential", dir_log=Path("data").joinpath("train"),
        seed=seed,
    )
    data_buffer_merged = DataBuffer()
    for data_buffer in data_buffers:
        data_buffer_merged += data_buffer
    return data_buffer_merged


def evaluate(data_buffer: DataBuffer):
    total_reward = torch.sum(data_buffer.r).item()
    return total_reward


def evaluate_agent(envs: list, agent, seed=None, save_name: str = None, plot_fig=True,):
    N = len(envs)
    agents = [agent for _ in range(N)]
    dir_log = Path("data").joinpath("evaluate")
    if save_name is not None:
        dir_log = dir_log.joinpath(save_name)
    data_buffers = run_sim(
        envs, agents,
        plot_fig=plot_fig, save_data=False, sim_type="parallel", dir_log=dir_log,
        seed=seed,
    )
    evaluations = [evaluate(data_buffer) for data_buffer in data_buffers]
    return evaluations


def train(agent, data_buffer, key, envs_eval: list, seed_eval=None):
    model = agent.network
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-1)
    grad_max_norm = 1e-1
    scheduler = None
    trainer = PCQL(
        epochs=200,
        weight_origin=1e3,
        weight_nonneg=1e3,
        grad_max_norm=grad_max_norm,
    )

    evaluation_hist = []

    def callback_eval(epoch: int):
        evaluations = evaluate_agent(envs_eval, agent, seed=seed_eval,
                                     save_name=Path(f"iter_{(epoch):03d}").joinpath(key),
                                     plot_fig=True,)
        mean_evaluation = mean(evaluations)
        print(f"mean evaluation: {mean_evaluation}")
        evaluation_hist.append(mean_evaluation)
        # plot
        if epoch % 5 == 0:
            _, ax = plt.subplots(1)
            ax.plot(evaluation_hist, label="evaluation history during training")
            ax.legend()
            plt.savefig(Path("data").joinpath("evaluation_history.png"))
            plt.clf()
            plt.close()

    _, _, _, _ = trainer.train(model, data_buffer, loss_fn, optimiser, scheduler,
                               callback_eval=callback_eval)

    return agent


if __name__ == "__main__":
    # seed and initialisation
    seed = 2022
    seed_eval = 2023
    torch.manual_seed(seed)
    N = 500
    N_eval = 24
    i_max, T = 20, 1e-0
    n, m = Env.observation_space_shape[0], Env.action_space_shape[0]
    lse = LSE(
        n, m, i_max, T,
        u_min=bounds_dict["u_min"],
        u_max=bounds_dict["u_max"],
    )
    agent_lse = ParametrisedConvexQAgent(lse)
    # data sampling
    training_data_buffer = sample_training_data(N, seed=seed)
    print(f"Total size of the data buffer: {len(training_data_buffer)}")
    envs_eval = sample_envs(N_eval, is_env_bounded=False, seed=seed_eval)
    # training
    agent_lse = train(agent_lse, training_data_buffer, envs_eval=envs_eval, key="lse")
    # evaluation
    agent_random = sample_random_policy(1, seed=seed_eval)[0]
    agent_lqr = DLQR(
        envs_eval[0].A, envs_eval[0].B,
        envs_eval[0].Q, envs_eval[0].R,
    )
    agents_eval_dict = {
        "random": agent_random,
        "lqr": agent_lqr,
        "lse": agent_lse,
    }
    for key in agents_eval_dict:
        agent_eval = agents_eval_dict[key]
        # TODO: early-stopped scenario looks like a good scenario (with high total reward); resolve this
        total_rewards = evaluate_agent(envs_eval, agent_eval, seed=seed_eval, save_name=key)
        print(f"Mean total reward for {key}: {mean(total_rewards)}")
