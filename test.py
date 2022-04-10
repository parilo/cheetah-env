import numpy as np

from cheetah_env.cheetah_env import CheetahEnv


def main():

    env = CheetahEnv()

    for i in range(2000):
        if i % 1000 == 0:
            env.reset()
        action = np.random.normal(0, 0.001, size=12)
        obs, _, _, _ = env.step(action)
        obs_shapes = {key: val.shape for key, val in obs.items()}
        np.set_printoptions(suppress=True)
        print(f'obs {obs_shapes}')
        print(f'obs {obs}')
        env.render()


if __name__ == "__main__":
    main()
