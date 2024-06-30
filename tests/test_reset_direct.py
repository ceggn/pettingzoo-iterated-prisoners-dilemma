from pettingzoo.utils import parallel_to_aec

from ipd_env import raw_env


def test_reset_after_parallel_to_aec():
    env_instance = raw_env()
    env_instance = parallel_to_aec(env_instance)
    observations, infos = env_instance.reset()
    print("After parallel_to_aec Reset - Observations:", observations)
    print("After parallel_to_aec Reset - Infos:", infos)

if __name__ == "__main__":
    test_reset_after_parallel_to_aec()