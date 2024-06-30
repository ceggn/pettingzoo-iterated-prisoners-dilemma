from pettingzoo.utils import parallel_to_aec, wrappers

from ipd_env import raw_env


def test_reset_after_assert_out_of_bounds():
    env_instance = raw_env()
    env_instance = parallel_to_aec(env_instance)
    env_instance = wrappers.AssertOutOfBoundsWrapper(env_instance)
    observations, infos = env_instance.reset()
    print("After AssertOutOfBoundsWrapper Reset - Observations:", observations)
    print("After AssertOutOfBoundsWrapper Reset - Infos:", infos)

if __name__ == "__main__":
    test_reset_after_assert_out_of_bounds()