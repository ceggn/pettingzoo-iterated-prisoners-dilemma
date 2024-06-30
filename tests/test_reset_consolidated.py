from pettingzoo.utils import parallel_to_aec, wrappers

from ipd_env import raw_env


def test_reset_direct():
    env_instance = raw_env()
    observations, infos = env_instance.reset()
    print("Direct Env Reset - Observations:", observations)
    print("Direct Env Reset - Infos:", infos)

def test_reset_after_parallel_to_aec():
    env_instance = raw_env()
    env_instance = parallel_to_aec(env_instance)
    observations, infos = env_instance.reset()
    print("After parallel_to_aec Reset - Observations:", observations)
    print("After parallel_to_aec Reset - Infos:", infos)

def test_reset_after_assert_out_of_bounds():
    env_instance = raw_env()
    env_instance = parallel_to_aec(env_instance)
    env_instance = wrappers.AssertOutOfBoundsWrapper(env_instance)
    observations, infos = env_instance.reset()
    print("After AssertOutOfBoundsWrapper Reset - Observations:", observations)
    print("After AssertOutOfBoundsWrapper Reset - Infos:", infos)

def test_reset_after_order_enforcing():
    env_instance = raw_env()
    env_instance = parallel_to_aec(env_instance)
    env_instance = wrappers.AssertOutOfBoundsWrapper(env_instance)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    observations, infos = env_instance.reset()
    print("After OrderEnforcingWrapper Reset - Observations:", observations)
    print("After OrderEnforcingWrapper Reset - Infos:", infos)

if __name__ == "__main__":
    print("Testing direct reset:")
    test_reset_direct()
    print("\nTesting reset after parallel_to_aec:")
    test_reset_after_parallel_to_aec()
    print("\nTesting reset after AssertOutOfBoundsWrapper:")
    test_reset_after_assert_out_of_bounds()
    print("\nTesting reset after OrderEnforcingWrapper:")
    test_reset_after_order_enforcing()