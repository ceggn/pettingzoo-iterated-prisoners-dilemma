from ipd_env import IteratedPrisonersDilemma


def main():
    env = IteratedPrisonersDilemma(max_steps=10)
    obs = env.reset()
    
    for _ in range(10):
        actions = {
            "player_0": env.action_spaces["player_0"].sample(),
            "player_1": env.action_spaces["player_1"].sample()
        }
        obs, rewards, dones, infos = env.step(actions)
        env.render()
        if all(dones.values()):
            break

    env.close()

if __name__ == "__main__":
    main()
