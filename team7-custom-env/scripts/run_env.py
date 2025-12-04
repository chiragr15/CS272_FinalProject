import gymnasium as gym
import custom_env

def main():
    env = gym.make("Team7-v0", render_mode="human")
    obs, info = env.reset()
    # obs, info = env.reset()
    # if not env.observation_space.contains(obs):
    #     print("reset() observation out of space:", obs.shape, env.observation_space)

    # obs, reward, done, truncated, info = env.step(env.action_space.sample())
    # if not env.observation_space.contains(obs):
    #     print("step() observation out of space:", obs.shape, env.observation_space)

    total = 0

    for _ in range(500):
        action = env.unwrapped.action_type.actions_indexes["IDLE"]
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs)
        total += reward
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total:.2f}")
            break

    env.close()

if __name__ == "__main__":
    main()
