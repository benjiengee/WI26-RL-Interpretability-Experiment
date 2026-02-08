from stable_baselines3 import DQN
import gymnasium as gym


def train_oracle():
    env = ... # TODO
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=100_000,    # max transitions stored in replay buffer
        learning_starts=1_000,  # num steps collected before performing gradient updates
        batch_size=64,          # SGD batch size
        gamma=0.99,             # discount factor
        train_freq=4,           # num env steps for each gradient update
        target_update_interval=1_000,   # num steps between target network updates
        exploration_fraction=0.1,       # fraction of total training steps before eps decays
        exploration_final_eps=0.05,     # minimum eps after decay
        verbose=1,                      # 0 -> silent, 1 -> info, 2 -> debug
        policy_kwargs=dict(net_arch=[256,256]) # 2 hidden layers, 256 width
    )

    model.learn(total_timesteps = 200_000)
    model_path = "./log/oracle/icu_env"
    model.save(model_path)
    print(f"Training complete. Saved model to {model_path}")
