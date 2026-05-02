from stable_baselines3 import PPO
from rl_env import AeroManiaEnv
import time

def evaluate():
    # Run in human mode so we can watch
    env = AeroManiaEnv(render_mode="human")
    
    # Load the latest trained model
    try:
        model = PPO.load("models/aeromania_rl_model_final", env=env)
        print("Model loaded successfully!")
    except:
        print("Final model not found, trying latest checkpoint...")
        model = PPO.load("models/checkpoints/aeromania_model_10000_steps", env=env)

    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    evaluate()