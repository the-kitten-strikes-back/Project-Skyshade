from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from rl_env import AeroManiaEnv
import os
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

class RichProgressBarCallback(BaseCallback):
    """A custom callback to display a sleek progress bar using the 'rich' library."""
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )
        self.task_id = None

    def _on_training_start(self):
        self.progress.start()
        self.task_id = self.progress.add_task("Training AI Pilot", total=self.total_timesteps)

    def _on_step(self) -> bool:
        self.progress.update(self.task_id, advance=1)
        return True

    def _on_training_end(self):
        self.progress.stop()

def main():
    # 1. Initialize the custom environment
    # Set render_mode="human" if you want to watch the window
    env = AeroManiaEnv(render_mode=None) 

    # 2. Define the PPO agent
    # We set verbose=0 because Rich will handle the output visually
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./ppo_aeromania_logs/")

    # 3. Setup Callbacks
    total_steps = 100000
    progress_callback = RichProgressBarCallback(total_timesteps=total_steps)
    
    # Checkpoint Callback: Saves the model every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="aeromania_model"
    )

    # 4. Train the model
    print(f"Initializing AeroMania RL Training for {total_steps} steps...")
    model.learn(total_timesteps=total_steps, callback=[progress_callback, checkpoint_callback])

    # 5. Save the final trained model
    os.makedirs("models", exist_ok=True)
    model.save("models/aeromania_rl_model_final")
    print("\nTraining complete! Final model saved to 'models/aeromania_rl_model_final.zip'")

if __name__ == "__main__":
    main()