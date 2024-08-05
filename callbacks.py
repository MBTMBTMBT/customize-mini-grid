import os
import numpy as np
from typing import Optional, Union, List
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym


class EvalSaveCallback(EventCallback):
    """
    Callback for evaluating an agent and saving the best and latest models.

    :param eval_env: The environment used for evaluation
    :param log_dir: Directory where logs and models are saved
    :param eval_freq: Frequency (in timesteps) at which the model is evaluated
    :param n_eval_episodes: Number of episodes to evaluate the model
    :param deterministic: Whether to use deterministic actions during evaluation
    :param verbose: Verbosity level (0: no output, 1: info messages, 2: debug messages)
    """

    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            log_dir: str,
            eval_freq: int = 10000,
            n_eval_episodes: int = 5,
            deterministic: bool = True,
            verbose: int = 1
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.log_dir = log_dir
        self.checkpoint_dir = os.path.join(log_dir, "checkpoints")
        self.best_model_path = os.path.join(log_dir, "best_model.zip")
        self.latest_model_path = os.path.join(log_dir, "latest_model.zip")
        self.evaluations: List[float] = []

        # Ensure directories exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Evaluate the model at specified frequency
        if self.n_calls % self.eval_freq == 0:
            mean_reward, _ = self.evaluate_and_log()

            # Save the latest model checkpoint
            self.model.save(self.latest_model_path)
            if self.verbose >= 1:
                print(f"Saved latest model to {self.latest_model_path}")

            # If current mean reward is the best we have seen, save the model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.best_model_path)
                if self.verbose >= 1:
                    print(f"New best model with mean reward {mean_reward:.2f} saved to {self.best_model_path}")

        return True

    def evaluate_and_log(self):
        """Evaluate the model and log results."""
        episode_rewards, _ = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            return_episode_rewards=True
        )

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        self.evaluations.append(mean_reward)

        # Log results
        if self.verbose >= 1:
            print(f"Evaluation: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Save evaluation results to a file for further analysis
        np.save(os.path.join(self.log_dir, 'evaluations.npy'), self.evaluations)

        return mean_reward, std_reward
