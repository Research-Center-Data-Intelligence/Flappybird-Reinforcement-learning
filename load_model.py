import torch
from stable_baselines3 import PPO
from env import FlappyBird

# Define models folder
models_dir = "models/PPO3"

def play_game(model_path):
    # Create the Flappy Bird environment
    env = FlappyBird()
    obs, _ = env.reset()

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO.load(model_path, env, device=device, learning_rate=0, ent_coef=0)

    # Run the game with the trained model
    while True:
        action, _ = model.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        env.render()
        if terminated:
            obs, info = env.reset()


if __name__ == "__main__":
    # Ask the user to select the best performing process and model to play the game
    best_process = (input("Please select best performing process: "))
    best_model = (input("please select best performing model: "))

    # Get model path
    model_path = f"{models_dir}/process_{best_process}_model_{best_model}"

    # Play the game with the selected model
    play_game(model_path)
