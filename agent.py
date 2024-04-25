import os
import torch
from stable_baselines3 import PPO
from env import FlappyBird
from multiprocessing import Process

# Set up directories
models_dir = f"models/PPO3"
log_dir = f"logs/PPO3"

# Use cuda as device for faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(process_number, base_process, base_model):
    """"
    Function to train the model.

    Args:
    process_number: int: The number of the process. Used to create a unique save name.
    base_process: int: The best performing process. The training will start from this process.
    base_model: int: The best performing model. The training will start from this model.
    """

    # Create directories if they don't exist
    # Directory models_dir
    # Directory log_dir
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create Snake environment
    # env is the environment class inside the env.py
    # env.reset() initializes environment
    env = FlappyBird()
    env.reset()

    # Define how many steps the model trains for
    # after completing this many steps
    # the model will save
    total_steps = 20000

    # Define how many models already exist for the process
    # This will make the program keep going with newest model
    number_of_models = len([f for f in os.listdir(models_dir) if f.startswith(f"process_{process_number}_model")])

    # Create log dir for each process
    log_dir_process = os.path.join(log_dir, f"process_{process_number}")

    # Training loop just leave it be
    # Makes sure the training goes on continuously
    while True:
        # Create new model if no models exist
        if number_of_models == 0:
            model = PPO("MlpPolicy",
                        env, verbose=1,
                        tensorboard_log=log_dir_process,
                        device=device,
                        )

        # Loads in model that is saved to models_dir
        else:
            model = PPO.load(f"{models_dir}/process_{base_process}_model_{base_model}",
                             env,
                             verbose=1,
                             device=device,
                             tensorboard_log=log_dir_process,

                             )

            print(f"Loading model {models_dir}/process_{base_process}_model_{base_model}")
        
        # Increment the number of models
        number_of_models += 1

        # Train the model
        model.learn(total_timesteps=total_steps, reset_num_timesteps=False, tb_log_name="PPO3")

        # Save the model at intervals
        # Intervals are total steps
        model.save(f"{models_dir}/process_{process_number}_model_{number_of_models}")

        # Makes sure the best model and process
        # Are updated to the most recent
        base_process = process_number
        base_model = number_of_models

        # Close the environment
        env.close()


if __name__ == "__main__":
    # Define which process and model are the best performing
    # See tensorboard graph to see which performs best
    # This will be the starting point for the training
    base_process = (input("Please select best performing process: "))
    base_model = (input("please select best performing model: "))

    # Number of processes you want to run
    # Be careful, higher numbers means higher GPU load
    number_of_processes = int(input("Please select number of processes you want to run simultaneously. More processes require more processing power: "))

    # Create and start processes
    processes = []
    for i in range(number_of_processes):
        process = Process(target=train_model, args=(i, base_process, base_model))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()