import matplotlib.pyplot as plt
def plot_rewards(rewards_history, save_path="./vlm_outputs/reward_plot.png"):
    """Plot the reward components throughout the episode."""
    if not rewards_history:
        print("No reward data to plot.")
        return
    
    # Extract timesteps and reward components
    timesteps = list(range(len(rewards_history)))
    safety_rewards = [r.get('safety_reward', 0) for r in rewards_history]
    progress_rewards = [r.get('progress_reward', 0) for r in rewards_history]
    smoothness_rewards = [r.get('smoothness_reward', 0) for r in rewards_history]
    collision_penalties = [r.get('collision_penalty', 0) for r in rewards_history]
    total_rewards = [r.get('total_reward', 0) for r in rewards_history]
    
    # Create subplots
    fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    
    # Plot each reward component
    axs[0].plot(timesteps, safety_rewards, 'r-', label='Safety Reward')
    axs[0].set_title('Safety Reward (Pedestrian Proximity)')
    axs[0].set_ylabel('Reward Value')
    axs[0].grid(True)
    
    axs[1].plot(timesteps, progress_rewards, 'g-', label='Progress Reward')
    axs[1].set_title('Progress Reward (Speed when Safe)')
    axs[1].set_ylabel('Reward Value')
    axs[1].grid(True)
    
    axs[2].plot(timesteps, smoothness_rewards, 'b-', label='Smoothness Reward')
    axs[2].set_title('Smoothness Reward (Jerky Driving Penalty)')
    axs[2].set_ylabel('Reward Value')
    axs[2].grid(True)
    
    axs[3].plot(timesteps, collision_penalties, 'm-', label='Collision Penalty')
    axs[3].set_title('Collision Penalty')
    axs[3].set_ylabel('Reward Value')
    axs[3].grid(True)
    
    axs[4].plot(timesteps, total_rewards, 'k-', label='Total Reward')
    axs[4].set_title('Total Reward')
    axs[4].set_xlabel('Timestep')
    axs[4].set_ylabel('Reward Value')
    axs[4].grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Reward plot saved to {save_path}")
    plt.close(fig)

