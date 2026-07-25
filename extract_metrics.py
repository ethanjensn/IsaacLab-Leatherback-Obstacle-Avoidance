from tensorboard.backend.event_processing import event_accumulator
import os
import glob

# Path to the TensorBoard event file
event_file = r"C:\Users\Jense\Documents\CODING-C\IsaacLab-Leatherback-Obstacle-Avoidance\runs\rsl_rl\leatherback_direct\2025-10-28_18-46-10\events.out.tfevents.1761677571.129-158-206-217.298761.0"

# Load the event file
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

print("="*80)
print("TENSORBOARD METRICS EXTRACTOR")
print("="*80)

# Print all available tags
print("\nAvailable scalar tags:")
for tag in ea.Tags()['scalars']:
    print(f"  - {tag}")

print("\n" + "="*80)
print("KEY METRICS SUMMARY")
print("="*80)

# Extract key metrics
key_metrics = [
    'Train/mean_reward',
    'Train/mean_episode_length',
    'Loss/learning_rate',
    'Loss/value_function',
    'Loss/surrogate',
]

for metric in key_metrics:
    if metric in ea.Tags()['scalars']:
        events = ea.Scalars(metric)
        if events:
            first_val = events[0].value
            last_val = events[-1].value
            max_val = max(e.value for e in events)
            min_val = min(e.value for e in events)
            steps = events[-1].step
            
            print(f"\n{metric}:")
            print(f"  Steps: {steps}")
            print(f"  First: {first_val:.4f}")
            print(f"  Last: {last_val:.4f}")
            print(f"  Max: {max_val:.4f}")
            print(f"  Min: {min_val:.4f}")
    else:
        print(f"\n{metric}: NOT FOUND")

print("\n" + "="*80)
print("FINDING BEST CHECKPOINT BY MEAN REWARD")
print("="*80)

# Find the checkpoint with highest mean_reward
if 'Train/mean_reward' in ea.Tags()['scalars']:
    reward_events = ea.Scalars('Train/mean_reward')
    
    # Find event with maximum reward
    best_event = max(reward_events, key=lambda e: e.value)
    
    print(f"\nBest mean_reward: {best_event.value:.4f}")
    print(f"Best step: {best_event.step}")
    
    # Find closest checkpoint file
    checkpoint_dir = r"C:\Users\Jense\Documents\CODING-C\IsaacLab-Leatherback-Obstacle-Avoidance\runs\rsl_rl\leatherback_direct\2025-10-28_18-46-10"
    
    # Get all model files
    model_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    
    # Parse step numbers from filenames
    checkpoint_steps = []
    for f in model_files:
        basename = os.path.basename(f)
        # Extract step number from model_X.pt
        step_str = basename.replace("model_", "").replace(".pt", "")
        try:
            step = int(step_str)
            checkpoint_steps.append((step, f))
        except ValueError:
            continue
    
    # Sort by step
    checkpoint_steps.sort()
    
    # Find closest checkpoint to best step
    if checkpoint_steps:
        best_step = best_event.step
        closest_checkpoint = min(checkpoint_steps, key=lambda x: abs(x[0] - best_step))
        
        print(f"\nRecommended checkpoint:")
        print(f"  File: {closest_checkpoint[1]}")
        print(f"  Step: {closest_checkpoint[0]}")
        print(f"  Reward: {best_event.value:.4f}")
        
        print(f"\nPlay command:")
        print(f'python scripts/reinforcement_learning/rsl_rl/play.py --task Template-Leatherback-Direct-v0 --checkpoint "{closest_checkpoint[1]}" --num_envs 32')
    else:
        print("No checkpoint files found")
else:
    print("train/mean_reward not found in logs")

print("\n" + "="*80)
print("ALL SCALAR TAGS WITH FINAL VALUES")
print("="*80)

for tag in ea.Tags()['scalars']:
    events = ea.Scalars(tag)
    if events:
        print(f"{tag}: {events[-1].value:.4f} (step {events[-1].step})")
