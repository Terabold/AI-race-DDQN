from scripts.Constants import CHECKPOINT_CENTERS
import numpy as np
import math


def calculate_reward(environment, step_info, prev_state=None):
    """
    Reward function focused on state deltas (changes) rather than absolute values
    """
    reward = 0.0
    breakdown = {}
    
    car = environment.car
    
    # PROGRESS REWARD (already delta-based)
    if environment.checkpoint_manager.current_idx < environment.checkpoint_manager.total_checkpoints:
        distance_delta = environment.prev_checkpoint_distance - environment.current_checkpoint_distance
        progress = distance_delta * 0.5
        
        if distance_delta > 0 and car.velocity > 0:
            speed_mult = 1.0 + (car.velocity / car.max_velocity) * 0.8
            progress *= speed_mult
        elif distance_delta < 0:
            progress *= 1.5
        
        reward += progress
        breakdown["progress"] = progress
    
    # WALL DANGER - DELTA BASED
    if prev_state is not None and len(car.wall_distances) > 0:
        # Extract previous wall distances from prev_state
        prev_wall_distances = prev_state[:15]  # First 15 values are wall rays
        curr_wall_distances = car.wall_distances / car.ray_length
        
        # Calculate minimum distance change
        prev_min_wall = prev_wall_distances.min()
        curr_min_wall = curr_wall_distances.min()
        wall_delta = curr_min_wall - prev_min_wall
        
        # Reward moving away from walls, penalize getting closer
        if wall_delta > 0:  # Moving away from wall
            wall_reward = wall_delta * 2.0
        else:  # Getting closer to wall
            wall_reward = wall_delta * 5.0  # Heavier penalty for approaching
        
        # Extra penalty if dangerously close
        if curr_min_wall < 0.05:
            wall_reward -= 5.0
        
        reward += wall_reward
        breakdown["wall_delta"] = wall_reward
    
    # BOMB DANGER - DELTA BASED (only obstacles, not walls)
    if prev_state is not None and len(car.bomb_distances) > 0:
        # Extract previous bomb distances from prev_state
        prev_bomb_distances = prev_state[15:30]  # Next 15 values are bomb rays
        curr_bomb_distances = car.bomb_distances / car.ray_length
        
        # Only consider rays that hit obstacles (not walls)
        if car.bomb_hit_obstacle.any():
            # Get minimum distance only for rays hitting obstacles
            obstacle_rays = curr_bomb_distances[car.bomb_hit_obstacle]
            if len(obstacle_rays) > 0:
                curr_min_bomb = obstacle_rays.min()
                
                # For previous state, estimate which rays had obstacles
                # Use heuristic: shorter bomb distances than wall distances likely hit obstacles
                prev_obstacle_rays = prev_bomb_distances[prev_bomb_distances < 0.9]
                prev_min_bomb = prev_obstacle_rays.min() if len(prev_obstacle_rays) > 0 else 1.0
                
                bomb_delta = curr_min_bomb - prev_min_bomb
                
                if bomb_delta < 0:  
                    bomb_reward = bomb_delta * 2.0
                else:  
                    bomb_reward = 0
                
                # Extra penalty if very close to obstacle
                if curr_min_bomb < 0.075:
                    bomb_reward -= 4.0
                
                reward += bomb_reward
                breakdown["bomb_delta"] = bomb_reward
    
    # VELOCITY DELTA - Encourage maintaining speed
    if prev_state is not None:
        prev_velocity = prev_state[30]  # Velocity is at index 30
        curr_velocity = max(0.0, car.velocity / car.max_velocity)
        velocity_delta = curr_velocity - prev_velocity
        
        # Reward acceleration, penalize slowing down (unless avoiding obstacle)
        if velocity_delta > 0:
            velocity_reward = velocity_delta * 0.5
        else:
            velocity_reward = velocity_delta * 0.2  # Small penalty for slowing
        
        reward += velocity_reward
        breakdown["velocity_delta"] = velocity_reward
    
    # EVENT-BASED REWARDS (already delta-based by nature)
    
    # Checkpoint crossed
    if step_info.get("checkpoint_crossed", False):
        if environment.checkpoint_manager.current_idx == environment.checkpoint_manager.total_checkpoints:
            cp_reward = 0.0
        else:
            cp_reward = 10.0
        reward += cp_reward
        breakdown["checkpoint"] = cp_reward

    # Backward crossing
    if step_info.get("backward_crossed", False):
        back = -8.0
        reward += back
        breakdown["backward"] = back

    # Obstacle hit - Delta event (state changed from not hit to hit)
    if step_info.get("hit_obstacle", False):
        obs = -15.0
        reward += obs
        breakdown["obstacle"] = obs

    # Collision - Delta event
    if step_info.get("collision", False):
        crash = -20.0  # Increased penalty - this is terminal failure
        reward += crash
        breakdown["collision"] = crash

    # Timeout - Delta event
    if step_info.get("timeout", False):
        timeout = -10.0
        reward += timeout
        breakdown["timeout"] = timeout

    # Finished - Delta event with time bonus
    if step_info.get("finished", False):
        finish = 50.0
        time_bonus = (environment.time_remaining / environment.max_time) * 30.0
        reward += finish + time_bonus
        breakdown["finish"] = finish
        breakdown["time_bonus"] = time_bonus
    
    breakdown["total"] = reward
    
    # Store in environment for visualization
    environment.last_reward = reward
    environment.last_reward_breakdown = breakdown.copy()
    environment.episode_reward += reward
    
    return float(reward), breakdown