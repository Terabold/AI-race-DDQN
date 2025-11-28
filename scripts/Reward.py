def calculate_reward(environment, step_info, prev_state=None):
    """
    Delta-based reward function - CLEANED UP VERSION
    Keeps your approach but more readable
    """
    reward = 0.0
    breakdown = {}
    car = environment.car
    
    # 1. CHECKPOINT PROGRESS (most important)
    if environment.checkpoint_manager.current_idx < environment.checkpoint_manager.total_checkpoints:
        distance_delta = environment.prev_checkpoint_distance - environment.current_checkpoint_distance
        
        if distance_delta > 0:  # Moving toward checkpoint
            speed_mult = 1.0 + (car.velocity / car.max_velocity) * 0.5
            progress = distance_delta * 0.3 * speed_mult
        else:  # Moving away
            progress = distance_delta * 0.8
        
        reward += progress
        breakdown["progress"] = progress
    
    # 2. WALL SAFETY - Delta-based (YOUR APPROACH - KEEP IT!)
    if prev_state is not None and len(car.wall_distances) > 0:
        prev_wall = prev_state[:15].min()
        curr_wall = (car.wall_distances / car.ray_length).min()
        wall_delta = curr_wall - prev_wall
        
        # Reward moving away, penalize approaching
        if wall_delta > 0:
            wall_reward = wall_delta * 2.0
        else:
            wall_reward = wall_delta * 5.0
        
        # Extra penalty if dangerously close
        if curr_wall < 0.05:
            wall_reward -= 5.0
        
        reward += wall_reward
        breakdown["wall"] = wall_reward
    
    # 3. OBSTACLE AVOIDANCE - Delta-based for obstacles only
    if prev_state is not None and len(car.bomb_distances) > 0 and car.bomb_hit_obstacle.any():
        # Current obstacle distances
        curr_obstacle_rays = (car.bomb_distances / car.ray_length)[car.bomb_hit_obstacle]
        
        if len(curr_obstacle_rays) > 0:
            curr_min_obstacle = curr_obstacle_rays.min()
            
            # Previous obstacle distances (estimate)
            prev_bomb = prev_state[15:30]
            prev_obstacle_rays = prev_bomb[prev_bomb < 0.9]
            prev_min_obstacle = prev_obstacle_rays.min() if len(prev_obstacle_rays) > 0 else 1.0
            
            obstacle_delta = curr_min_obstacle - prev_min_obstacle
            
            # Penalize approaching obstacles
            if obstacle_delta < 0:
                obstacle_reward = obstacle_delta * 3.0
            else:
                obstacle_reward = 0
            
            # Extra penalty if very close
            if curr_min_obstacle < 0.075:
                obstacle_reward -= 4.0
            
            reward += obstacle_reward
            breakdown["obstacle"] = obstacle_reward
    
    # 4. SPEED MAINTENANCE - Delta-based
    if prev_state is not None:
        prev_velocity = prev_state[30]
        curr_velocity = max(0.0, car.velocity / car.max_velocity)
        velocity_delta = curr_velocity - prev_velocity
        
        if velocity_delta > 0:
            velocity_reward = velocity_delta * 0.5
        else:
            velocity_reward = velocity_delta * 0.2
        
        reward += velocity_reward
        breakdown["velocity"] = velocity_reward
    
    # 5. EVENTS (big rewards/penalties)
    if step_info.get("checkpoint_crossed", False):
        cp_reward = 15.0
        reward += cp_reward
        breakdown["checkpoint"] = cp_reward

    if step_info.get("backward_crossed", False):
        back = -10.0
        reward += back
        breakdown["backward"] = back

    if step_info.get("hit_obstacle", False):
        obs = -8.0
        reward += obs
        breakdown["hit_obstacle"] = obs

    if step_info.get("collision", False):
        crash = -25.0
        reward += crash
        breakdown["collision"] = crash

    if step_info.get("timeout", False):
        timeout = -15.0
        reward += timeout
        breakdown["timeout"] = timeout

    if step_info.get("finished", False):
        finish = 100.0
        time_bonus = (environment.time_remaining / environment.max_time) * 50.0
        reward += finish + time_bonus
        breakdown["finish"] = finish
        breakdown["time_bonus"] = time_bonus
    
    breakdown["total"] = reward
    
    environment.last_reward = reward
    environment.last_reward_breakdown = breakdown.copy()
    environment.episode_reward += reward
    
    return float(reward), breakdown