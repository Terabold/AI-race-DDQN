# rewards - teaches the ai what's good and bad
# positive = good, negative = bad
# delta based: reward CHANGE not just position


def calculate_reward(environment, step_info, prev_state=None):
    reward = 0.0
    breakdown = {}
    car = environment.car
    
    # 1. checkpoint progress - most important
    if environment.checkpoint_manager.current_idx < environment.checkpoint_manager.total_checkpoints:
        distance_delta = environment.prev_checkpoint_distance - environment.current_checkpoint_distance
        
        if distance_delta > 0:  # moving toward
            speed_mult = 1.0 + (car.velocity / car.max_velocity) * 0.5
            progress = distance_delta * 0.3 * speed_mult
        else:  # moving away
            progress = distance_delta * 0.8
        
        reward += progress
        breakdown["progress"] = progress
    
    # 2. wall safety
    if prev_state is not None and len(car.wall_distances) > 0:
        prev_wall = prev_state[:15].min()
        curr_wall = (car.wall_distances / car.ray_length).min()
        wall_delta = curr_wall - prev_wall
        
        if wall_delta > 0:
            wall_reward = wall_delta * 2.0
        else:
            wall_reward = wall_delta * 5.0
        
        if curr_wall < 0.05:  # too close!
            wall_reward -= 5.0
        
        reward += wall_reward
        breakdown["wall"] = wall_reward
    
    # 3. bomb avoidance
    if prev_state is not None and len(car.bomb_distances) > 0 and car.bomb_hit_obstacle.any():
        curr_obstacle_rays = (car.bomb_distances / car.ray_length)[car.bomb_hit_obstacle]
        
        if len(curr_obstacle_rays) > 0:
            curr_min_obstacle = curr_obstacle_rays.min()
            
            prev_bomb = prev_state[15:30]
            prev_obstacle_rays = prev_bomb[prev_bomb < 0.9]
            prev_min_obstacle = prev_obstacle_rays.min() if len(prev_obstacle_rays) > 0 else 1.0
            
            obstacle_delta = curr_min_obstacle - prev_min_obstacle
            
            if obstacle_delta < 0:
                obstacle_reward = obstacle_delta * 3.0
            else:
                obstacle_reward = 0
            
            if curr_min_obstacle < 0.075:  # very close
                obstacle_reward -= 4.0
            
            reward += obstacle_reward
            breakdown["obstacle"] = obstacle_reward
    
    # 4. speed - reward going fast
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
    
    # 5. big events
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
    return float(reward), breakdown