def calculate_reward(environment, step_info, prev_state=None):
    reward = 0.0
    breakdown = {}
    car = environment.car
    
    if environment.checkpoint_manager.current_idx < environment.checkpoint_manager.total_checkpoints:  #אם יש עוד צ'קפוינטים שלא עברנו
        distance_delta = environment.prev_checkpoint_distance - environment.current_checkpoint_distance # חישוב השינוי במרחק לצ'קפוינט - חיובי אומר שהתקרבנו
        
        if distance_delta > 0:
            # מכפיל מהירות: אם הרכב בשיא המהירות, המכפיל יהיה 1.5
            # 1.0 + 0.5 = 1.5
            speed_mult = 1.0 + (car.velocity / car.max_velocity) * 0.5
            # תגמול התקדמות
            # 10px * 0.3 * 1.5 = 4.5
            progress = distance_delta * 0.3 * speed_mult
        else:
            progress = distance_delta * 0.8
        
        reward += progress
        breakdown["progress"] = progress
    
    if prev_state is not None and len(car.wall_distances) > 0: 
        # מציאת המרחק הכי קרוב לקיר מתוך 15 הקרניים הראשונות [d1...d15]
        prev_wall = prev_state[:15].min()
        curr_wall = (car.wall_distances / car.ray_length).min()
        # חישוב ההפרש: חיובי אומר שהתרחקנו מהקיר (טוב), שלילי אומר שהתקרבנו (רע)
        wall_delta = curr_wall - prev_wall
        
        if wall_delta > 0:
            wall_reward = wall_delta * 2.0
        else:
            wall_reward = wall_delta * 5.0  # קנס חזק יותר על קרבה לקיר
        
        if curr_wall < 0.05:  # פחות מ-20 פיקסלים מהקיר
            wall_reward -= 5.0
        
        reward += wall_reward
        breakdown["wall"] = wall_reward
    
    # אינדקסים 15-29 במערך המצב = קרני פצצה
    if prev_state is not None and len(car.bomb_distances) > 0 and car.bomb_hit_obstacle.any(): # יש מצב קודם, קיימים נתוני מרחקים לפצצות, ויש לפחות פגיעה אחת במכשול
        curr_obstacle_rays = (car.bomb_distances / car.ray_length)[car.bomb_hit_obstacle] # נרמול מרחקים ושמירה רק של הקרניים שאכן זיהו פצצה        
        if len(curr_obstacle_rays) > 0:
            curr_min_obstacle = curr_obstacle_rays.min() # הקרן הכי קרובה לפצצה
            
            prev_bomb = prev_state[15:30]
            prev_obstacle_rays = prev_bomb[prev_bomb < 0.9] # סינון הקרניים שזיהו פצצה במצב הקודם
            prev_min_obstacle = prev_obstacle_rays.min() if len(prev_obstacle_rays) > 0 else 1.0
            
            obstacle_delta = curr_min_obstacle - prev_min_obstacle
            
            if obstacle_delta < 0:  # מתקרב לפצצה
                obstacle_reward = obstacle_delta * 3.0
            else:
                obstacle_reward = 0  # אין בונוס על התרחקות
            
            if curr_min_obstacle < 0.075:  # פחות מ-30 פיקסלים מפצצה
                obstacle_reward -= 4.0
            
            reward += obstacle_reward
            breakdown["obstacle"] = obstacle_reward
    
    # אינדקס 30 במערך המצב זה מהירות נוכחית
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
        time_bonus = (environment.time_remaining / environment.max_time) * 50.0  # ככל שסיים מהר יותר, בונוס גדול יותר
        reward += finish + time_bonus
        breakdown["finish"] = finish
        breakdown["time_bonus"] = time_bonus
    
    breakdown["total"] = reward
    return float(reward), breakdown