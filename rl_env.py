import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ursina import *
import entities
import physics

class AeroManiaEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(AeroManiaEnv, self).__init__()
        self.render_mode = render_mode
        
        # Toggle headless based on render_mode
        is_headless = render_mode != "human"
        self.app = Ursina(headless=is_headless, window_type='none' if is_headless else 'onscreen')
        
        # Actions: Pitch, Yaw, Throttle, Fire
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        # Observations: rel_pos (3), rel_vel (3), own_speed (1), target_in_sights (1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        # Setup training entities: AI is the Agent, Player is the Opponent
        self.ai = entities.RLPlane(position=(0, 2000, 0))
        self.player = entities.EnemyPlane(position=(0, 2000, 500))

        # Inject dependencies into the entities module scope so EnemyPlane logic works
        entities.plane = self.ai
        entities.missiles = []
        entities.flares = []
        entities.enemy_planes = [self.player]
        entities.missile_velocity = physics.missile_velocity
        entities.speed = 200
        
        # Mock UI elements to prevent crashes in headless mode
        entities.explosion_3d = lambda pos: None
        entities.explosion = type('Mock', (), {'play': lambda: None})
        entities.enemy_destroyed_display = type('Mock', (), {'text': ''})
        entities.points_display = type('Mock', (), {'text': 'Points: 0'})
        
        if not is_headless:
            self.ai.color = color.blue
            self.player.color = color.red
            Sky()
            DirectionalLight()
            self.e_cam = EditorCamera()
            self.e_cam.position = self.ai.position + Vec3(0, 10, -30)
            self.hud = {
                'altitude': Text(text='', position=(-0.85, 0.45), scale=1.5),
                'speed': Text(text='', position=(-0.85, 0.40), scale=1.5),
                'health': Text(text='', position=(-0.85, 0.35), scale=1.5, color=color.green),
                'range': Text(text='', position=(0.5, 0.45), scale=1.5, color=color.red),
                'missiles': Text(text='', position=(0.5, 0.40), scale=1.5),
                'ammo': Text(text='', position=(0.5, 0.35), scale=1.5),
                'alignment': Text(text='', position=(0, 0.45), origin=(0,0), scale=1.2)
            }
        
        # Simulation step delta
        self.sim_dt = 1/20.0 

    def _get_obs(self):
        rel_pos = self.player.position - self.ai.position
        return np.array([
            rel_pos.x, rel_pos.y, rel_pos.z,
            self.ai.rotation_x, self.ai.rotation_y,
            self.ai.speed,
            distance(self.ai.position, self.player.position),
            float(self._check_alignment())
        ], dtype=np.float32)

    def _check_alignment(self):
        # Dot product to see if player is in front of AI
        to_player = (self.player.position - self.ai.position).normalized()
        forward = self.ai.forward
        return forward.dot(to_player)

    def step(self, action):
        # 1. Apply actions to the RLPlane
        prev_enemy_health = self.player.health
        time.dt = self.sim_dt
        self.ai.apply_actions(pitch=action[0], yaw=action[1], throttle_change=0, fire=action[3])
        
        # Update Ursina engine state (positions, internal logic)
        self.app.step()
        
        # 2. Calculate Reward
        alignment = self._check_alignment()
        dist = distance(self.ai.position, self.player.position)

        if self.render_mode == "human":
            self.hud['altitude'].text = f"ALT: {self.ai.y:.0f}m"
            self.hud['speed'].text = f"SPD: {self.ai.speed:.0f}"
            self.hud['health'].text = f"HP: {self.ai.health:.0f}"
            self.hud['range'].text = f"RNG: {dist:.0f}m"
            self.hud['missiles'].text = f"MSL: {self.ai.missile_count}"
            self.hud['ammo'].text = f"GUN: {self.ai.gun_ammo}"
            self.hud['alignment'].text = f"ALIGN: {alignment:.2f}"
            self.hud['alignment'].color = color.green if alignment > 0.8 else color.white
        
        reward = 0
        
        # Reward for dealing damage (Guns or Missiles)
        damage_dealt = prev_enemy_health - self.player.health
        if damage_dealt > 0:
            reward += damage_dealt * 2.0
            
        reward += alignment * 0.1  # Reward for facing the enemy
        if dist < 1000: reward += 0.05 # Reward for closing in
        if dist > 5000: reward -= 0.1  # Penalty for losing the target
        
        # 3. Check if done
        terminated = bool(self.ai.health <= 0 or self.player.health <= 0 or dist < 100 or dist > 7000)
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset positions for new episode
        self.ai.position = Vec3(0, 2000, 0)
        self.ai.rotation = Vec3(0,0,0)
        self.ai.health = 100

        # Reset EnemyPlane state
        self.player.position = Vec3(0, 2000, 500)
        self.player.rotation = Vec3(0,0,0)
        self.player.health = 50
        self.player.state = 'patrol'
        
        if self.player not in entities.enemy_planes:
            entities.enemy_planes.append(self.player)
        entities.missiles.clear()
        entities.flares.clear()

        self.app.step()
        return self._get_obs(), {}