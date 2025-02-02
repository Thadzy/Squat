import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import math

@dataclass
class SimConfig:
    """Configuration constants for the simulation"""
    WIDTH: int = 1200
    HEIGHT: int = 800
    FPS: int = 60
    PADDLE_LENGTH: int = 150
    PADDLE_THICKNESS: int = 10
    BALL_RADIUS: int = 10
    BASE_HIT_FORCE: int = 2000
    ANGLE_INCREMENT: float = np.pi / 36  # 5 degree steps
    GRAVITY: Tuple[float, float] = (0, 900)
    BALL_START_POS: Tuple[int, int] = (600, 100)
    PADDLE_POS: Tuple[int, int] = (600, 600)
    OPTIMAL_HIT_HEIGHT: int = 500
    HIT_HEIGHT_TOLERANCE: int = 20
    ANGLE_ADJUSTMENT_RATE: float = 0.1

class SquashHitSimulation:
    def __init__(self, config: SimConfig = SimConfig()):
        self.config = config
        self.setup_pygame()
        self.setup_pymunk()
        self.create_objects()
        self.trajectory_points: List[Tuple[float, float]] = []
        self.hit_points: List[Tuple[float, float]] = []
        self.ball_dropped = False
        self.ball_hit = False
        self.hit_successful = False
        self.hit_data = []
        self.current_force = self.config.BASE_HIT_FORCE
        self.auto_hit = False
        self.auto_adjust = False
        
    def setup_pygame(self) -> None:
        """Initialize Pygame components"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.config.WIDTH, self.config.HEIGHT))
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        pygame.display.set_caption("Squash Ball Hit Analysis")
        
    def setup_pymunk(self) -> None:
        """Initialize Pymunk space and physics"""
        self.space = pymunk.Space()
        self.space.gravity = self.config.GRAVITY
        
    def create_objects(self) -> None:
        """Create all simulation objects"""
        self.create_walls()
        self.create_paddle()
        
    def create_paddle(self) -> None:
        """Create the hitting paddle"""
        self.paddle_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.paddle_body.position = self.config.PADDLE_POS
        self.paddle_shape = pymunk.Poly.create_box(
            self.paddle_body,
            (self.config.PADDLE_LENGTH, self.config.PADDLE_THICKNESS)
        )
        self.paddle_shape.color = (200, 200, 100, 255)
        self.paddle_shape.friction = 0.7
        self.paddle_shape.elasticity = 0.9
        self.space.add(self.paddle_body)
        self.space.add(self.paddle_shape)
        
    def create_ball(self) -> None:
        """Create the squash ball"""
        self.ball_body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, self.config.BALL_RADIUS))
        self.ball_body.position = self.config.BALL_START_POS
        self.ball_shape = pymunk.Circle(self.ball_body, self.config.BALL_RADIUS)
        self.ball_shape.elasticity = 0.85
        self.ball_shape.friction = 0.4
        self.ball_shape.color = (255, 255, 0, 255)
        self.space.add(self.ball_body, self.ball_shape)
        
        self.collision_handler = self.space.add_collision_handler(0, 0)
        self.collision_handler.separate = self.handle_collision
        
    def create_walls(self) -> None:
        """Create boundary walls"""
        walls = [
            [(0, 0), (0, self.config.HEIGHT)],
            [(self.config.WIDTH, 0), (self.config.WIDTH, self.config.HEIGHT)],
            [(0, self.config.HEIGHT), (self.config.WIDTH, self.config.HEIGHT)],
        ]
        
        for wall in walls:
            shape = pymunk.Segment(self.space.static_body, wall[0], wall[1], 5)
            shape.elasticity = 0.8
            shape.friction = 0.5
            self.space.add(shape)
            
    def handle_collision(self, arbiter, space, data) -> bool:
        """Handle collision between ball and paddle"""
        if not self.ball_hit and hasattr(self, 'ball_body'):
            shapes = arbiter.shapes
            if self.paddle_shape in shapes:
                self.ball_hit = True
                self.hit_successful = True
                self.record_hit_data()
                self.hit_points.append(self.ball_body.position)
                
                # Apply force on hit
                if self.auto_hit:
                    force = (
                        self.current_force * math.cos(self.paddle_body.angle),
                        -self.current_force * math.sin(self.paddle_body.angle)
                    )
                    self.ball_body.apply_impulse_at_local_point(force)
        return True
        
    def record_hit_data(self) -> None:
        """Record data about the hit"""
        hit_data = {
            'angle': math.degrees(self.paddle_body.angle),
            'force': self.current_force,
            'ball_velocity_before': self.ball_body.velocity,
            'ball_velocity_after': self.ball_body.velocity,
            'hit_position': self.ball_body.position,
            'hit_height': self.config.HEIGHT - self.ball_body.position.y
        }
        self.hit_data.append(hit_data)
        
    def calculate_optimal_angle(self) -> float:
        """Calculate the optimal hitting angle based on ball position and velocity"""
        if not hasattr(self, 'ball_body'):
            return 0.0
            
        ball_pos = self.ball_body.position
        ball_vel = self.ball_body.velocity
        
        # Calculate time to reach paddle height
        time_to_paddle = math.sqrt(2 * (ball_pos.y - self.config.PADDLE_POS[1]) / 
                                 self.config.GRAVITY[1])
                                 
        # Predict ball position at impact
        predicted_x = ball_pos.x + ball_vel.x * time_to_paddle
        
        # Calculate angle between paddle and predicted position
        dx = predicted_x - self.config.PADDLE_POS[0]
        dy = self.config.OPTIMAL_HIT_HEIGHT
        
        return math.atan2(dy, dx)
    
    def auto_adjust_angle(self) -> None:
        """Automatically adjust paddle angle for optimal hitting"""
        if self.auto_adjust and hasattr(self, 'ball_body'):
            optimal_angle = self.calculate_optimal_angle()
            current_angle = self.paddle_body.angle
            
            # Smoothly adjust current angle towards optimal angle
            angle_diff = optimal_angle - current_angle
            if abs(angle_diff) > 0.01:
                self.paddle_body.angle += angle_diff * self.config.ANGLE_ADJUSTMENT_RATE
                
    def draw_interface(self) -> None:
        """Draw simulation interface and data"""
        font = pygame.font.Font(None, 36)
        
        # Draw current settings
        angle_text = font.render(f"Paddle Angle: {math.degrees(self.paddle_body.angle):.1f}°", True, (255, 255, 255))
        self.screen.blit(angle_text, (10, 10))
        
        force_text = font.render(f"Hit Force: {self.current_force}", True, (255, 255, 255))
        self.screen.blit(force_text, (10, 50))
        
        if hasattr(self, 'ball_body'):
            height_text = font.render(
                f"Ball Height: {self.config.HEIGHT - self.ball_body.position.y:.1f}",
                True,
                (255, 255, 255)
            )
            self.screen.blit(height_text, (10, 90))
            
            velocity = math.hypot(self.ball_body.velocity.x, self.ball_body.velocity.y)
            vel_text = font.render(f"Ball Velocity: {velocity:.1f}", True, (255, 255, 255))
            self.screen.blit(vel_text, (10, 130))
        
        # Draw feature toggles
        auto_hit_text = font.render(
            f"Auto-Hit: {'ON' if self.auto_hit else 'OFF'} (A to toggle)",
            True,
            (255, 255, 0) if self.auto_hit else (255, 255, 255)
        )
        self.screen.blit(auto_hit_text, (10, self.config.HEIGHT - 120))
        
        auto_adjust_text = font.render(
            f"Auto-Adjust: {'ON' if self.auto_adjust else 'OFF'} (T to toggle)",
            True,
            (255, 255, 0) if self.auto_adjust else (255, 255, 255)
        )
        self.screen.blit(auto_adjust_text, (10, self.config.HEIGHT - 90))
        
        # Draw controls
        controls_text = font.render(
            "SPACE: Drop Ball | Arrows: Adjust Angle | +/-: Adjust Force | R: Reset",
            True,
            (255, 255, 255)
        )
        self.screen.blit(controls_text, (10, self.config.HEIGHT - 30))
        
        # Draw optimal hit zone
        pygame.draw.line(
            self.screen,
            (0, 255, 0) if self.auto_adjust else (100, 100, 100),
            (0, self.config.HEIGHT - self.config.OPTIMAL_HIT_HEIGHT),
            (self.config.WIDTH, self.config.HEIGHT - self.config.OPTIMAL_HIT_HEIGHT),
            2
        )
        
    def draw_trajectories(self) -> None:
        """Draw ball trajectories"""
        if len(self.trajectory_points) > 1:
            pygame.draw.lines(self.screen, (255, 255, 0), False, self.trajectory_points, 2)
            
        if len(self.hit_points) > 1:
            pygame.draw.lines(self.screen, (0, 255, 0), False, self.hit_points, 2)
            
    def reset_simulation(self) -> None:
        """Reset simulation state"""
        if hasattr(self, 'ball_body'):
            self.space.remove(self.ball_body, self.ball_shape)
            delattr(self, 'ball_body')
            delattr(self, 'ball_shape')
        self.trajectory_points.clear()
        self.hit_points.clear()
        self.ball_dropped = False
        self.ball_hit = False
        self.hit_successful = False
        
    def run(self) -> None:
        """Main simulation loop"""
        running = True
        
        while running:
            self.screen.fill((30, 30, 30))
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_simulation()
                    elif event.key == pygame.K_SPACE and not self.ball_dropped:
                        self.create_ball()
                        self.ball_dropped = True
                    elif event.key == pygame.K_a:
                        self.auto_hit = not self.auto_hit
                    elif event.key == pygame.K_t:
                        self.auto_adjust = not self.auto_adjust
                        
            keys = pygame.key.get_pressed()
            
            # Manual paddle control
            if not self.auto_adjust:
                if keys[pygame.K_LEFT]:
                    self.paddle_body.angle += self.config.ANGLE_INCREMENT
                if keys[pygame.K_RIGHT]:
                    self.paddle_body.angle -= self.config.ANGLE_INCREMENT
            
            # Force adjustment
            if keys[pygame.K_EQUALS]:  # Plus key
                self.current_force = min(self.current_force + 100, 5000)
            if keys[pygame.K_MINUS]:
                self.current_force = max(self.current_force - 100, 1000)
            
            # Update paddle angle if auto-adjust is enabled
            if self.auto_adjust:
                self.auto_adjust_angle()
            
            # Track ball
            if hasattr(self, 'ball_body'):
                if self.ball_body.velocity.length > 0.1:
                    current_pos = self.ball_body.position
                    
                    if not self.ball_hit:
                        self.trajectory_points.append(current_pos)
                    else:
                        self.hit_points.append(current_pos)
            
            # Update physics
            self.space.step(1/self.config.FPS)
            
            # Draw everythingt
            self.space.debug_draw(self.draw_options)
            self.draw_trajectories()
            self.draw_interface()
            
            pygame.display.flip()
            self.clock.tick(self.config.FPS)
            
        pygame.quit()
        
        # Print final analysis
        if self.hit_data:
            print("\nHit Analysis:")
            for i, hit in enumerate(self.hit_data, 1):
                print(f"\nHit {i}:")
                print(f"Angle: {hit['angle']:.1f}°")
                print(f"Force: {hit['force']}")
                print(f"Hit Height: {hit['hit_height']:.1f}")
                print(f"Velocity Before: {hit['ball_velocity_before'].length:.1f}")
                print(f"Velocity After: {hit['ball_velocity_after'].length:.1f}")

if __name__ == "__main__":
    simulation = SquashHitSimulation()
    simulation.run()