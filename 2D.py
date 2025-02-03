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
    PADDLE_LENGTH: int = 120
    PADDLE_THICKNESS: int = 10
    BALL_RADIUS: int = 10
    BASE_HIT_FORCE: int = 500
    ANGLE_INCREMENT: float = np.pi / 36  # 5 degree steps
    GRAVITY: Tuple[float, float] = (0, 900)
    BALL_START_POS: Tuple[int, int] = (450, 200)  # Starting more to the left
    PADDLE_POS: Tuple[int, int] = (400, 600)      # Starting more to the left
    PADDLE_MOVEMENT_SPEED: float = 100  # Increased from 5.0 to 15.0
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
        self.hit_mode = "system"  # "system" or "surface"
        self.paddle_start_pos = self.config.PADDLE_POS
        self.movement_time = 0
        
    def setup_pygame(self) -> None:
        """Initialize Pygame components"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.config.WIDTH, self.config.HEIGHT))
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        pygame.display.set_caption("Squash Ball Hit Analysis - Two Modes")
        
    def setup_pymunk(self) -> None:
        """Initialize Pymunk space and physics"""
        self.space = pymunk.Space()
        self.space.gravity = self.config.GRAVITY
        
    def create_objects(self) -> None:
        """Create all simulation objects"""
        self.create_walls()
        self.create_paddle()
        
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

    def handle_collision(self, arbiter, space, data) -> bool:
        """Handle collision between ball and paddle"""
        if not self.ball_hit and hasattr(self, 'ball_body'):
            shapes = arbiter.shapes
            if self.paddle_shape in shapes:
                self.ball_hit = True
                self.hit_successful = True
                self.record_hit_data()
                self.hit_points.append(self.ball_body.position)
                
                # Calculate and apply force based on mode
                if self.hit_mode == "system":
                    # System mode: 45-degree force
                    force_angle = math.radians(45)
                    force = (
                        self.current_force * math.cos(force_angle),
                        -self.current_force * math.sin(force_angle)
                    )
                else:
                    # Surface mode: horizontal movement with 45-degree surface deflection
                    force = (
                        self.current_force,
                        -self.current_force * math.tan(math.radians(45))
                    )
                
                self.ball_body.apply_impulse_at_local_point(force)
        return True

    def record_hit_data(self) -> None:
        """Record data about the hit"""
        hit_data = {
            'mode': self.hit_mode,
            'angle': 45,  # Fixed 45-degree angle for both modes
            'force': self.current_force,
            'ball_velocity_before': self.ball_body.velocity,
            'ball_velocity_after': self.ball_body.velocity,
            'hit_position': self.ball_body.position,
            'hit_height': self.config.HEIGHT - self.ball_body.position.y
        }
        self.hit_data.append(hit_data)

    def update_paddle(self) -> None:
        """Update paddle position and angle based on hit mode"""
        self.movement_time += 1/60  # Assuming 60 FPS
        
        if self.hit_mode == "system":
            # Move paddle along 45-degree path
            movement = self.config.PADDLE_MOVEMENT_SPEED
            dx = movement * math.cos(math.radians(45))
            dy = -movement * math.sin(math.radians(45))
            
            new_x = self.paddle_start_pos[0] + dx * self.movement_time
            new_y = self.paddle_start_pos[1] + dy * self.movement_time
            
            # Keep paddle angle at 45 degrees
            self.paddle_body.angle = math.radians(45)
            self.paddle_body.position = (new_x, new_y)
        else:
            # Move paddle horizontally
            dx = self.config.PADDLE_MOVEMENT_SPEED
            new_x = self.paddle_start_pos[0] + dx * self.movement_time
            
            # Keep vertical position constant
            self.paddle_body.position = (new_x, self.paddle_start_pos[1])
            # Keep surface angle at 45 degrees
            self.paddle_body.angle = math.radians(45)

    def draw_interface(self) -> None:
        """Draw simulation interface and data"""
        font = pygame.font.Font(None, 36)
        
        # Draw current mode
        mode_text = font.render(
            f"Mode: {self.hit_mode.upper()} (M to switch)", 
            True, 
            (255, 255, 0)
        )
        self.screen.blit(mode_text, (10, 10))
        
        # Draw force
        force_text = font.render(
            f"Force: {self.current_force}", 
            True, 
            (255, 255, 255)
        )
        self.screen.blit(force_text, (10, 50))
        
        # Draw controls
        controls_text = font.render(
            "SPACE: Drop Ball | R: Reset | +/-: Adjust Force",
            True,
            (255, 255, 255)
        )
        self.screen.blit(controls_text, (10, self.config.HEIGHT - 30))

        # Draw movement paths
        if self.hit_mode == "system":
            # Draw 45-degree movement path
            pygame.draw.line(
                self.screen,
                (100, 100, 255),
                self.paddle_start_pos,
                (self.paddle_start_pos[0] + 300, self.paddle_start_pos[1] - 300),
                2
            )
        else:
            # Draw horizontal movement path
            pygame.draw.line(
                self.screen,
                (100, 100, 255),
                self.paddle_start_pos,
                (self.paddle_start_pos[0] + 300, self.paddle_start_pos[1]),
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
            
        self.paddle_body.position = self.paddle_start_pos
        self.movement_time = 0
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
                    elif event.key == pygame.K_m:
                        self.hit_mode = "surface" if self.hit_mode == "system" else "system"
                        self.reset_simulation()
                    elif event.key == pygame.K_EQUALS:  # Plus key
                        self.current_force = min(self.current_force + 100, 5000)
                    elif event.key == pygame.K_MINUS:
                        self.current_force = max(self.current_force - 100, 1000)
            
            if self.ball_dropped and not self.ball_hit:
                self.update_paddle()
            
            # Track ball trajectory
            if hasattr(self, 'ball_body'):
                if self.ball_body.velocity.length > 0.1:
                    current_pos = self.ball_body.position
                    if not self.ball_hit:
                        self.trajectory_points.append(current_pos)
                    else:
                        self.hit_points.append(current_pos)
            
            # Update physics
            self.space.step(1/self.config.FPS)
            
            # Draw everything
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
                print(f"Mode: {hit['mode'].upper()}")
                print(f"Angle: {hit['angle']}°")
                print(f"Force: {hit['force']}")
                print(f"Hit Height: {hit['hit_height']:.1f}")
                print(f"Velocity Before: {hit['ball_velocity_before'].length:.1f}")
                print(f"Velocity After: {hit['ball_velocity_after'].length:.1f}")

if __name__ == "__main__":
    simulation = SquashHitSimulation()
    simulation.run()