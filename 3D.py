import pybullet as p
import pybullet_data
import numpy as np
import time

# Initialize PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Machine parameters based on your prototype equation (20/10=35)
MACHINE_ANGLE_RATIO = 3.5  # Derived from 35/10
FORCE_MULTIPLIER = 35      # From equation result

def create_hitting_machine():
    # Base platform
    base_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.1])
    base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.1], rgbaColor=[0.3, 0.3, 0.3, 1])
    base = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=base_coll, 
                            baseVisualShapeIndex=base_vis, basePosition=[0, 0, 0.1])

    # Rotating arm
    arm_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.0, 0.1, 0.05])
    arm_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[1.0, 0.1, 0.05], rgbaColor=[0.8, 0.8, 0.8, 1])
    arm = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=arm_coll,
                           baseVisualShapeIndex=arm_vis, basePosition=[0, 0, 0.3])

    # Create revolute joint between base and arm
    constraint = p.createConstraint(base, -1, arm, -1,  # Issue: Possibly incorrect joint setup
                                   p.JOINT_REVOLUTE, [0, 0, 1],  # Should be JOINT_REVOLUTE instead of PRISMATIC
                                   [0, 0, 0.3], [0, 0, 0])
    
    p.changeConstraint(constraint, maxForce=500)

    return base, arm, constraint  # Ensure it returns three values

# Create objects
machine_base, machine_arm = create_hitting_machine()

# Get joint index
joint_index = 0  # First (and only) joint of the arm

# Apply control
p.setJointMotorControl2(bodyUniqueId=machine_arm,
                        jointIndex=joint_index,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=0)


def create_squash_ball():
    ball_radius = 0.05
    ball_coll = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
    ball_vis = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 0, 0, 1])
    ball = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=ball_coll,
                            baseVisualShapeIndex=ball_vis, basePosition=[0, 0, 1])
    p.changeDynamics(ball, -1, restitution=0.8, lateralFriction=0.5)
    return ball

# Create objects
machine_base, machine_arm, machine_joint = create_hitting_machine()
ball = create_squash_ball()

# Control parameters
current_angle = 0
angle_increment = np.pi/36  # 5 degrees

while True:
    keys = p.getKeyboardEvents()
    
    # Angle control (Q/E keys)
    if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
        current_angle += angle_increment
    if ord('e') in keys and keys[ord('e')] & p.KEY_WAS_TRIGGERED:
        current_angle -= angle_increment
        
    # Apply angle to machine
    p.setJointMotorControl2(bodyUniqueId=machine_base,
                           jointIndex=machine_joint,
                           controlMode=p.POSITION_CONTROL,
                           targetPosition=current_angle)
    
    # Hit ball with spacebar (using 20/10=35 ratio)
    if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
        # Calculate force based on machine angle and prototype ratio
        force_magnitude = abs(current_angle) * FORCE_MULTIPLIER * MACHINE_ANGLE_RATIO
        force_vector = [
            force_magnitude * np.sin(current_angle),
            0,
            force_magnitude * np.cos(current_angle)
        ]
        p.applyExternalForce(ball, -1, force_vector, [0, 0, 0], p.WORLD_FRAME)

    # Simulation step
    p.stepSimulation()
    time.sleep(1.0/240.0)

p.disconnect()