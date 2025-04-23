import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R
import sys

# --- Parameters ---
N_JOINTS = 3 # Shoulder, Elbow, Wrist
LINK_LENGTHS = np.array([0.5, 0.5, 0.4]) # Shoulder->Elbow, Elbow->Wrist, Wrist->End

# Initial State
# Orientation: List of identity quaternions [w, x, y, z] for each joint
initial_orientations_list = [np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(N_JOINTS)]
# Angular Velocity: List of zero vectors [wx, wy, wz] for each joint
initial_angular_velocities_list = [np.zeros(3) for _ in range(N_JOINTS)]

# Flatten initial state for solve_ivp: [q1_w, q1_x, q1_y, q1_z, q2_w, ..., w1_x, w1_y, w1_z, w2_x, ...]
# Order: All quaternions first, then all angular velocities
y0 = np.concatenate(
    [q for q in initial_orientations_list] + \
    [w for w in initial_angular_velocities_list]
)
STATE_SIZE = N_JOINTS * 7
QUAT_SIZE = N_JOINTS * 4 # Index slicing helper

# Applied Torques (3D vector for each joint, around local axes) - TUNE THESE!
# Example: Apply torque around Y axis of Elbow (joint 1) and Z axis of Wrist (joint 2)
APPLIED_TORQUES = np.array([
    [0.3, 0.0, 0.1],   # Shoulder torque (e.g., around Z)
    [0.0, 0.5, 0.0],   # Elbow torque (e.g., around Y for flexion)
    [0.0, 0.4, -0.4]   # Wrist torque (e.g., around Z)
])
# Ensure shape is (N_JOINTS, 3)
assert APPLIED_TORQUES.shape == (N_JOINTS, 3)


# Angle Limits (Simplified: Assuming they apply to a specific Euler angle, e.g., Y-axis rotation 'y')
# Need to be careful about Euler sequence and interpretation
ANGLE_LIMITS_DEG = [
    (-90, 90), # Shoulder Y-rotation limit?
    (0, 150),  # Elbow Y-rotation limit (flexion)? - MOST IMPORTANT
    (-90, 90)  # Wrist Y-rotation limit?
]
ANGLE_LIMITS_RAD = [(np.deg2rad(lim[0]), np.deg2rad(lim[1])) for lim in ANGLE_LIMITS_DEG]
# Which Euler angle index corresponds to these limits (e.g., for 'zyx', Y is index 1)
LIMIT_EULER_INDEX = 1
LIMIT_EULER_ORDER = 'zyx' # Choose a convention, affects interpretation

T_START = 0
T_END = 5 # Shorter time for potentially complex simulation
N_FRAMES = 150
TIME = np.linspace(T_START, T_END, N_FRAMES)

LIMIT_STIFFNESS = 200.0 # May need different tuning
LIMIT_DAMPING = 10.0

# Inertia (Simplified: Assume diagonal inertia tensor, or even identity scaled)
# Using identity means torque directly equals angular acceleration
# Format: Array of shape (N_JOINTS, 3) for diagonal elements Ixx, Iyy, Izz
JOINT_INERTIA_DIAG = np.array([
    [1.0, 1.0, 1.0], # Shoulder inertia (placeholder)
    [1.0, 1.0, 1.0], # Elbow inertia (placeholder)
    [1.0, 1.0, 1.0]  # Wrist inertia (placeholder)
]) * 0.1 # Scale it down maybe
INV_JOINT_INERTIA_DIAG = 1.0 / JOINT_INERTIA_DIAG

# --- Helper for Quaternion Multiplication ---
# --- Add Hamilton Product implementation ---
def hamilton_product(q, p):
    """
    Calculates the Hamilton product q*p for two quaternions.
    Input quaternions are in [w, x, y, z] order.
    """
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

# --- Modify arm_dynamics_3d ---
def arm_dynamics_3d(t, y, applied_torques, n_joints, limits_rad, k, b, inv_inertia_diag):
    """3D arm dynamics using quaternions."""
    dydt = np.zeros_like(y)
    # Ensure y has the correct shape before reshaping
    if y.shape[0] != STATE_SIZE:
         raise ValueError(f"Input state y has incorrect size {y.shape[0]}, expected {STATE_SIZE}")

    quats = y[:QUAT_SIZE].reshape((n_joints, 4))
    omegas = y[QUAT_SIZE:].reshape((n_joints, 3))

    # Initialize derivatives (important!)
    d_quats_dt = np.zeros((n_joints, 4))
    d_omegas_dt = np.zeros((n_joints, 3))

    limit_torques_3d = np.zeros((n_joints, 3)) # Initialize limit torques for this step

    for i in range(n_joints):
        q = quats[i]
        omega = omegas[i] # Angular velocity in body frame of link i

        # --- Calculate Limit Torques ---
        # (Keep your existing limit calculation logic here)
        # Ensure it correctly calculates limit_torques_3d[i]
        # ... (your limit checking code using Euler angles) ...
        try:
            current_rot = R.from_quat([q[1], q[2], q[3], q[0]]) # x,y,z,w format
            euler_angles = current_rot.as_euler(LIMIT_EULER_ORDER, degrees=False)
            angle_to_check = euler_angles[LIMIT_EULER_INDEX]
            lower_lim, upper_lim = limits_rad[i]

            penetration = 0.0
            limit_torque_magnitude = 0.0
            limit_torque_vec = np.zeros(3) # Vector for limit torque

            if angle_to_check < lower_lim:
                penetration = lower_lim - angle_to_check
                spring_force = k * penetration
                axis_idx = 1 # Example: Index for Y axis if LIMIT_EULER_INDEX is for Y rotation
                damping_force = -b * omega[axis_idx]
                limit_torque_magnitude = spring_force + damping_force
                limit_axis = np.zeros(3)
                limit_axis[axis_idx] = 1.0 # Positive torque around the limit axis
                limit_torque_vec = limit_axis * limit_torque_magnitude

            elif angle_to_check > upper_lim:
                penetration = angle_to_check - upper_lim
                spring_force = -k * penetration # Negative torque
                axis_idx = 1
                damping_force = -b * omega[axis_idx]
                limit_torque_magnitude = spring_force + damping_force
                limit_axis = np.zeros(3)
                limit_axis[axis_idx] = -1.0 # Negative torque around the limit axis
                limit_torque_vec = limit_axis * abs(limit_torque_magnitude)

            limit_torques_3d[i] = limit_torque_vec # Store the calculated 3D limit torque

        except ValueError as e:
            print(f"Warning: Euler conversion issue at t={t:.3f}, joint {i}, q={q}. Error: {e}", file=sys.stderr)
            # Keep limit_torques_3d[i] as zero if conversion fails

        # --- Quaternion Derivative ---
        # dq/dt = 0.5 * q * omega_quat
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
        # Use direct Hamilton product implementation
        dq_dt = 0.5 * hamilton_product(q, omega_quat) # <--- USE HAMILTON PRODUCT
        d_quats_dt[i, :] = dq_dt

        # --- Angular Velocity Derivative ---
        # d(omega)/dt = Inv_Inertia * (Applied_Torque + Limit_Torque - Gyro_Effects)
        total_torque = applied_torques[i] + limit_torques_3d[i] # Use calculated limit torque
        d_omega_dt = inv_inertia_diag[i] * total_torque
        d_omegas_dt[i, :] = d_omega_dt

    # Flatten derivatives back into dydt
    dydt[:QUAT_SIZE] = d_quats_dt.flatten()
    dydt[QUAT_SIZE:] = d_omegas_dt.flatten()

    return dydt

# --- Kinematics Function (3D) ---
def calculate_joint_positions_3d(orientations_list, link_lengths):
    """Calculates 3D positions using quaternions."""
    n = len(orientations_list)
    positions = np.zeros((n + 1, 3)) # Base + n joints
    cumulative_rotation = R.identity()

    for i in range(n):
        # Orientation of frame i relative to frame i-1
        q_i = orientations_list[i]
        # Convert to Rotation object [x, y, z, w]
        joint_rotation = R.from_quat([q_i[1], q_i[2], q_i[3], q_i[0]])

        # Orientation of frame i relative to world
        cumulative_rotation = cumulative_rotation * joint_rotation

        # Link vector in local frame i (before rotation by joint i+1)
        # Assumes link extends along the X-axis of the *previous* frame
        link_vector_local = np.array([link_lengths[i], 0.0, 0.0])

        # Position of joint i+1 = Pos(i) + WorldRotation(i) * LinkVector(i)_local
        # Position of joint i is positions[i]
        # We need world rotation of frame *i* to transform link i
        # The cumulative_rotation calculated *after* multiplying by joint_rotation
        # represents the orientation of frame i+1. So we need the one *before*
        # Let's recalculate cleanly:
        # Pos(i+1) = Pos(i) + Rot_world_i * LinkVec_local_i
        # Rot_world_i = Rot_world_0 * Rot_0_1 * ... * Rot_(i-1)_i

        # Alternative structure:
        # current_world_pos = positions[i]
        # current_world_rot = world orientation of frame i
        # next_world_pos = current_world_pos + current_world_rot.apply(link_vector_local)
        # positions[i+1] = next_world_pos
        # next_world_rot = current_world_rot * joint_rotation (Rotation of i+1 relative to i)
        # --> requires storing world rotations explicitly or recalculating

        # Let's try the simpler cumulative approach again, carefully
        # cumulative_rotation tracks orientation of frame i relative to world
        # We need to rotate link_lengths[i] (in frame i's coord system) to world coords

        link_vec_world = cumulative_rotation.apply(link_vector_local)
        positions[i+1] = positions[i] + link_vec_world

    return positions


# --- Solve ODE ---
print("Starting 3D ODE integration...")
sol = solve_ivp(
    fun=arm_dynamics_3d,
    t_span=[T_START, T_END],
    y0=y0,
    method='RK45', # Try RK45 first, may need Radau/BDF if stiff
    t_eval=TIME,
    args=(APPLIED_TORQUES, N_JOINTS, ANGLE_LIMITS_RAD, LIMIT_STIFFNESS, LIMIT_DAMPING, INV_JOINT_INERTIA_DIAG),
    dense_output=True,
    # max_step=0.01 # May need this
)
print("Integration finished.")

# --- Post-Processing: Normalize Quaternions ---
print("Normalizing quaternions...")
solution_y_raw = sol.y # Shape (STATE_SIZE, N_FRAMES)
quats_raw = solution_y_raw[:QUAT_SIZE, :]
norms = np.linalg.norm(quats_raw.reshape(N_JOINTS, 4, -1), axis=1) # Calculate norm for each quat at each time
# Avoid division by zero if norm is zero (shouldn't happen)
norms[norms == 0] = 1.0
# Normalize
quats_normalized = quats_raw.reshape(N_JOINTS, 4, -1) / norms[:, np.newaxis, :]
# Reshape back and place into solution array (or create a new one)
solution_y_normalized = solution_y_raw.copy()
solution_y_normalized[:QUAT_SIZE, :] = quats_normalized.reshape(QUAT_SIZE, -1)
print("Normalization complete.")


# --- Extract Results ---
solution_time = sol.t
# Use normalized solution
solution_y = solution_y_normalized.T # Shape (N_FRAMES, STATE_SIZE)

orientations_flat = solution_y[:, :QUAT_SIZE]
angular_velocities_flat = solution_y[:, QUAT_SIZE:]

# Reshape for easier access
orientations_all_frames = orientations_flat.reshape(len(solution_time), N_JOINTS, 4)
velocities_all_frames = angular_velocities_flat.reshape(len(solution_time), N_JOINTS, 3)


print(f"Simulation ran until t = {solution_time[-1]:.3f} seconds.")
print(f"Solver status: {sol.status}")
print(f"Solver message: {sol.message}")

# --- Calculate Positions for Animation ---
print("Calculating 3D positions for animation...")
all_positions = []
for i in range(len(solution_time)):
    # Pass the list of quaternions for this time step
    current_orientations_list = [orientations_all_frames[i, j, :] for j in range(N_JOINTS)]
    positions = calculate_joint_positions_3d(current_orientations_list, LINK_LENGTHS)
    all_positions.append(positions)
all_positions = np.array(all_positions) # Shape (n_times, n_joints+1, 3)


# --- Animation Setup (Mostly the same) ---
# (Code is identical to previous version - plotting 'all_positions')
print("Setting up animation...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Recalculate bounds based on 3D motion
max_range = 0
if all_positions.size > 0:
    min_coords = np.min(all_positions.reshape(-1, 3), axis=0)
    max_coords = np.max(all_positions.reshape(-1, 3), axis=0)
    center = (max_coords + min_coords) / 2.0
    max_range = np.max(max_coords - min_coords)
    # Ensure max_range is not zero if arm didn't move
    if max_range < 1e-6: max_range = sum(LINK_LENGTHS)
    plot_limit = max_range * 0.6 + 0.5
else:
    print("Warning: No positions generated by simulation.")
    center = np.zeros(3)
    plot_limit = sum(LINK_LENGTHS) * 0.6 + 0.5 # Estimate based on arm length

ax.set_xlim([center[0] - plot_limit, center[0] + plot_limit])
ax.set_ylim([center[1] - plot_limit, center[1] + plot_limit])
ax.set_zlim([center[2] - plot_limit, center[2] + plot_limit])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_aspect('equal', adjustable='box')
title = ax.set_title('3D Arm Pose (Quaternions)')
points, = ax.plot([], [], [], 'ro', markersize=8, label='Joints')
lines = [ax.plot([], [], [], lw=3, c='royalblue')[0] for _ in range(N_JOINTS)] # N_JOINTS lines connect N_JOINTS+1 points

def update(frame):
    if frame >= len(all_positions):
        return [points] + lines + [title]
    current_positions = all_positions[frame] # Shape (N_JOINTS+1, 3)
    points.set_data(current_positions[:, 0], current_positions[:, 1])
    points.set_3d_properties(current_positions[:, 2])
    # Draw N_JOINTS lines connecting the N_JOINTS+1 points
    for i in range(N_JOINTS):
        x_data = [current_positions[i, 0], current_positions[i+1, 0]]
        y_data = [current_positions[i, 1], current_positions[i+1, 1]]
        z_data = [current_positions[i, 2], current_positions[i+1, 2]]
        lines[i].set_data(x_data, y_data)
        lines[i].set_3d_properties(z_data)
    title.set_text(f'3D Arm Pose (Time: {solution_time[frame]:.2f}s)')
    return [points] + lines + [title]

ani = FuncAnimation(fig, update, frames=len(solution_time),
                    interval=max(10, int(1000 * (T_END / N_FRAMES))),
                    blit=False)
ax.legend()
plt.tight_layout()

# --- Show Plot ---
try:
    # Save as GIF using pillow
    ani.save('animation_pillow_3d.gif', writer='pillow')

    # Save as GIF using imagemagick (ensure imagemagick is installed and in PATH)
    ani.save('animation_imagemagick_3d.gif', writer='imagemagick')

    plt.show()
    print("Animation window opened.")
except Exception as e:
    print(f"\nCould not display animation window: {e}")
    # ... (saving suggestion) ...