import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R
import sys
import time # For basic profiling if needed

# --- Arm Configuration ---
JOINT_TYPES = ['spherical', 'revolute', 'spherical'] # Shoulder, Elbow, Wrist
N_JOINTS = len(JOINT_TYPES)

# Define hinge axis for revolute joints (relative to the *previous* link's frame)
# Example: Elbow (joint 1) rotates around the Y-axis of the frame attached to the end of the upper arm (link 0)
HINGE_AXES = {
    1: np.array([0.0, 1.0, 0.0]) # Elbow hinge axis (local Y)
}
# Define link lengths (Shoulder-Elbow, Elbow-Wrist, Wrist-EndEffector)
LINK_LENGTHS = np.array([0.5, 0.5, 0.2])
assert len(LINK_LENGTHS) == N_JOINTS

# --- State Vector Setup ---
# Define state size for each joint type
STATE_INFO = {
    'spherical': {'size': 7, 'q_slice': slice(0, 4), 'w_slice': slice(4, 7)}, # Quat (w,x,y,z) + AngVel (x,y,z)
    'revolute':  {'size': 2, 'q_slice': slice(0, 1), 'w_slice': slice(1, 2)}  # Angle (th) + AngVel (th_dot)
}

# Calculate total state size and start/end indices for each joint's state block
joint_state_indices = []
joint_q_indices = [] # Indices for orientation part (quat or angle)
joint_w_indices = [] # Indices for velocity part (3D or 1D)
current_idx = 0
for i in range(N_JOINTS):
    j_type = JOINT_TYPES[i]
    info = STATE_INFO[j_type]
    size = info['size']
    q_size = info['q_slice'].stop - info['q_slice'].start
    w_size = info['w_slice'].stop - info['w_slice'].start

    # Overall slice for this joint in y
    joint_state_indices.append(slice(current_idx, current_idx + size))
    # Slice for orientation part within the overall state y
    joint_q_indices.append(slice(current_idx + info['q_slice'].start, current_idx + info['q_slice'].stop))
     # Slice for velocity part within the overall state y
    joint_w_indices.append(slice(current_idx + info['w_slice'].start, current_idx + info['w_slice'].stop))

    current_idx += size
TOTAL_STATE_SIZE = current_idx
print(f"Total state size: {TOTAL_STATE_SIZE}") # Should be 7+2+7 = 16

# --- Initial Conditions ---
y0_list = []
for i in range(N_JOINTS):
    j_type = JOINT_TYPES[i]
    if j_type == 'spherical':
        y0_list.extend([1.0, 0.0, 0.0, 0.0]) # Initial quaternion (w,x,y,z) = identity
        y0_list.extend([0.0, 0.0, 0.0])       # Initial 3D omega = zero vector
    elif j_type == 'revolute':
        y0_list.append(0.0) # Initial angle (theta) = 0
        y0_list.append(0.0) # Initial angular velocity (theta_dot) = 0
y0 = np.array(y0_list)
assert y0.shape[0] == TOTAL_STATE_SIZE, "Initial state vector size mismatch!"

# --- Physics Parameters ---
# Applied Torques: Provide a 3D vector for each joint.
# The dynamics function will only use the relevant component(s) for each joint type.
APPLIED_TORQUES = np.array([
    [0.0, 0.1, 0.1],   # Shoulder torque (around Y and Z)
    [0.1, 0.8, 0.0],   # Elbow torque (only Y component used by hinge)
    [0.2, 0.0, -0.5]   # Wrist torque (around Z)
]) * 0.5 # Scale down torques maybe
assert APPLIED_TORQUES.shape == (N_JOINTS, 3)

# Angle Limits: Define limits appropriately for each joint type/axis
# Spherical joints need more complex limit definitions ideally, but we'll stick to
# checking one Euler angle for simplicity for now. Revolute limits are straightforward.
# Format: List of tuples. For Revolute: (min_angle, max_angle). For Spherical: (min_euler_y, max_euler_y) - adapt as needed
ANGLE_LIMITS_DEG = [
    (-120, 120), # Shoulder 'Y' Euler limit (example)
    (0, 150),    # Elbow Hinge limit
    (-90, 90)    # Wrist 'Y' Euler limit (example)
]
ANGLE_LIMITS_RAD = [(np.deg2rad(lim[0]), np.deg2rad(lim[1])) for lim in ANGLE_LIMITS_DEG]
LIMIT_EULER_INDEX = 1 # Which Euler angle to check for spherical joints
LIMIT_EULER_ORDER = 'zyx'

# Inertia: Define inertia relevant to each joint type
# Spherical: 3D diagonal inertia [Ixx, Iyy, Izz]
# Revolute: Scalar inertia around the hinge axis
JOINT_INERTIA = [
    np.array([0.1, 0.1, 0.1]), # Shoulder diag inertia
    0.05,                      # Elbow scalar inertia around hinge axis
    np.array([0.02, 0.02, 0.02]) # Wrist diag inertia
]
INV_JOINT_INERTIA = []
for I in JOINT_INERTIA:
    if np.isscalar(I):
        INV_JOINT_INERTIA.append(1.0 / I if I != 0 else np.inf)
    else: # Assume numpy array
        # Handle potential zero inertia components
        inv_I = np.zeros_like(I)
        non_zero_mask = I != 0
        inv_I[non_zero_mask] = 1.0 / I[non_zero_mask]
        inv_I[~non_zero_mask] = np.inf # Assign infinity for zero inertia
        INV_JOINT_INERTIA.append(inv_I)


LIMIT_STIFFNESS = 250.0
LIMIT_DAMPING = 12.0

# --- Simulation Time ---
T_START = 0
T_END = 6 # Shorter time for checking
N_FRAMES = 200
TIME = np.linspace(T_START, T_END, N_FRAMES)

# --- Helper Function ---
def hamilton_product(q, p):
    """ Calculates the Hamilton product q*p for two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

# --- Dynamics Function (Mixed Types) ---
def arm_dynamics_mixed(t, y, applied_torques, joint_types, limits_rad, k, b, inv_inertia, hinge_axes):
    """Handles dynamics for mixed spherical and revolute joints."""
    dydt = np.zeros_like(y)

    for i in range(N_JOINTS):
        j_type = joint_types[i]
        j_indices = joint_state_indices[i] # Slice for the whole state of joint i
        q_indices = joint_q_indices[i]     # Slice for orientation part
        w_indices = joint_w_indices[i]     # Slice for velocity part

        j_state = y[j_indices]

        limit_torque = 0.0 # Initialize (can be scalar or vector)

        # --- Calculate Limit Torques ---
        if j_type == 'spherical':
            q = y[q_indices] # Shape (4,)
            omega = y[w_indices] # Shape (3,)
            limit_torque = np.zeros(3) # Vector for spherical joint limit torque
            try:
                # Check Euler angle limit (example using Y 'zyx')
                current_rot = R.from_quat([q[1], q[2], q[3], q[0]]) # x,y,z,w
                euler_angles = current_rot.as_euler(LIMIT_EULER_ORDER, degrees=False)
                angle_to_check = euler_angles[LIMIT_EULER_INDEX]
                lower_lim, upper_lim = limits_rad[i]
                axis_idx = LIMIT_EULER_INDEX # Assumes index corresponds to axis (0=X, 1=Y, 2=Z for 'xyz')

                penetration = 0.0
                torque_magnitude = 0.0

                if angle_to_check < lower_lim:
                    penetration = lower_lim - angle_to_check
                    spring_force = k * penetration
                    damping_force = -b * omega[axis_idx] # Damp along checked axis
                    torque_magnitude = spring_force + damping_force
                    limit_axis = np.zeros(3); limit_axis[axis_idx] = 1.0 # Torque axis
                    limit_torque = limit_axis * torque_magnitude

                elif angle_to_check > upper_lim:
                    penetration = angle_to_check - upper_lim
                    spring_force = -k * penetration
                    damping_force = -b * omega[axis_idx]
                    torque_magnitude = spring_force + damping_force
                    limit_axis = np.zeros(3); limit_axis[axis_idx] = -1.0 # Torque axis
                    limit_torque = limit_axis * abs(torque_magnitude)

            except Exception as e: # Catch potential Rotation errors
                print(f"Warning: Limit check error at t={t:.3f}, joint {i} (Spherical). Error: {e}", file=sys.stderr)
                limit_torque = np.zeros(3) # Default to zero torque on error

        elif j_type == 'revolute':
            theta = y[q_indices][0] # Scalar angle
            theta_dot = y[w_indices][0] # Scalar angular velocity
            limit_torque = 0.0 # Scalar for revolute joint limit torque
            lower_lim, upper_lim = limits_rad[i]

            penetration = 0.0
            if theta < lower_lim:
                penetration = lower_lim - theta
                spring_force = k * penetration
                damping_force = -b * theta_dot
                limit_torque = spring_force + damping_force
            elif theta > upper_lim:
                penetration = angle_to_check - upper_lim # ERROR: Should use theta
                penetration = theta - upper_lim # Corrected
                spring_force = -k * penetration
                damping_force = -b * theta_dot
                limit_torque = spring_force + damping_force

        # --- Calculate State Derivatives ---
        if j_type == 'spherical':
            q = y[q_indices]
            omega = y[w_indices]
            inv_I = inv_inertia[i] # Shape (3,) diagonal inverse inertia

            # Quaternion Derivative: dq/dt = 0.5 * q * omega_quat
            omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
            dq_dt = 0.5 * hamilton_product(q, omega_quat)

            # Angular Velocity Derivative: d(omega)/dt = Inv_I * (Torque_applied + Torque_limit)
            total_torque = applied_torques[i] + limit_torque # limit_torque is already 3D vector
            d_omega_dt = inv_I * total_torque # Element-wise multiplication

            # Store derivatives
            dydt[q_indices] = dq_dt
            dydt[w_indices] = d_omega_dt

        elif j_type == 'revolute':
            theta_dot = y[w_indices][0]
            inv_I = inv_inertia[i] # Scalar inverse inertia
            hinge_axis = hinge_axes[i] # Axis in previous frame's coords

            # Angle Derivative: d(theta)/dt = theta_dot
            d_theta_dt = theta_dot

            # Angular Velocity Derivative: d(theta_dot)/dt = Inv_I * (Torque_applied_scalar + Torque_limit_scalar)
            # Project applied 3D torque onto the hinge axis
            applied_torque_scalar = np.dot(applied_torques[i], hinge_axis)
            total_torque_scalar = applied_torque_scalar + limit_torque # limit_torque is already scalar
            d_theta_dot_dt = inv_I * total_torque_scalar

            # Store derivatives
            dydt[q_indices] = d_theta_dt # Stores scalar in the correct slice start
            dydt[w_indices] = d_theta_dot_dt # Stores scalar in the correct slice start

    return dydt


# --- Kinematics Function (Mixed Types) ---
def calculate_joint_positions_mixed(y_state_at_time, joint_types, link_lengths, hinge_axes):
    """Calculates 3D positions for mixed joint types."""
    n = len(joint_types)
    positions = np.zeros((n + 1, 3)) # Base + n joints = n+1 points
    cumulative_rotation = R.identity() # Orientation of current frame relative to world

    for i in range(n):
        j_type = joint_types[i]
        q_indices = joint_q_indices[i] # Slice for orientation part in y

        joint_rotation = R.identity() # Rotation of frame i relative to frame i-1

        if j_type == 'spherical':
            q = y_state_at_time[q_indices] # Get quaternion [w,x,y,z]
            # Ensure quaternion is valid before creating Rotation object
            if not np.isclose(np.linalg.norm(q), 1.0):
                # This should ideally be handled by post-solve normalization
                # print(f"Warning: Normalizing quaternion in kinematics for joint {i}")
                q = q / np.linalg.norm(q)
            if np.isclose(np.linalg.norm(q), 0.0):
                 print(f"ERROR: Zero norm quaternion encountered in kinematics for joint {i}", file=sys.stderr)
                 joint_rotation = R.identity() # Fallback
            else:
                joint_rotation = R.from_quat([q[1], q[2], q[3], q[0]]) # x,y,z,w

        elif j_type == 'revolute':
            theta = y_state_at_time[q_indices][0] # Get angle
            axis = hinge_axes[i]
            joint_rotation = R.from_rotvec(theta * axis)

        # Calculate position of the end of link i (start of link i+1)
        # Link vector extends along X-axis of the *current* frame (frame i)
        link_vector_local = np.array([link_lengths[i], 0.0, 0.0])
        # Transform link vector to world coordinates using orientation of frame i
        link_vec_world = cumulative_rotation.apply(link_vector_local)
        positions[i+1] = positions[i] + link_vec_world

        # Update cumulative rotation for the next frame (frame i+1)
        # Rot_world_(i+1) = Rot_world_i * Rot_i_(i+1)
        # Here, joint_rotation represents Rot_i_(i+1)
        cumulative_rotation = cumulative_rotation * joint_rotation


    return positions

# --- Solve ODE ---
print("Starting Mixed-Joint ODE integration...")
start_time = time.time()
sol = solve_ivp(
    fun=arm_dynamics_mixed,
    t_span=[T_START, T_END],
    y0=y0,
    method='RK45', # Start with RK45, try 'Radau' or 'BDF' if stiff
    t_eval=TIME,
    args=(APPLIED_TORQUES, JOINT_TYPES, ANGLE_LIMITS_RAD, LIMIT_STIFFNESS, LIMIT_DAMPING, INV_JOINT_INERTIA, HINGE_AXES),
    dense_output=True,
    # rtol=1e-5, atol=1e-8 # May need tighter tolerances
    # max_step=0.01
)
end_time = time.time()
print(f"Integration finished in {end_time - start_time:.2f} seconds.")

# --- Post-Processing: Normalize Quaternions ---
print("Normalizing quaternions in solution...")
solution_y_raw = sol.y # Shape (TOTAL_STATE_SIZE, N_FRAMES)
solution_y_normalized = solution_y_raw.copy()

for i in range(N_JOINTS):
    if JOINT_TYPES[i] == 'spherical':
        q_indices_in_y = joint_q_indices[i] # Get the slice for this joint's quaternion
        quats_for_joint = solution_y_normalized[q_indices_in_y, :] # Shape (4, N_FRAMES)
        # Check for zero norms before normalizing
        norms = np.linalg.norm(quats_for_joint, axis=0) # Norm along quat axis for each time step
        non_zero_mask = norms > 1e-10 # Avoid division by near-zero
        # Normalize only where norm is non-zero
        quats_for_joint[:, non_zero_mask] /= norms[non_zero_mask]
        # Optional: Handle zero norms if they occur (e.g., reset to identity)
        zero_norm_indices = np.where(~non_zero_mask)[0]
        if len(zero_norm_indices) > 0:
             print(f"Warning: Found {len(zero_norm_indices)} near-zero norm quaternions for joint {i}. Resetting to identity.", file=sys.stderr)
             identity_quat = np.array([1.0, 0.0, 0.0, 0.0])[:, np.newaxis]
             quats_for_joint[:, zero_norm_indices] = identity_quat

        # Place normalized quaternions back
        solution_y_normalized[q_indices_in_y, :] = quats_for_joint

print("Normalization complete.")


# --- Extract Results ---
solution_time = sol.t
solution_y = solution_y_normalized.T # Shape (N_FRAMES, TOTAL_STATE_SIZE)

print(f"Simulation ran until t = {solution_time[-1]:.3f} seconds.")
print(f"Solver status: {sol.status}") # 0=success, 1=terminated by event, <0=failed
print(f"Solver message: {sol.message}")


# --- Calculate Positions for Animation ---
print("Calculating 3D positions for animation...")
all_positions = []
start_fk_time = time.time()
for k in range(len(solution_time)):
    # Pass the full state vector for the current time step
    positions = calculate_joint_positions_mixed(solution_y[k, :], JOINT_TYPES, LINK_LENGTHS, HINGE_AXES)
    all_positions.append(positions)
all_positions = np.array(all_positions)
end_fk_time = time.time()
print(f"FK calculation finished in {end_fk_time - start_fk_time:.2f} seconds.")

# --- Animation Setup (Identical to previous 3D version) ---
print("Setting up animation...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

max_range = 0
if all_positions.size > 0:
    min_coords = np.min(all_positions.reshape(-1, 3), axis=0)
    max_coords = np.max(all_positions.reshape(-1, 3), axis=0)
    center = (max_coords + min_coords) / 2.0
    max_range = np.max(max_coords - min_coords)
    if max_range < 1e-6: max_range = sum(LINK_LENGTHS) * 2 # Handle static case
    plot_limit = max_range * 0.6 + 0.5
else:
    print("Warning: No positions generated by simulation.")
    center = np.zeros(3)
    plot_limit = sum(LINK_LENGTHS) * 0.6 + 0.5

ax.set_xlim([center[0] - plot_limit, center[0] + plot_limit])
ax.set_ylim([center[1] - plot_limit, center[1] + plot_limit])
ax.set_zlim([center[2] - plot_limit, center[2] + plot_limit])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_aspect('equal', adjustable='box')
title_obj = ax.set_title('3D Arm Pose (Mixed Joints)') # Store title object
points, = ax.plot([], [], [], 'ro', markersize=8, label='Joints')
lines = [ax.plot([], [], [], lw=3, c='royalblue')[0] for _ in range(N_JOINTS)] # N_JOINTS lines

def update(frame):
    if frame >= len(all_positions):
        # Prevent animation from crashing if frame index exceeds data length
        # This can happen if the solver fails early or produces fewer steps than N_FRAMES
        return [points] + lines + [title_obj]
    current_positions = all_positions[frame]
    points.set_data(current_positions[:, 0], current_positions[:, 1])
    points.set_3d_properties(current_positions[:, 2])
    for i in range(N_JOINTS):
        x_data = [current_positions[i, 0], current_positions[i+1, 0]]
        y_data = [current_positions[i, 1], current_positions[i+1, 1]]
        z_data = [current_positions[i, 2], current_positions[i+1, 2]]
        lines[i].set_data(x_data, y_data)
        lines[i].set_3d_properties(z_data)
    title_obj.set_text(f'Arm Pose (Mixed Joints - Time: {solution_time[frame]:.2f}s)')
    return [points] + lines + [title_obj]

ani = FuncAnimation(fig, update, frames=N_FRAMES, # Use N_FRAMES or len(solution_time)? Use len(solution_time)
                    interval=max(10, int(1000 * (T_END / len(solution_time)))) if len(solution_time)>1 else 50,
                    blit=False)
ax.legend()
plt.tight_layout()

# --- Show Plot ---
try:
    plt.show()
    print("Animation window opened.")
except Exception as e:
    print(f"\nCould not display animation window: {e}")
    # Consider saving animation if show fails
    # print("Attempting to save animation...")
    # ani.save('mixed_joint_arm.mp4', writer='ffmpeg', fps=30)