import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import time
import gymnasium as gym
import glfw


env = gym.make("InvertedDoublePendulum-v5", render_mode="human")


obs, info = env.reset()

viewer = env.unwrapped.mujoco_renderer.viewer
# Resize the window (set desired width and height in pixels)
width = 1280
height = 720
glfw.set_window_size(viewer.window, width, height)
input()

nv = env.unwrapped.model.nv
nu = env.unwrapped.model.nu

# Change integrator to enable transition dynamics computation later
env.unwrapped.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER


# Number of states and controlled inputs
nx = 2 * nv  # X, θ1, θ2, V, ω1, ω2
nu = nu

# Finite-difference parameters
ϵ = 1e-6
centred = True

# Compute the Jacobians
A = np.zeros((nx, nx))
B = np.zeros((nx, nu))
mujoco.mjd_transitionFD(
    env.unwrapped.model, env.unwrapped.data, ϵ, centred, A, B, None, None
)

Q = np.diag([10, 10, 10, 1, 15, 15])
R = 0.1 * np.eye(nu)

# Solve the continuous-time Riccati equation
# N = 0
# Solve discrete Riccati equation.
P = scipy.linalg.solve_discrete_are(A, B, Q, R)

# Compute the feedback gain matrix K.
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
print(K)

control_signals = []
states = []

q_star = np.array([0, 0, 0, 0, 0, 0])
start = time.time()

# Access the viewer object


while time.time() - start < 10:

    # Compute the control signal
    data = env.unwrapped.data
    q = np.concatenate([data.qpos, data.qvel])
    u = -K @ (q - q_star)
    data.ctrl = u

    # Store the control signal and state for later plotting
    control_signals.append(u)
    states.append(q)

    step_start = time.time()

    env.step(u)

    env.render()

    time.sleep(1 / 60)

print(f"Final state: {q}")

states = np.array(states).T

plt.subplot(3, 1, 1)
plt.plot(control_signals)
plt.title("Control signals")
plt.xlabel("Time step")
plt.ylabel("Control signal")

plt.subplot(3, 1, 2)
plt.plot(states[0], label="x", color="red")
plt.plot(states[2], label="v", color="blue")
plt.plot(q_star[0] * np.ones_like(states[0]), label="x*", linestyle="--", color="red")
plt.plot(q_star[2] * np.ones_like(states[2]), label="v*", linestyle="--", color="blue")
plt.title("States")
plt.xlabel("Time step")
plt.ylabel("State")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(states[1], label="θ", color="green")
plt.plot(states[3], label="ω", color="purple")
plt.plot(q_star[1] * np.ones_like(states[1]), label="θ*", linestyle="--", color="green")
plt.plot(
    q_star[3] * np.ones_like(states[3]), label="ω*", linestyle="--", color="purple"
)
plt.title("States")
plt.xlabel("Time step")
plt.ylabel("State")
plt.legend()
plt.show()
