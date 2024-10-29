import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import time
import gymnasium as gym
import glfw
import cvxpy as cp


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


# Define time horizon and time step
N = 10  # Prediction horizon
dt = 0.1  # Time step

# Define state and control input variables
x = cp.Variable((nv, N + 1))
u = cp.Variable((1, N))  # Control input: torque

# Define cost function and constraints
cost = 0
constraints = []

for t in range(N):
    # Cost function (e.g., quadratic cost on state deviation and control effort)
    cost += cp.quad_form(x[:, t], np.eye(nv)) + cp.quad_form(u[:, t], np.eye(nu))

    # Dynamics constraints (linearized or full nonlinear dynamics)
    constraints += [x[:, t + 1] == x[:, t] + dt * (A @ x[:, t] + B @ u[:, t])]

# Terminal state constraint (if needed)
constraints += [x[:, N] == np.zeros(4)]

# Solve optimization problem
problem = cp.Problem(cp.Minimize(cost), constraints)
problem.solve()

print("Optimal cost:", problem.value)
print("Optimal control signal:", u.value)
