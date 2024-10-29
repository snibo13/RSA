import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import time

model = mujoco.MjModel.from_xml_path("inverted_pendulum.xml")
data = mujoco.MjData(model)

# Number of states and controlled inputs
nx = 2 * model.nv
nu = model.nu

# Finite-difference parameters
ϵ = 1e-6
centred = True

# Compute the Jacobians
A = np.zeros((nx, nx))
B = np.zeros((nx, nu))
mujoco.mjd_transitionFD(model, data, ϵ, centred, A, B, None, None)

Q = np.diag([1, 10, 1, 15])
R = np.eye(nu)

# Solve the continuous-time Riccati equation
# N = 0
# Solve discrete Riccati equation.
P = scipy.linalg.solve_discrete_are(A, B, Q, R)
print(f"P = {P}")

# Compute the feedback gain matrix K.
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
print(f"K = {K}")
input()

control_signals = []
states = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 10:

        # Compute the control signal
        q = np.concatenate([data.qpos, data.qvel])
        u = -K @ q
        data.ctrl = u

        # Store the control signal and state for later plotting
        control_signals.append(u)
        states.append(q)

        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(model, data)

        # Sync the data back to the viewer.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time.sleep(1 / 60)

states = np.array(states).T

plt.subplot(3, 1, 1)
plt.plot(control_signals)
plt.title("Control signals")
plt.xlabel("Time step")
plt.ylabel("Control signal")

plt.subplot(3, 1, 2)
plt.plot(states[0], label="x")
plt.plot(states[2], label="v")
plt.title("States")
plt.xlabel("Time step")
plt.ylabel("State")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(states[1], label="θ")
plt.plot(states[3], label="ω")
plt.title("States")
plt.xlabel("Time step")
plt.ylabel("State")
plt.legend()
plt.show()
