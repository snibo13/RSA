import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import time

model = mujoco.MjModel.from_xml_path("inverted_pendulum.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 10:
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(model, data)

        # Sync the data back to the viewer.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time.sleep(1 / 30)
