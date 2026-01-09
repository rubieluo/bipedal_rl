import time
import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np
from playground.open_duck_mini_v2 import joystick

def main():
    # 1. Load the EXACT same environment used for training
    env = joystick.Joystick()
    
    # 2. Get the CPU model (for the viewer)
    model = env._mj_model
    data = mujoco.MjData(model)

    # 3. Initialize the JAX (GPU) logic
    rng = jax.random.PRNGKey(0)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # 4. Run the "Reset" logic to generate the Spawn Position
    print("Resetting environment...")
    state = jit_reset(rng)
    
    # Copy the GPU spawn position to the CPU viewer so we can see it
    data.qpos[:] = np.array(state.data.qpos)
    data.qvel[:] = np.array(state.data.qvel)
    mujoco.mj_forward(model, data)

    print(f"Spawn Height (Z): {data.qpos[2]:.4f} meters")

    # 5. Launch the Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Toggle these to see what happens
        APPLY_GRAVITY = True 
        
        while viewer.is_running():
            step_start = time.time()

            if APPLY_GRAVITY:
                # OPTION A: Run the actual Env physics (The "Game")
                # Create a "Do Nothing" action
                action = jp.zeros(env.action_size)
                state = jit_step(state, action)
                
                # Copy GPU result back to CPU viewer
                data.qpos[:] = np.array(state.data.qpos)
                data.qvel[:] = np.array(state.data.qvel)
                
                # If the env resets automatically, print it
                if state.done:
                    print("Environment Reset Triggered! (Robot died)")
            
            else:
                # OPTION B: Just freeze at the spawn point so you can measure height
                pass

            viewer.sync()
            time.sleep(0.02)

if __name__ == "__main__":
    main()