"""MuJoCo Hello World：木块在重力作用下自由落体。

木块中心初始高度 1 m，每 200 ms 打印一次木块高度，触地后结束。

用法:
    python3 01_mujoco_helloworld.py
"""

import time

import mujoco

PRINT_INTERVAL = 0.2   # 打印间隔（秒）
INITIAL_HEIGHT = 1.0   # 木块中心初始高度（米）
HALF_SIZE = 0.05       # 木块半边长（米），触地时中心高度等于该值

MODEL_XML = f"""
<mujoco model="falling_box">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.5 0.5 0.5 1"/>
    <body name="box" pos="0 0 {INITIAL_HEIGHT}">
      <freejoint name="box_free"/>
      <geom name="box_geom" type="box"
            size="{HALF_SIZE} {HALF_SIZE} {HALF_SIZE}" rgba="0.9 0.5 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def print_state(data):
    # freejoint 的 qpos[0:3] 是位置，qvel[0:3] 是线速度
    height = data.qpos[2]
    vz = data.qvel[2]
    print(f"{data.time:12.3f} {height:12.4f} {vz:14.3f}")


def main():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    print(f"木块初始高度 {INITIAL_HEIGHT:.3f} m，重力 9.81 m/s^2，打印间隔 {PRINT_INTERVAL * 1000:.0f} ms")
    print(f"{'仿真时间(s)':>10} {'高度(m)':>12} {'垂直速度(m/s)':>14}")

    print_state(data)
    next_print_time = PRINT_INTERVAL
    start_wall = time.time()

    while data.time < 5.0:
        mujoco.mj_step(model, data)

        # 按真实时间节奏推进仿真
        lag = data.time - (time.time() - start_wall)
        if lag > 0:
            time.sleep(lag)

        if data.time + 1e-9 >= next_print_time:
            print_state(data)
            next_print_time += PRINT_INTERVAL

        # 木块中心高度降到半边长即为触地，结束仿真
        if data.qpos[2] <= HALF_SIZE + 1e-3:
            print(f"木块触地，用时 {data.time:.3f} s，触地速度 {data.qvel[2]:.3f} m/s")
            break


if __name__ == "__main__":
    main()
