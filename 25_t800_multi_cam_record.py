"""25: 头部 + 左右手腕三路相机，逐帧渲染落盘（headless 数据采集）。

默认行走 5 s、30 fps、320x240，box 场景带 10 cm 木块真值，输出到
recordings/<task>/<scene>/seed_XXXX/ep_YYYY/:
    meta.json            场景/seed/策略哈希/指标等元信息
    states.csv           逐帧：仿真/墙钟时间、指令、控制量、基座位姿、关节角
    cameras.csv          逐帧：每台相机的世界位姿
    ground_truth.csv     逐帧：木块位姿 + 左右末端位姿
    frames/head/*.png
    frames/wrist_left/*.png
    frames/wrist_right/*.png

用法:
    python3 25_t800_multi_cam_record.py
更多参数（时长/帧率/分辨率/速度指令/场景/seed/输出目录）:
    python3 -m t800_sim.cli record --help
采集后校验:
    python3 -m t800_sim.cli validate --session recordings
"""

from t800_sim.tasks import record

if __name__ == "__main__":
    record.run()
