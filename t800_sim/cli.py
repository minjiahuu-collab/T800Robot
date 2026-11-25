"""命令行入口。

用法:
    python3 -m t800_sim.cli teleop              # 20: WASD 遥控
    python3 -m t800_sim.cli teleop --box wrist  # 23: 木块挂右手腕
    python3 -m t800_sim.cli chase               # 22: 追着木块踢
    python3 -m t800_sim.cli batch --vx 0.3 0.6  # 批量评测速度网格
"""

import argparse

from . import scenes

WRIST_BOX_SIZE = 0.10
WRIST_BODY = "LINK_WRIST_END_R"


def _wrist_box(spec) -> None:
    scenes.add_box(spec, "wood_box", WRIST_BOX_SIZE, parent=WRIST_BODY, collide=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="t800_sim")
    sub = parser.add_subparsers(dest="task", required=True)

    p_teleop = sub.add_parser("teleop", help="WASD 遥控")
    p_teleop.add_argument(
        "--box", choices=("none", "wrist"), default="none",
        help="wrist: 木块绑定在右手手腕",
    )
    sub.add_parser("chase", help="自动追着 10 cm 木块走并踢击")

    p_batch = sub.add_parser("batch", help="批量评测：并行跑速度指令网格并输出 CSV")
    p_batch.add_argument("--vx", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p_batch.add_argument("--vy", type=float, nargs="+", default=[0.0])
    p_batch.add_argument("--yaw", type=float, nargs="+", default=[0.0])
    p_batch.add_argument("--duration", type=float, default=5.0)
    p_batch.add_argument("--workers", type=int, default=4)
    p_batch.add_argument("--out", default="batch_results.csv")

    args = parser.parse_args(argv)

    if args.task == "teleop":
        from .tasks import teleop

        if args.box == "wrist":
            teleop.run([_wrist_box], title="T800 WASD + box on right wrist",
                       box_size=WRIST_BOX_SIZE)
        else:
            teleop.run()
    elif args.task == "chase":
        from .tasks.chase_box import run as run_chase

        run_chase()
    else:
        from .batch import run_grid

        run_grid(args.vx, args.vy, args.yaw, args.duration, args.workers, args.out)


if __name__ == "__main__":
    main()