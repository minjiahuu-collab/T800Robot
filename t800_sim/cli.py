"""命令行入口。

用法:
    python3 -m t800_sim.cli teleop              # 20: WASD 遥控
    python3 -m t800_sim.cli teleop --box wrist  # 23: 木块挂右手腕
    python3 -m t800_sim.cli chase               # 22: 追着木块踢
    python3 -m t800_sim.cli record               # 25: 三路相机逐帧采集
    python3 -m t800_sim.cli record --randomize   # 带域随机化的采集
    python3 -m t800_sim.cli record-grid          # 多指令/多 seed 网格采集
    python3 -m t800_sim.cli validate             # 校验采集资产完整性
    python3 -m t800_sim.cli manifest             # 数据集 manifest + train/val 划分
    python3 -m t800_sim.cli cred-check           # P1 确定性与步长收敛
    python3 -m t800_sim.cli cred-lint            # P2 物理合理性检查
    python3 -m t800_sim.cli rand-sample          # 校验域随机化配置并预览采样
    python3 -m t800_sim.cli batch --vx 0.3 0.6  # 批量评测速度网格
"""

import argparse

from . import scenes

WRIST_BOX_SIZE = 0.10
WRIST_BODY = "LINK_WRIST_END_R"


def _load_rand_plan(args):
    """按命令行开关加载随机化配置；未开启时返回 None。"""
    if not args.randomize:
        return None
    from .randomization import load_plan

    return load_plan(args.rand_config)


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

    p_rec = sub.add_parser("record", help="头部+双腕三路相机逐帧采集落盘")
    p_rec.add_argument("--duration", type=float, default=5.0, help="采集时长 (s)")
    p_rec.add_argument("--fps", type=float, default=30.0, help="采集帧率 (不超过 100)")
    p_rec.add_argument("--width", type=int, default=320)
    p_rec.add_argument("--height", type=int, default=240)
    p_rec.add_argument(
        "--cmd", type=float, nargs=3, default=[0.6, 0.0, 0.0],
        metavar=("VX", "VY", "YAW"), help="采样期间的速度指令",
    )
    p_rec.add_argument(
        "--scene", choices=("empty", "box"), default="box",
        help="box: 场景内放 10 cm 木块作为真值物体",
    )
    p_rec.add_argument("--seed", type=int, default=0, help="随机种子")
    p_rec.add_argument("--episode", type=int, default=0, help="同一 seed 下的 episode 序号")
    p_rec.add_argument("--randomize", action="store_true",
                       help="为每条 episode 采样并应用域随机化")
    p_rec.add_argument("--rand-config", default=None,
                       help="随机化配置（默认 configs/randomization.yaml）")
    p_rec.add_argument("--task", dest="task_name", default="walk",
                       help="任务名（目录分层）")
    p_rec.add_argument(
        "--noise", type=float, nargs=3, default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "YAW"), help="初始位姿高斯噪声标准差",
    )
    p_rec.add_argument("--out", default="recordings", help="输出根目录")

    p_grid = sub.add_parser("record-grid", help="多指令/多 seed 网格采集")
    p_grid.add_argument("--vx", type=float, nargs="+", default=[0.3, 0.6])
    p_grid.add_argument("--vy", type=float, nargs="+", default=[0.0])
    p_grid.add_argument("--yaw", type=float, nargs="+", default=[0.0])
    p_grid.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p_grid.add_argument("--episodes", type=int, default=1)
    p_grid.add_argument("--duration", type=float, default=5.0)
    p_grid.add_argument("--fps", type=float, default=30.0)
    p_grid.add_argument("--width", type=int, default=320)
    p_grid.add_argument("--height", type=int, default=240)
    p_grid.add_argument("--scene", choices=("empty", "box"), default="box")
    p_grid.add_argument(
        "--noise", type=float, nargs=3, default=[0.1, 0.05, 0.1],
        metavar=("X", "Y", "YAW"),
    )
    p_grid.add_argument("--randomize", action="store_true",
                        help="为每条 episode 采样并应用域随机化")
    p_grid.add_argument("--rand-config", default=None,
                        help="随机化配置（默认 configs/randomization.yaml）")
    p_grid.add_argument("--out", default="recordings")

    p_val = sub.add_parser("validate", help="校验采集 session 的完整性")
    p_val.add_argument(
        "--session", default="recordings",
        help="session 目录，或含多个 session 的根目录（递归查找）",
    )

    p_man = sub.add_parser("manifest", help="汇总数据集 manifest 并划分 train/val")
    p_man.add_argument("--root", default="recordings", help="数据集根目录")
    p_man.add_argument("--out", default=None, help="manifest CSV（默认 <root>/manifest.csv）")
    p_man.add_argument("--split-out", default=None, help="划分汇总 JSON（默认 <root>/split.json）")
    p_man.add_argument("--val-ratio", type=float, default=0.3, help="验证集比例")
    p_man.add_argument(
        "--split-by", choices=("seed", "session"), default="seed",
        help="seed: 按初始条件分组隔离；session: 按路径哈希",
    )
    p_man.add_argument("--seed", type=int, default=0, help="划分随机种子")

    p_cred = sub.add_parser("cred-check", help="P1 确定性复跑与步长收敛检查")
    p_cred.add_argument("--duration", type=float, default=2.0)
    p_cred.add_argument(
        "--cmd", type=float, nargs=3, default=[0.6, 0.0, 0.0],
        metavar=("VX", "VY", "YAW"),
    )
    p_cred.add_argument("--scene", choices=("empty", "box"), default="box")
    p_cred.add_argument("--seed", type=int, default=0)
    p_cred.add_argument("--repeats", type=int, default=2, help="复跑次数（含基准）")
    p_cred.add_argument("--timestep-ratio", type=float, default=0.5,
                        help="收敛检查的步长比例")
    p_cred.add_argument("--metric-tol", type=float, default=0.02,
                        help="dx/mean_vx 允许的相对漂移")
    p_cred.add_argument("--out", default=None, help="报告 JSON 输出路径")

    p_lint = sub.add_parser("cred-lint", help="P2 物理合理性检查")
    p_lint.add_argument("--session", default="recordings", help="session 或根目录")
    p_lint.add_argument("--deep", type=int, default=0,
                        help="对前 N 条 session 重仿真做接触检查")
    p_lint.add_argument("--out", default=None, help="报告 JSON 输出路径")

    p_tier = sub.add_parser("cred-tier", help="P5 可信度分级 (A/B/C/D)")
    p_tier.add_argument("--root", default="recordings", help="数据集根目录")
    p_tier.add_argument("--deep", type=int, default=0,
                         help="对前 N 条 session 跑 deep rollout")
    p_tier.add_argument("--real-robot", action="store_true",
                        help="已有真机对比报告（外部有效性得满分）")
    p_tier.add_argument("--no-physics-calibrated", action="store_true",
                        help="物理参数未据真机标定（外部有效性扣分）")
    p_tier.add_argument("--repeat-ok", action="store_true",
                        help="同环境已做过双跑一致性检查并通过")
    p_tier.add_argument("--out", default=None, help="完整 JSON 报告输出路径")

    p_bf = sub.add_parser("cred-backfill",
                          help="给现有 session 回填 environment 指纹")
    p_bf.add_argument("--root", default="recordings")

    p_sens = sub.add_parser("cred-sensitivity",
                            help="P3 参数敏感性分析")
    p_sens.add_argument("--dim", nargs="+",
                        choices=("mass", "ground_friction", "robot_friction",
                                 "stiffness", "gravity"),
                        default=None,
                        help="要扫描的维度（默认全扫）")
    p_sens.add_argument("--vx", type=float, nargs="+",
                        default=[0.3, 0.6, 0.8, 0.5],
                        help="指令 vx 取值")
    p_sens.add_argument("--vy", type=float, nargs="+",
                        default=[0.0, 0.0, 0.0, 0.5],
                        help="指令 vy 取值（与 --vx 一一对齐）")
    p_sens.add_argument("--yaw", type=float, nargs="+", default=None)
    p_sens.add_argument("--seeds", type=int, nargs="+", default=[0])
    p_sens.add_argument("--duration", type=float, default=5.0)
    p_sens.add_argument("--out", default=None, help="CSV 输出路径（可选）")

    p_rand = sub.add_parser("rand-sample", help="校验域随机化配置并预览采样")
    p_rand.add_argument("--config", default=None,
                        help="配置文件（默认 configs/randomization.yaml）")
    p_rand.add_argument("--sessions", nargs="+", default=["default"],
                        help="用于派生种子的 session 名")
    p_rand.add_argument("--episodes", type=int, default=3,
                        help="每个 session 采样条数（0 表示只做校验）")
    p_rand.add_argument("--bodies", type=int, default=None,
                        help="逐 body 参数的 body 数量（不填则按单向量预览）")
    p_rand.add_argument("--out", default=None, help="样本 JSON 输出路径")

    p_batch = sub.add_parser("batch", help="批量评测：并行跑速度指令网格并输出 CSV")
    p_batch.add_argument("--vx", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p_batch.add_argument("--vy", type=float, nargs="+", default=[0.0])
    p_batch.add_argument("--yaw", type=float, nargs="+", default=[0.0])
    p_batch.add_argument("--duration", type=float, default=5.0)
    p_batch.add_argument("--workers", type=int, default=4)
    p_batch.add_argument("--randomize", action="store_true",
                         help="每条 trial 采样并应用域随机化")
    p_batch.add_argument("--rand-config", default=None,
                         help="随机化配置（默认 configs/randomization.yaml）")
    p_batch.add_argument("--episodes", type=int, default=1,
                         help="每条指令的随机化 trial 数（仅 --randomize 生效）")
    p_batch.add_argument("--seed", type=int, default=0,
                         help="batch 随机化 session 的种子")
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
    elif args.task == "record":
        from .tasks.record import run as run_record

        run_record(
            args.duration, args.fps, args.width, args.height, args.cmd, args.out,
            scene=args.scene, seed=args.seed, episode=args.episode,
            task_name=args.task_name, init_noise=args.noise,
            rand_plan=_load_rand_plan(args),
        )
    elif args.task == "record-grid":
        from .tasks.record import run_grid

        run_grid(
            args.vx, args.vy, args.yaw, args.seeds, args.episodes,
            args.duration, args.fps, args.width, args.height,
            scene=args.scene, init_noise=args.noise, out_root=args.out,
            rand_plan=_load_rand_plan(args),
        )
    elif args.task == "validate":
        from .validate import main as run_validate

        code = run_validate(args.session)
        if code:
            raise SystemExit(code)
    elif args.task == "manifest":
        from .dataset import build_manifest

        build_manifest(
            args.root, val_ratio=args.val_ratio, split_by=args.split_by,
            seed=args.seed, out=args.out, split_out=args.split_out,
        )
    elif args.task == "cred-check":
        from .credibility.determinism import main as run_cred_check

        code = run_cred_check(
            args.duration, args.cmd, args.scene, args.seed, args.repeats,
            args.timestep_ratio, args.metric_tol, args.out,
        )
        if code:
            raise SystemExit(code)
    elif args.task == "cred-lint":
        from .credibility.physics_lint import main as run_cred_lint

        code = run_cred_lint(args.session, args.deep, args.out)
        if code:
            raise SystemExit(code)
    elif args.task == "cred-tier":
        from .credibility.tier import main as run_cred_tier

        code = run_cred_tier(
            args.root, deep=args.deep, repeat_ok=args.repeat_ok,
            real_robot_ok=args.real_robot,
            physics_calibrated=not args.no_physics_calibrated,
            out=args.out,
        )
        if code:
            raise SystemExit(code)
    elif args.task == "cred-backfill":
        from .credibility.backfill import backfill
        backfill(args.root)
    elif args.task == "cred-sensitivity":
        from .credibility.sensitivity import main as run_sens
        yaw = args.yaw or [0.0] * len(args.vx)
        cmds = list(zip(args.vx, args.vy, yaw))
        code = run_sens(
            dims=args.dim, cmds=cmds, seeds=args.seeds,
            duration=args.duration, out=args.out,
        )
        if code:
            raise SystemExit(code)
    elif args.task == "rand-sample":
        from .randomization.sampler import main as run_rand_sample

        code = run_rand_sample(
            config=args.config, sessions=args.sessions,
            episodes=args.episodes, out=args.out, bodies=args.bodies,
        )
        if code:
            raise SystemExit(code)
    else:
        from .batch import run_grid

        run_grid(
            args.vx, args.vy, args.yaw, args.duration, args.workers, args.out,
            rand_plan=_load_rand_plan(args),
            episodes=args.episodes, seed=args.seed,
        )


if __name__ == "__main__":
    main()