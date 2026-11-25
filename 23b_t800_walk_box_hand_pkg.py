"""23 的包化版本：木块绑定在右手手腕，WASD 遥控。"""

from t800_sim import scenes
from t800_sim.tasks import teleop

BOX_SIZE = 0.10
BOX_BODY = "LINK_WRIST_END_R"
BOX_OFFSET = (0.0, 0.0, 0.0)


def attach_box(spec) -> None:
    scenes.add_box(
        spec, "wood_box", BOX_SIZE, pos=BOX_OFFSET, parent=BOX_BODY, collide=False
    )


if __name__ == "__main__":
    teleop.run([attach_box], title="T800 WASD + box on right wrist", box_size=BOX_SIZE)