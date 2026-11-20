"""Minimal test script for Damiao motor with ID 3."""

import pytest

from lerobot.utils.import_utils import _can_available

if not _can_available:
    pytest.skip("python-can not available", allow_module_level=True)

from lerobot.motors import Motor
from lerobot.motors.damiao import DamiaoMotorsBus


@pytest.mark.skip(reason="Requires physical Damiao motor and CAN interface")
def test_damiao_motor():
    motors = {
        "joint_3": Motor(
            id=0x03,
            model="damiao",
            norm_mode="degrees",
            motor_type_str="dm4310",
            recv_id=0x13,
        ),
    }

    bus = DamiaoMotorsBus(port="can0", motors=motors)

    try:
        print("Connecting...")
        bus.connect()
        print("✓ Connected")

        print("Enabling torque...")
        bus.enable_torque()
        print("✓ Torque enabled")

        print("Reading all states...")
        states = bus.sync_read_all_states()
        print(f"✓ States: {states}")

        print("Reading position...")
        positions = bus.sync_read("Present_Position")
        print(f"✓ Position: {positions}")

        print("Testing MIT control batch...")
        current_pos = states["joint_3"]["position"]
        commands = {"joint_3": (10.0, 0.5, current_pos, 0.0, 0.0)}
        bus._mit_control_batch(commands)
        print("✓ MIT control batch sent")

        print("Disabling torque...")
        bus.disable_torque()
        print("✓ Torque disabled")

        print("Setting zero position...")
        bus.set_zero_position()
        print("✓ Zero position set")

    finally:
        print("Disconnecting...")
        bus.disconnect(disable_torque=True)
        print("✓ Disconnected")


if __name__ == "__main__":
    test_damiao_motor()


class _FakeCanBus:
    """Fake CAN bus: answers each enable frame with a state frame on the motor's recv_id,
    except the first `drop[id]` frames sent to motor `id`, which go unanswered."""

    def __init__(self, recv_ids: dict[int, int], drop: dict[int, int]):
        import can

        self._can = can
        self.recv_ids = recv_ids
        self.drop = dict(drop)
        self.sent: list[int] = []
        self.queue: list = []

    def send(self, msg):
        self.sent.append(msg.arbitration_id)
        if self.drop.get(msg.arbitration_id, 0) > 0:
            self.drop[msg.arbitration_id] -= 1
            return
        data = [msg.arbitration_id, 0x80, 0x00, 0x80, 0x08, 0x00, 30, 30]
        self.queue.append(self._can.Message(arbitration_id=self.recv_ids[msg.arbitration_id], data=data))

    def recv(self, timeout=None):
        return self.queue.pop(0) if self.queue else None


def _make_bus(n_motors: int = 3) -> DamiaoMotorsBus:
    motors = {
        f"joint_{i}": Motor(
            id=i, model="damiao", norm_mode="degrees", motor_type_str="dm4310", recv_id=0x10 + i
        )
        for i in range(1, n_motors + 1)
    }
    return DamiaoMotorsBus(port="can0", motors=motors)


def test_handshake_retries_motor_that_misses_first_frame():
    bus = _make_bus()
    fake = _FakeCanBus(recv_ids={1: 0x11, 2: 0x12, 3: 0x13}, drop={1: 1})
    bus.canbus = fake

    bus._handshake()

    # joint_1 pinged twice (first frame dropped), the others once
    assert fake.sent == [1, 2, 3, 1]
    # state cache is decoded from the motor's reply, not from the outgoing enable frame
    assert bus._last_known_states["joint_1"]["temp_mos"] == 30.0


def test_handshake_raises_when_motor_never_answers():
    bus = _make_bus()
    fake = _FakeCanBus(recv_ids={1: 0x11, 2: 0x12, 3: 0x13}, drop={2: 99})
    bus.canbus = fake

    with pytest.raises(ConnectionError, match=r"after 3 attempts.*\['joint_2'\]"):
        bus._handshake()
    assert fake.sent == [1, 2, 3, 2, 2]
