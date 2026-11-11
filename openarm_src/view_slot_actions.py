"""Play back OpenArm slot actions with the complete URDF visual meshes."""

from __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _vector(value: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not value:
        return default
    values = tuple(float(part) for part in value.split())
    if len(values) != 3:
        raise ValueError(f"Expected three values, got {value!r}")
    return values


def _matmul(a: tuple[tuple[float, ...], ...], b: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _rotate(axis: tuple[float, float, float], angle: float) -> tuple[tuple[float, ...], ...]:
    length = math.sqrt(sum(component * component for component in axis)) or 1.0
    x, y, z = (component / length for component in axis)
    c, s, one_minus_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return (
        (c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s),
        (y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s),
        (z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c),
    )


def _rpy(rpy: tuple[float, float, float]) -> tuple[tuple[float, ...], ...]:
    roll, pitch, yaw = rpy
    rx = _rotate((1, 0, 0), roll)
    ry = _rotate((0, 1, 0), pitch)
    rz = _rotate((0, 0, 1), yaw)
    return _matmul(_matmul(rz, ry), rx)


def _transform_point(
    rotation: tuple[tuple[float, ...], ...], translation: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(sum(rotation[i][j] * translation[j] for j in range(3)) for i in range(3))


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    joint_type: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    visual_link: str


class UrdfSkeleton:
    """Minimal URDF kinematics for the OpenArm left-arm chain."""

    def __init__(self, urdf_path: Path, side: str = "left") -> None:
        root = ET.parse(urdf_path).getroot()
        joints: dict[str, Joint] = {}
        for element in root.findall("joint"):
            origin = element.find("origin")
            axis = element.find("axis")
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                continue
            joint = Joint(
                name=element.attrib["name"],
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                joint_type=element.attrib.get("type", "fixed"),
                xyz=_vector(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
                rpy=_vector(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
                axis=_vector(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
                visual_link=child.attrib["link"],
            )
            joints[joint.name] = joint

        wanted = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        selected = [joints[name] for name in wanted if name in joints]
        if len(selected) != 7:
            raise ValueError(f"Could not find all seven {side} arm joints in {urdf_path}")

        by_child = {joint.child: joint for joint in joints.values()}
        chain: list[Joint] = []
        link = selected[0].parent
        while link in by_child:
            joint = by_child[link]
            chain.append(joint)
            link = joint.parent
        chain.reverse()
        chain.extend(selected)
        self.chain = chain
        self.joints = joints
        self.urdf_path = urdf_path
        self._link_joints = {joint.child: joint for joint in joints.values()}

    def link_transforms(
        self, pose: dict[str, float]
    ) -> dict[str, tuple[tuple[float, ...], ...]]:
        """Return world transforms for every link in the URDF tree."""
        transforms: dict[str, tuple[tuple[float, ...], ...]] = {
            "world": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        }
        positions: dict[str, tuple[float, float, float]] = {"world": (0.0, 0.0, 0.0)}
        pending = list(self.joints.values())
        while pending:
            progressed = False
            for joint in pending[:]:
                if joint.parent not in transforms:
                    continue
                parent_rotation = transforms[joint.parent]
                parent_position = positions[joint.parent]
                rotation = _matmul(parent_rotation, _rpy(joint.rpy))
                if joint.joint_type != "fixed":
                    suffix = joint.name.rsplit("joint", 1)[-1]
                    key = f"left_joint_{suffix}.pos"
                    rotation = _matmul(rotation, _rotate(joint.axis, math.radians(float(pose.get(key, 0.0)))))
                offset = _transform_point(parent_rotation, joint.xyz)
                positions[joint.child] = tuple(parent_position[i] + offset[i] for i in range(3))
                transforms[joint.child] = rotation
                pending.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError("URDF joint tree is disconnected from world")
        return transforms, positions

    def points(self, pose: dict[str, float]) -> list[tuple[float, float, float]]:
        rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        position = (0.0, 0.0, 0.0)
        points = [position]
        for joint in self.chain:
            position = tuple(position[i] + _transform_point(rotation, joint.xyz)[i] for i in range(3))
            rotation = _matmul(rotation, _rpy(joint.rpy))
            if joint.joint_type != "fixed":
                key = f"left_joint_{joint.name.rsplit('joint', 1)[-1]}.pos"
                rotation = _matmul(rotation, _rotate(joint.axis, math.radians(float(pose.get(key, 0.0)))))
            points.append(position)
        return points


def _interpolate(start: dict[str, float], target: dict[str, float], frames: int) -> list[dict[str, float]]:
    keys = set(start) | set(target)
    result: list[dict[str, float]] = []
    for frame in range(1, max(frames, 1) + 1):
        alpha = frame / max(frames, 1)
        smooth = alpha * alpha * alpha * (alpha * (alpha * 6 - 15) + 10)
        result.append({key: float(start.get(key, target.get(key, 0.0))) + smooth * (
            float(target.get(key, start.get(key, 0.0))) - float(start.get(key, target.get(key, 0.0)))
        ) for key in keys})
    return result


def build_animation(data: dict[str, Any], selected_slot: str, fps: int) -> list[tuple[str, dict[str, float]]]:
    slots = data["slots"]
    slot_ids = sorted(slots, key=int) if selected_slot == "all" else [selected_slot]
    current = dict(slots[slot_ids[0]]["approach"])
    animation: list[tuple[str, dict[str, float]]] = []
    durations = (("approach", 1.2), ("place", 0.8), ("release", 0.4), ("retract", 0.8))
    for slot_id in slot_ids:
        for phase, duration in durations:
            target = dict(slots[slot_id][phase])
            if phase == "release":
                target = {**current, **target}
            frames = _interpolate(current, target, round(duration * fps))
            animation.extend((f"slot {slot_id} · {phase}", frame) for frame in frames)
            current = frames[-1]
    return animation


def _mesh_path(filename: str, urdf_path: Path) -> Path:
    if filename.startswith("package://openarm_description/"):
        package_root = next(parent for parent in urdf_path.parents if parent.name == "openarm_description")
        return package_root / filename.removeprefix("package://openarm_description/")
    return (urdf_path.parent / filename).resolve()


def _load_mesh_scene(skeleton: UrdfSkeleton) -> tuple[Any, dict[str, list[tuple[str, Any]]]]:
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError(
            "Mesh rendering requires optional dependencies. Install with: "
            "uv sync --extra openarm-viz"
        ) from error

    root = ET.parse(skeleton.urdf_path).getroot()
    scene = trimesh.Scene()
    link_geometry: dict[str, list[tuple[str, Any]]] = {}
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        for index, visual in enumerate(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None or "filename" not in mesh.attrib:
                continue
            path = _mesh_path(mesh.attrib["filename"], skeleton.urdf_path)
            if not path.is_file():
                raise FileNotFoundError(f"URDF mesh not found: {path}")
            loaded = trimesh.load_mesh(path, process=False)
            geometries = loaded.geometry.values() if isinstance(loaded, trimesh.Scene) else (loaded,)
            origin = visual.find("origin")
            origin_matrix = trimesh.transformations.compose_matrix(
                translate=_vector(origin.attrib.get("xyz") if origin is not None else None, (0, 0, 0)),
                angles=_vector(origin.attrib.get("rpy") if origin is not None else None, (0, 0, 0)),
            )
            scale = _vector(mesh.attrib.get("scale"), (1, 1, 1))
            scale_matrix = trimesh.transformations.scale_and_translate(scale=scale)
            for geometry_index, geometry in enumerate(geometries):
                name = f"{link_name}_{index}_{geometry_index}"
                local_transform = origin_matrix @ scale_matrix
                scene.add_geometry(geometry, node_name=name, geom_name=name, transform=local_transform)
                link_geometry.setdefault(link_name, []).append((name, local_transform))
    if not scene.geometry:
        raise ValueError("No visual mesh geometry was found in the URDF")
    return scene, link_geometry


class MeshViewer:
    def __init__(self, skeleton: UrdfSkeleton, animation: list[tuple[str, dict[str, float]]], fps: int) -> None:
        self.skeleton, self.animation, self.fps = skeleton, animation, fps
        self.index, self.playing, self.speed = 0, True, 1.0
        self.scene, self.link_geometry = _load_mesh_scene(skeleton)

    def update(self, _scene: Any = None) -> None:
        label, pose = self.animation[self.index]
        transforms, positions = self.skeleton.link_transforms(pose)
        for link, geometries in self.link_geometry.items():
            if link not in transforms:
                continue
            world = transforms[link]
            translation = positions[link]
            matrix = (
                (world[0][0], world[0][1], world[0][2], translation[0]),
                (world[1][0], world[1][1], world[1][2], translation[1]),
                (world[2][0], world[2][1], world[2][2], translation[2]),
                (0.0, 0.0, 0.0, 1.0),
            )
            for geometry, local_transform in geometries:
                self.scene.graph.update(
                    frame_from="world",
                    frame_to=geometry,
                    matrix=matrix @ local_transform,
                )
        if self.playing:
            self.index = (self.index + 1) % len(self.animation)
        print(f"\r{label} | frame {self.index + 1}/{len(self.animation)} | speed {self.speed:.1f}x", end="", flush=True)

    def run(self) -> None:
        try:
            self.scene.show(
                viewer="gl",
                callback=self.update,
                callback_period=max(1 / self.fps, 0.001),
                smooth=False,
            )
        except ImportError as error:
            raise RuntimeError(
                "OpenGL rendering requires pyglet<2, which is installed by: "
                "uv sync --extra openarm-viz"
            ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=Path(
        os.environ.get("OPENARM_URDF", "~/ros2_ws/src/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf")
    ))
    parser.add_argument("--actions", type=Path, default=Path("openarm_src/tasks/teabag_4slots/tray_slots.json"))
    parser.add_argument("--slot", choices=("all", "1", "2", "3", "4"), default="all")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    urdf_path, actions_path = args.urdf.expanduser().resolve(), args.actions.expanduser().resolve()
    if not urdf_path.is_file():
        parser.error(f"URDF not found: {urdf_path}")
    if not actions_path.is_file():
        parser.error(f"Action JSON not found: {actions_path}")
    with actions_path.open(encoding="utf-8") as file:
        data = json.load(file)
    animation = build_animation(data, args.slot, args.fps)
    MeshViewer(UrdfSkeleton(urdf_path), animation, args.fps).run()


if __name__ == "__main__":
    main()
