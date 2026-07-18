#!/usr/bin/env python3
"""Isaac/USD adapters for conservative SE(2) collision proxy extraction.

Isaac modules are imported inside functions so this file remains syntax- and
unit-testable on judge hosts without Isaac Sim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

try:
    from .se2_route_validator import PrismProxy, convex_hull
except ImportError:  # Direct execution inside the submission container.
    from se2_route_validator import PrismProxy, convex_hull


@dataclass(frozen=True)
class CollisionProxySet:
    robot: tuple[PrismProxy, ...]
    environment: tuple[PrismProxy, ...]
    support_surface_paths: tuple[str, ...]
    candidate_robot_paths: tuple[str, ...]
    candidate_environment_paths: tuple[str, ...]
    unresolved_paths: tuple[str, ...]
    traversal_backend: str

    @property
    def complete(self) -> bool:
        return bool(
            self.robot
            and self.environment
            and self.support_surface_paths
            and not self.unresolved_paths
            and len(self.robot) == len(self.candidate_robot_paths)
            and len(self.environment) + len(self.support_surface_paths)
            == len(self.candidate_environment_paths)
        )

    def record(self) -> dict[str, Any]:
        return {
            "classification": (
                "conservative_collision_bound_proxy_inventory"
            ),
            "complete": self.complete,
            "traversal_backend": self.traversal_backend,
            "robot_proxy_count": len(self.robot),
            "environment_proxy_count": len(self.environment),
            "candidate_robot_collision_prim_count": len(
                self.candidate_robot_paths
            ),
            "candidate_environment_collision_prim_count": len(
                self.candidate_environment_paths
            ),
            "support_surface_paths": list(self.support_surface_paths),
            "unresolved_paths": list(self.unresolved_paths),
            "robot_proxies": [_proxy_record(proxy) for proxy in self.robot],
            "environment_proxies": [
                _proxy_record(proxy) for proxy in self.environment
            ],
            "limitation": (
                "Each enabled authored collider is conservatively represented "
                "by the convex hull of its transformed local bound corners. "
                "This may reject a geometrically valid route and is not an "
                "organizer-provided collision contract."
            ),
        }


def extract_collision_proxy_set(
    stage: Any,
    *,
    robot_path: str,
    base_frame_path: str,
    base_world_z: float,
    support_z_tolerance: float = 0.035,
    support_min_area: float = 20.0,
) -> CollisionProxySet:
    """Extract all enabled collision prims, failing closed on any omission."""

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if not (
        math.isfinite(base_world_z)
        and math.isfinite(support_z_tolerance)
        and support_z_tolerance > 0.0
        and math.isfinite(support_min_area)
        and support_min_area > 0.0
    ):
        raise ValueError("collision proxy extraction limits must be finite")

    base_prim = stage.GetPrimAtPath(base_frame_path)
    if not base_prim or not base_prim.IsValid():
        raise RuntimeError(f"invalid base frame prim: {base_frame_path}")
    purposes = [
        UsdGeom.Tokens.default_,
        UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy,
    ]
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), purposes, useExtentsHint=True
    )
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    base_world = xform_cache.GetLocalToWorldTransform(base_prim)
    world_base = base_world.GetInverse()

    try:
        prims: Iterable[Any] = Usd.PrimRange.Stage(
            stage, Usd.TraverseInstanceProxies()
        )
        traversal_backend = "Usd.PrimRange.Stage+TraverseInstanceProxies"
    except Exception:  # noqa: BLE001
        prims = stage.Traverse()
        traversal_backend = "stage.Traverse"
    all_prims: list[Any] = []
    seen: set[str] = set()
    for prim in prims:
        path = str(prim.GetPath())
        if path not in seen:
            seen.add(path)
            all_prims.append(prim)

    robot_proxies: list[PrismProxy] = []
    environment_proxies: list[PrismProxy] = []
    support_paths: list[str] = []
    candidate_robot_paths: list[str] = []
    candidate_environment_paths: list[str] = []
    unresolved: list[str] = []
    base_translation = base_world.ExtractTranslation()
    base_xy = (float(base_translation[0]), float(base_translation[1]))

    candidate_prims: dict[str, Any] = {}
    for prim in all_prims:
        path = str(prim.GetPath())
        collision_attr = prim.GetAttribute("physics:collisionEnabled")
        has_api = prim.HasAPI(UsdPhysics.CollisionAPI)
        if not (
            has_api
            or (collision_attr and collision_attr.IsValid())
        ):
            continue
        if (
            collision_attr
            and collision_attr.IsValid()
            and collision_attr.Get() is False
        ):
            continue
        under_robot = (
            path == robot_path or path.startswith(f"{robot_path}/")
        )
        candidates = (
            candidate_robot_paths
            if under_robot
            else candidate_environment_paths
        )
        candidates.append(path)
        candidate_prims[path] = prim

    candidate_paths_by_specificity = sorted(
        candidate_prims, key=lambda value: (-len(value), value)
    )
    geometry_by_candidate: dict[str, list[Any]] = {
        path: [] for path in candidate_prims
    }
    for prim in all_prims:
        if not prim.IsA(UsdGeom.Boundable):
            continue
        path = str(prim.GetPath())
        owner = next(
            (
                candidate
                for candidate in candidate_paths_by_specificity
                if path == candidate
                or path.startswith(f"{candidate}/")
            ),
            None,
        )
        if owner is not None:
            geometry_by_candidate[owner].append(prim)

    for path in sorted(candidate_prims):
        under_robot = (
            path == robot_path or path.startswith(f"{robot_path}/")
        )
        try:
            geometry_prims = geometry_by_candidate[path]
            if not geometry_prims:
                raise ValueError(
                    "collision prim has no boundable descendant geometry"
                )
            world_corners: list[Any] = []
            for geometry_prim in geometry_prims:
                local_bound = bbox_cache.ComputeLocalBound(geometry_prim)
                local_range = local_bound.GetRange()
                local_min = local_range.GetMin()
                local_max = local_range.GetMax()
                values = [
                    float(local_min[index]) for index in range(3)
                ] + [float(local_max[index]) for index in range(3)]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(
                        "non-finite descendant collision bound at "
                        f"{geometry_prim.GetPath()}"
                    )
                if any(
                    float(local_min[index]) > float(local_max[index])
                    for index in range(3)
                ):
                    raise ValueError(
                        "inverted descendant collision bound at "
                        f"{geometry_prim.GetPath()}"
                    )
                local_bound_matrix = local_bound.GetMatrix()
                prim_world = xform_cache.GetLocalToWorldTransform(
                    geometry_prim
                )
                for corner in _box_corners(local_min, local_max, Gf):
                    prim_point = local_bound_matrix.Transform(corner)
                    world_corners.append(prim_world.Transform(prim_point))
            if not world_corners:
                raise ValueError("collision geometry produced no bound corners")
            world_points = [
                (float(point[0]), float(point[1]))
                for point in world_corners
            ]
            world_z = [float(point[2]) for point in world_corners]
            if under_robot:
                base_corners = [
                    world_base.Transform(point) for point in world_corners
                ]
                proxy = PrismProxy(
                    vertices_xy=convex_hull(
                        (float(point[0]), float(point[1]))
                        for point in base_corners
                    ),
                    z_min=min(float(point[2]) for point in base_corners),
                    z_max=max(float(point[2]) for point in base_corners),
                    source_path=path,
                )
                proxy.validate()
                robot_proxies.append(proxy)
                continue

            proxy = PrismProxy(
                vertices_xy=convex_hull(world_points),
                z_min=min(world_z),
                z_max=max(world_z),
                source_path=path,
            )
            proxy.validate()
            if _is_support_surface(
                proxy,
                base_xy=base_xy,
                base_world_z=base_world_z,
                z_tolerance=support_z_tolerance,
                minimum_area=support_min_area,
            ):
                support_paths.append(path)
            else:
                environment_proxies.append(proxy)
        except Exception as error:  # noqa: BLE001
            unresolved.append(
                f"{path} [{type(error).__name__}: {error}]"
            )

    return CollisionProxySet(
        robot=tuple(sorted(robot_proxies, key=lambda item: item.source_path)),
        environment=tuple(
            sorted(environment_proxies, key=lambda item: item.source_path)
        ),
        support_surface_paths=tuple(sorted(support_paths)),
        candidate_robot_paths=tuple(sorted(candidate_robot_paths)),
        candidate_environment_paths=tuple(
            sorted(candidate_environment_paths)
        ),
        unresolved_paths=tuple(sorted(unresolved)),
        traversal_backend=traversal_backend,
    )


def _box_corners(local_min: Any, local_max: Any, gf: Any) -> Iterable[Any]:
    for x in (float(local_min[0]), float(local_max[0])):
        for y in (float(local_min[1]), float(local_max[1])):
            for z in (float(local_min[2]), float(local_max[2])):
                yield gf.Vec3d(x, y, z)


def _is_support_surface(
    proxy: PrismProxy,
    *,
    base_xy: tuple[float, float],
    base_world_z: float,
    z_tolerance: float,
    minimum_area: float,
) -> bool:
    if not (
        proxy.z_min <= base_world_z + z_tolerance
        and proxy.z_max <= base_world_z + z_tolerance
    ):
        return False
    if abs(_polygon_area(proxy.vertices_xy)) < minimum_area:
        return False
    return _point_in_convex_polygon(base_xy, proxy.vertices_xy)


def _polygon_area(vertices: tuple[tuple[float, float], ...]) -> float:
    return 0.5 * sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(vertices, (*vertices[1:], vertices[0]))
    )


def _point_in_convex_polygon(
    point: tuple[float, float],
    vertices: tuple[tuple[float, float], ...],
) -> bool:
    signs = []
    for start, end in zip(vertices, (*vertices[1:], vertices[0])):
        cross = (
            (end[0] - start[0]) * (point[1] - start[1])
            - (end[1] - start[1]) * (point[0] - start[0])
        )
        if abs(cross) > 1e-12:
            signs.append(math.copysign(1.0, cross))
    return not signs or min(signs) == max(signs)


def _proxy_record(proxy: PrismProxy) -> dict[str, Any]:
    return {
        "source_path": proxy.source_path,
        "vertices_xy": [
            [round(point[0], 9), round(point[1], 9)]
            for point in proxy.vertices_xy
        ],
        "z_min": round(proxy.z_min, 9),
        "z_max": round(proxy.z_max, 9),
    }
