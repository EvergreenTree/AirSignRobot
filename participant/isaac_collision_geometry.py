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


Point3 = tuple[float, float, float]
WitnessKey = tuple[str, str]


@dataclass(frozen=True)
class CollisionGeometryWitness:
    """Immutable evidence binding one collider to one Boundable descendant."""

    collider_path: str
    boundable_path: str
    local_corners: tuple[Point3, ...]
    captured_corners: tuple[Point3, ...]
    captured_frame: str

    @property
    def key(self) -> WitnessKey:
        return self.collider_path, self.boundable_path

    def validate(self) -> None:
        for label, path in (
            ("collider", self.collider_path),
            ("Boundable", self.boundable_path),
        ):
            if (
                not isinstance(path, str)
                or not path.startswith("/")
                or path == "/"
            ):
                raise ValueError(f"{label} witness path must be absolute")
        if not _path_is_within_any(
            self.boundable_path, (self.collider_path,)
        ):
            raise ValueError(
                "Boundable witness path is not within its collider path"
            )
        if self.captured_frame not in {"base", "world"}:
            raise ValueError("witness frame must be 'base' or 'world'")
        if (
            not self.local_corners
            or len(self.local_corners) != len(self.captured_corners)
        ):
            raise ValueError(
                "witness local and captured corners must be non-empty and match"
            )
        for corner in (*self.local_corners, *self.captured_corners):
            if len(corner) != 3 or not all(
                math.isfinite(float(value)) for value in corner
            ):
                raise ValueError("witness corners must be finite 3-D points")

    def record(self) -> dict[str, Any]:
        return {
            "collider_path": self.collider_path,
            "boundable_path": self.boundable_path,
            "captured_frame": self.captured_frame,
            "local_corners": [list(corner) for corner in self.local_corners],
            "captured_corners": [
                list(corner) for corner in self.captured_corners
            ],
        }


@dataclass(frozen=True)
class CollisionProxySet:
    robot: tuple[PrismProxy, ...]
    environment: tuple[PrismProxy, ...]
    support_surface_paths: tuple[str, ...]
    candidate_robot_paths: tuple[str, ...]
    candidate_environment_paths: tuple[str, ...]
    unresolved_paths: tuple[str, ...]
    traversal_backend: str
    attached_payload_root_paths: tuple[str, ...] = ()
    robot_geometry_witnesses: tuple[CollisionGeometryWitness, ...] = ()
    environment_geometry_witnesses: tuple[
        CollisionGeometryWitness, ...
    ] = ()
    support_surface_proxies: tuple[PrismProxy, ...] = ()

    @property
    def attached_payload_proxy_paths(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                proxy.source_path
                for proxy in self.robot
                if _path_is_within_any(
                    proxy.source_path,
                    self.attached_payload_root_paths,
                )
            )
        )

    @property
    def missing_attached_payload_root_paths(self) -> tuple[str, ...]:
        represented = self.attached_payload_proxy_paths
        return tuple(
            root
            for root in self.attached_payload_root_paths
            if not any(
                path == root or path.startswith(f"{root}/")
                for path in represented
            )
        )

    @property
    def complete(self) -> bool:
        robot_proxy_paths = tuple(
            sorted(proxy.source_path for proxy in self.robot)
        )
        environment_accounted_paths = tuple(
            sorted(
                [
                    proxy.source_path
                    for proxy in self.environment
                ]
                + list(self.support_surface_paths)
            )
        )
        return bool(
            self.robot
            and self.environment
            and self.support_surface_paths
            and not self.unresolved_paths
            and not self.missing_attached_payload_root_paths
            and robot_proxy_paths
            == tuple(sorted(self.candidate_robot_paths))
            and environment_accounted_paths
            == tuple(sorted(self.candidate_environment_paths))
            and tuple(
                sorted(
                    proxy.source_path
                    for proxy in self.support_surface_proxies
                )
            )
            == tuple(sorted(self.support_surface_paths))
            and _witness_inventory_complete(
                self.robot_geometry_witnesses,
                candidate_paths=self.candidate_robot_paths,
                expected_frame="base",
            )
            and _witness_inventory_complete(
                self.environment_geometry_witnesses,
                candidate_paths=self.candidate_environment_paths,
                expected_frame="world",
            )
        )

    def record(self) -> dict[str, Any]:
        return {
            "classification": (
                "conservative_collision_bound_proxy_inventory"
            ),
            "complete": self.complete,
            "traversal_backend": self.traversal_backend,
            "attached_payload_root_paths": list(
                self.attached_payload_root_paths
            ),
            "attached_payload_proxy_paths": list(
                self.attached_payload_proxy_paths
            ),
            "attached_payload_proxy_count": len(
                self.attached_payload_proxy_paths
            ),
            "missing_attached_payload_root_paths": list(
                self.missing_attached_payload_root_paths
            ),
            "robot_proxy_count": len(self.robot),
            "environment_proxy_count": len(self.environment),
            "candidate_robot_collision_prim_count": len(
                self.candidate_robot_paths
            ),
            "candidate_environment_collision_prim_count": len(
                self.candidate_environment_paths
            ),
            "support_surface_paths": list(self.support_surface_paths),
            "support_surface_proxies": [
                _proxy_record(proxy)
                for proxy in self.support_surface_proxies
            ],
            "unresolved_paths": list(self.unresolved_paths),
            "robot_geometry_witness_count": len(
                self.robot_geometry_witnesses
            ),
            "environment_geometry_witness_count": len(
                self.environment_geometry_witnesses
            ),
            "robot_geometry_witnesses": [
                witness.record()
                for witness in self.robot_geometry_witnesses
            ],
            "environment_geometry_witnesses": [
                witness.record()
                for witness in self.environment_geometry_witnesses
            ],
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


def _witness_inventory_complete(
    witnesses: Iterable[CollisionGeometryWitness],
    *,
    candidate_paths: Iterable[str],
    expected_frame: str,
) -> bool:
    items = tuple(witnesses)
    if not items:
        return False
    try:
        for witness in items:
            witness.validate()
    except Exception:  # noqa: BLE001 - completeness is a boolean gate
        return False
    keys = tuple(witness.key for witness in items)
    return bool(
        len(keys) == len(set(keys))
        and all(witness.captured_frame == expected_frame for witness in items)
        and set(witness.collider_path for witness in items)
        == set(candidate_paths)
    )


def _collect_collision_inventory(
    stage: Any,
    *,
    robot_path: str,
    attached_root_paths: tuple[str, ...],
    usd: Any,
    usd_geom: Any,
    usd_physics: Any,
) -> dict[str, Any]:
    """Collect enabled collider owners and their most-specific geometry."""

    try:
        prims: Iterable[Any] = usd.PrimRange.Stage(
            stage, usd.TraverseInstanceProxies()
        )
        traversal_backend = "Usd.PrimRange.Stage+TraverseInstanceProxies"
    except Exception:  # noqa: BLE001 - retain non-instance fallback
        prims = stage.Traverse()
        traversal_backend = "stage.Traverse"
    all_prims: list[Any] = []
    seen: set[str] = set()
    for prim in prims:
        path = str(prim.GetPath())
        if path not in seen:
            seen.add(path)
            all_prims.append(prim)

    candidate_prims: dict[str, Any] = {}
    candidate_robot_paths: list[str] = []
    candidate_environment_paths: list[str] = []
    for prim in all_prims:
        path = str(prim.GetPath())
        collision_attr = prim.GetAttribute("physics:collisionEnabled")
        has_api = prim.HasAPI(usd_physics.CollisionAPI)
        if not (
            has_api or (collision_attr and collision_attr.IsValid())
        ):
            continue
        if (
            collision_attr
            and collision_attr.IsValid()
            and collision_attr.Get() is False
        ):
            continue
        under_robot = (
            _path_is_within_any(path, (robot_path,))
            or _path_is_within_any(path, attached_root_paths)
        )
        (
            candidate_robot_paths
            if under_robot
            else candidate_environment_paths
        ).append(path)
        candidate_prims[path] = prim

    candidate_paths_by_specificity = sorted(
        candidate_prims, key=lambda value: (-len(value), value)
    )
    geometry_by_candidate: dict[str, list[Any]] = {
        path: [] for path in candidate_prims
    }
    for prim in all_prims:
        if not prim.IsA(usd_geom.Boundable):
            continue
        path = str(prim.GetPath())
        owner = next(
            (
                candidate
                for candidate in candidate_paths_by_specificity
                if _path_is_within_any(path, (candidate,))
            ),
            None,
        )
        if owner is not None:
            geometry_by_candidate[owner].append(prim)

    return {
        "candidate_prims": candidate_prims,
        "candidate_robot_paths": tuple(sorted(candidate_robot_paths)),
        "candidate_environment_paths": tuple(
            sorted(candidate_environment_paths)
        ),
        "geometry_by_candidate": {
            path: tuple(
                sorted(
                    geometry,
                    key=lambda prim: str(prim.GetPath()),
                )
            )
            for path, geometry in geometry_by_candidate.items()
        },
        "traversal_backend": traversal_backend,
    }


def extract_collision_proxy_set(
    stage: Any,
    *,
    robot_path: str,
    base_frame_path: str,
    base_world_z: float,
    attached_payload_root_paths: Iterable[str] = (),
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
    attached_root_paths = tuple(
        sorted(set(attached_payload_root_paths))
    )
    if any(
        not isinstance(path, str)
        or not path.startswith("/")
        or path == "/"
        for path in attached_root_paths
    ):
        raise ValueError("attached paths must be absolute non-root USD paths")

    base_prim = stage.GetPrimAtPath(base_frame_path)
    if not base_prim or not base_prim.IsValid():
        raise RuntimeError(f"invalid base frame prim: {base_frame_path}")
    purposes = [
        UsdGeom.Tokens.default_,
        UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy,
        # Isaac robot collision meshes are commonly authored as guide-purpose
        # geometry. Omitting guide returns an empty GfRange for those prims.
        UsdGeom.Tokens.guide,
    ]
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), purposes, useExtentsHint=True
    )
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    base_world = xform_cache.GetLocalToWorldTransform(base_prim)
    world_base = base_world.GetInverse()

    inventory = _collect_collision_inventory(
        stage,
        robot_path=robot_path,
        attached_root_paths=attached_root_paths,
        usd=Usd,
        usd_geom=UsdGeom,
        usd_physics=UsdPhysics,
    )
    candidate_prims = inventory["candidate_prims"]
    candidate_robot_paths = inventory["candidate_robot_paths"]
    candidate_environment_paths = inventory[
        "candidate_environment_paths"
    ]
    geometry_by_candidate = inventory["geometry_by_candidate"]
    traversal_backend = inventory["traversal_backend"]
    robot_proxies: list[PrismProxy] = []
    environment_proxies: list[PrismProxy] = []
    robot_witnesses: list[CollisionGeometryWitness] = []
    environment_witnesses: list[CollisionGeometryWitness] = []
    support_paths: list[str] = []
    support_proxies: list[PrismProxy] = []
    unresolved: list[str] = []
    base_translation = base_world.ExtractTranslation()
    base_xy = (float(base_translation[0]), float(base_translation[1]))

    for path in sorted(candidate_prims):
        under_robot = (
            _path_is_within_any(path, (robot_path,))
            or _path_is_within_any(path, attached_root_paths)
        )
        try:
            geometry_prims = geometry_by_candidate[path]
            if not geometry_prims:
                raise ValueError(
                    "collision prim has no boundable descendant geometry"
                )
            world_corners: list[Any] = []
            candidate_witnesses: list[CollisionGeometryWitness] = []
            for geometry_prim in geometry_prims:
                prim_corners = _local_geometry_corners(
                    geometry_prim,
                    bbox_cache=bbox_cache,
                    usd=Usd,
                    usd_geom=UsdGeom,
                    gf=Gf,
                )
                prim_world = xform_cache.GetLocalToWorldTransform(
                    geometry_prim
                )
                geometry_world_corners = tuple(
                    prim_world.Transform(corner)
                    for corner in prim_corners
                )
                world_corners.extend(geometry_world_corners)
                captured_corners = (
                    tuple(
                        world_base.Transform(point)
                        for point in geometry_world_corners
                    )
                    if under_robot
                    else geometry_world_corners
                )
                witness = CollisionGeometryWitness(
                    collider_path=path,
                    boundable_path=str(geometry_prim.GetPath()),
                    local_corners=tuple(
                        _point3_tuple(corner) for corner in prim_corners
                    ),
                    captured_corners=tuple(
                        _point3_tuple(corner)
                        for corner in captured_corners
                    ),
                    captured_frame="base" if under_robot else "world",
                )
                witness.validate()
                candidate_witnesses.append(witness)
            if not world_corners:
                raise ValueError(
                    "collision geometry produced no bound corners"
                )
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
                robot_witnesses.extend(candidate_witnesses)
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
                support_proxies.append(proxy)
            else:
                environment_proxies.append(proxy)
            environment_witnesses.extend(candidate_witnesses)
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
        attached_payload_root_paths=attached_root_paths,
        robot_geometry_witnesses=tuple(
            sorted(robot_witnesses, key=lambda witness: witness.key)
        ),
        environment_geometry_witnesses=tuple(
            sorted(environment_witnesses, key=lambda witness: witness.key)
        ),
        support_surface_proxies=tuple(
            sorted(support_proxies, key=lambda item: item.source_path)
        ),
    )


def collision_geometry_containment_record(
    captured_proxies: Iterable[PrismProxy],
    current_proxies: Iterable[PrismProxy],
    *,
    captured_witness_keys: Iterable[WitnessKey],
    current_witness_keys: Iterable[WitnessKey],
    dynamic_allowance: float,
) -> dict[str, Any]:
    """Check exact inventory and current-proxy containment without Isaac."""

    base: dict[str, Any] = {
        "classification": "live_collision_geometry_containment_gate",
        "passed": False,
        "failure": "invalid_containment_inputs",
        "reason": None,
        "dynamic_allowance_metres": None,
        "inventory_exact": False,
        "captured_proxy_paths": [],
        "current_proxy_paths": [],
        "missing_proxy_paths": [],
        "unexpected_proxy_paths": [],
        "captured_witness_keys": [],
        "current_witness_keys": [],
        "missing_witness_keys": [],
        "unexpected_witness_keys": [],
        "maximum_planar_escape_metres": None,
        "maximum_z_escape_metres": None,
        "proxy_containment": [],
    }
    try:
        allowance = float(dynamic_allowance)
        if not math.isfinite(allowance) or allowance < 0.0:
            raise ValueError(
                "dynamic geometry allowance must be finite and non-negative"
            )
        captured_by_path = _unique_proxy_mapping(captured_proxies)
        current_by_path = _unique_proxy_mapping(current_proxies)
        captured_keys = _normalize_witness_keys(captured_witness_keys)
        current_keys = _normalize_witness_keys(current_witness_keys)
        if not captured_by_path:
            raise ValueError("captured robot proxy inventory is empty")
        if not captured_keys:
            raise ValueError("captured geometry witness inventory is empty")
    except Exception as error:  # noqa: BLE001 - fail-closed pure record
        return {
            **base,
            "reason": f"{type(error).__name__}: {error}",
        }

    captured_paths = tuple(sorted(captured_by_path))
    current_paths = tuple(sorted(current_by_path))
    missing_paths = tuple(sorted(set(captured_paths) - set(current_paths)))
    unexpected_paths = tuple(
        sorted(set(current_paths) - set(captured_paths))
    )
    missing_keys = tuple(sorted(set(captured_keys) - set(current_keys)))
    unexpected_keys = tuple(sorted(set(current_keys) - set(captured_keys)))
    inventory_exact = not (
        missing_paths or unexpected_paths or missing_keys or unexpected_keys
    )
    record = {
        **base,
        "dynamic_allowance_metres": allowance,
        "inventory_exact": inventory_exact,
        "captured_proxy_paths": list(captured_paths),
        "current_proxy_paths": list(current_paths),
        "missing_proxy_paths": list(missing_paths),
        "unexpected_proxy_paths": list(unexpected_paths),
        "captured_witness_keys": [list(key) for key in captured_keys],
        "current_witness_keys": [list(key) for key in current_keys],
        "missing_witness_keys": [list(key) for key in missing_keys],
        "unexpected_witness_keys": [list(key) for key in unexpected_keys],
    }
    if not inventory_exact:
        return {
            **record,
            "failure": "collision_geometry_inventory_mismatch",
            "reason": (
                "live collider/proxy inventory differs from the certified "
                "capture"
            ),
        }

    containment: list[dict[str, Any]] = []
    maximum_planar_escape = 0.0
    maximum_z_escape = 0.0
    for path in captured_paths:
        captured = captured_by_path[path]
        current = current_by_path[path]
        captured_hull = convex_hull(captured.vertices_xy)
        planar_escape = max(
            _distance_to_convex_polygon(point, captured_hull)
            for point in current.vertices_xy
        )
        z_escape = max(
            0.0,
            float(captured.z_min) - float(current.z_min),
            float(current.z_max) - float(captured.z_max),
        )
        maximum_planar_escape = max(
            maximum_planar_escape, planar_escape
        )
        maximum_z_escape = max(maximum_z_escape, z_escape)
        containment.append(
            {
                "source_path": path,
                "planar_escape_metres": planar_escape,
                "z_escape_metres": z_escape,
                "passed": bool(
                    (
                        planar_escape <= allowance
                        or math.isclose(
                            planar_escape,
                            allowance,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    )
                    and (
                        z_escape <= allowance
                        or math.isclose(
                            z_escape,
                            allowance,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    )
                ),
            }
        )

    passed = all(item["passed"] for item in containment)
    return {
        **record,
        "passed": passed,
        "failure": (
            None if passed else "dynamic_geometry_envelope_violation"
        ),
        "reason": (
            None
            if passed
            else "live collision geometry left its certified inflated proxy"
        ),
        "maximum_planar_escape_metres": maximum_planar_escape,
        "maximum_z_escape_metres": maximum_z_escape,
        "proxy_containment": containment,
    }


def live_collision_geometry_containment_record(
    stage: Any,
    captured: CollisionProxySet,
    *,
    robot_path: str,
    base_frame_path: str,
    dynamic_allowance: float,
) -> dict[str, Any]:
    """Recapture robot and payload witnesses in the current robot base frame."""

    common = {
        "classification": "live_collision_geometry_containment_gate",
        "passed": False,
        "failure": "invalid_live_geometry_capture",
        "reason": None,
        "dynamic_allowance_metres": dynamic_allowance,
        "capture_traversal_backend": captured.traversal_backend,
        "live_traversal_backend": None,
        "unresolved_live_geometry": [],
    }
    try:
        allowance = float(dynamic_allowance)
        if not math.isfinite(allowance) or allowance < 0.0:
            raise ValueError(
                "dynamic geometry allowance must be finite and non-negative"
            )
        if not captured.complete:
            raise ValueError("captured collision proxy set is incomplete")
        for label, path in (
            ("robot", robot_path),
            ("base frame", base_frame_path),
        ):
            if (
                not isinstance(path, str)
                or not path.startswith("/")
                or path == "/"
            ):
                raise ValueError(f"{label} path must be absolute")
    except Exception as error:  # noqa: BLE001 - reject before pxr import
        return {
            **common,
            "reason": f"{type(error).__name__}: {error}",
        }

    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        base_prim = stage.GetPrimAtPath(base_frame_path)
        if not base_prim or not base_prim.IsValid():
            raise RuntimeError(f"invalid base frame prim: {base_frame_path}")
        inventory = _collect_collision_inventory(
            stage,
            robot_path=robot_path,
            attached_root_paths=captured.attached_payload_root_paths,
            usd=Usd,
            usd_geom=UsdGeom,
            usd_physics=UsdPhysics,
        )
        purposes = [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ]
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), purposes, useExtentsHint=True
        )
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        world_base = xform_cache.GetLocalToWorldTransform(
            base_prim
        ).GetInverse()
        captured_robot_witness_by_key = {
            witness.key: witness
            for witness in captured.robot_geometry_witnesses
        }
        captured_environment_witness_by_key = {
            witness.key: witness
            for witness in captured.environment_geometry_witnesses
        }
        current_robot_witness_keys = tuple(
            sorted(
                (
                    collider_path,
                    str(geometry_prim.GetPath()),
                )
                for collider_path in inventory[
                    "candidate_robot_paths"
                ]
                for geometry_prim in inventory[
                    "geometry_by_candidate"
                ][collider_path]
            )
        )
        current_environment_witness_keys = tuple(
            sorted(
                (
                    collider_path,
                    str(geometry_prim.GetPath()),
                )
                for collider_path in inventory[
                    "candidate_environment_paths"
                ]
                for geometry_prim in inventory[
                    "geometry_by_candidate"
                ][collider_path]
            )
        )
        current_robot_proxies: list[PrismProxy] = []
        current_environment_proxies: list[PrismProxy] = []
        unresolved: list[str] = []
        for collider_path in inventory["candidate_robot_paths"]:
            base_corners: list[Any] = []
            geometry_prims = inventory["geometry_by_candidate"][
                collider_path
            ]
            if not geometry_prims:
                unresolved.append(
                    f"{collider_path} [no Boundable descendants]"
                )
                continue
            for geometry_prim in geometry_prims:
                key = collider_path, str(geometry_prim.GetPath())
                witness = captured_robot_witness_by_key.get(key)
                if witness is None:
                    unresolved.append(
                        f"{collider_path} [unexpected witness {key[1]}]"
                    )
                    continue
                live_local = tuple(
                    _point3_tuple(corner)
                    for corner in _local_geometry_corners(
                        geometry_prim,
                        bbox_cache=bbox_cache,
                        usd=Usd,
                        usd_geom=UsdGeom,
                        gf=Gf,
                    )
                )
                if live_local != witness.local_corners:
                    unresolved.append(
                        f"{collider_path} [local geometry changed at {key[1]}]"
                    )
                    continue
                prim_world = xform_cache.GetLocalToWorldTransform(
                    geometry_prim
                )
                base_corners.extend(
                    world_base.Transform(
                        prim_world.Transform(Gf.Vec3d(*corner))
                    )
                    for corner in witness.local_corners
                )
            if not base_corners:
                continue
            proxy = PrismProxy(
                vertices_xy=convex_hull(
                    (float(point[0]), float(point[1]))
                    for point in base_corners
                ),
                z_min=min(float(point[2]) for point in base_corners),
                z_max=max(float(point[2]) for point in base_corners),
                source_path=collider_path,
            )
            proxy.validate()
            current_robot_proxies.append(proxy)

        for collider_path in inventory["candidate_environment_paths"]:
            world_corners: list[Any] = []
            geometry_prims = inventory["geometry_by_candidate"][
                collider_path
            ]
            if not geometry_prims:
                unresolved.append(
                    f"{collider_path} [no Boundable descendants]"
                )
                continue
            for geometry_prim in geometry_prims:
                key = collider_path, str(geometry_prim.GetPath())
                witness = captured_environment_witness_by_key.get(key)
                if witness is None:
                    unresolved.append(
                        f"{collider_path} [unexpected witness {key[1]}]"
                    )
                    continue
                live_local = tuple(
                    _point3_tuple(corner)
                    for corner in _local_geometry_corners(
                        geometry_prim,
                        bbox_cache=bbox_cache,
                        usd=Usd,
                        usd_geom=UsdGeom,
                        gf=Gf,
                    )
                )
                if live_local != witness.local_corners:
                    unresolved.append(
                        f"{collider_path} [local geometry changed at {key[1]}]"
                    )
                    continue
                prim_world = xform_cache.GetLocalToWorldTransform(
                    geometry_prim
                )
                world_corners.extend(
                    prim_world.Transform(Gf.Vec3d(*corner))
                    for corner in witness.local_corners
                )
            if not world_corners:
                continue
            proxy = PrismProxy(
                vertices_xy=convex_hull(
                    (float(point[0]), float(point[1]))
                    for point in world_corners
                ),
                z_min=min(float(point[2]) for point in world_corners),
                z_max=max(float(point[2]) for point in world_corners),
                source_path=collider_path,
            )
            proxy.validate()
            current_environment_proxies.append(proxy)

        robot_containment = collision_geometry_containment_record(
            captured.robot,
            current_robot_proxies,
            captured_witness_keys=tuple(
                witness.key
                for witness in captured.robot_geometry_witnesses
            ),
            current_witness_keys=current_robot_witness_keys,
            dynamic_allowance=allowance,
        )
        environment_containment = collision_geometry_containment_record(
            (
                *captured.environment,
                *captured.support_surface_proxies,
            ),
            current_environment_proxies,
            captured_witness_keys=tuple(
                witness.key
                for witness in captured.environment_geometry_witnesses
            ),
            current_witness_keys=current_environment_witness_keys,
            dynamic_allowance=allowance,
        )
        robot_planar = robot_containment.get(
            "maximum_planar_escape_metres"
        )
        robot_z = robot_containment.get("maximum_z_escape_metres")
        environment_planar = environment_containment.get(
            "maximum_planar_escape_metres"
        )
        environment_z = environment_containment.get(
            "maximum_z_escape_metres"
        )
        numeric_escapes = (
            robot_planar,
            robot_z,
            environment_planar,
            environment_z,
        )
        escapes_valid = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in numeric_escapes
        )
        relative_planar_escape = (
            float(robot_planar) + float(environment_planar)
            if escapes_valid
            else None
        )
        relative_z_escape = (
            float(robot_z) + float(environment_z)
            if escapes_valid
            else None
        )
        combined_within_allowance = bool(
            escapes_valid
            and max(relative_planar_escape, relative_z_escape)
            <= allowance + 1e-12
        )
        result = {
            **common,
            "passed": bool(
                robot_containment.get("passed", False)
                and environment_containment.get("passed", False)
                and combined_within_allowance
            ),
            "failure": None,
            "reason": None,
            "inventory_exact": bool(
                robot_containment.get("inventory_exact", False)
                and environment_containment.get("inventory_exact", False)
            ),
            "maximum_planar_escape_metres": relative_planar_escape,
            "maximum_z_escape_metres": relative_z_escape,
            "robot_geometry": robot_containment,
            "environment_geometry": environment_containment,
            "capture_traversal_backend": captured.traversal_backend,
            "live_traversal_backend": inventory["traversal_backend"],
            "unresolved_live_geometry": sorted(unresolved),
            "current_robot_proxies": [
                _proxy_record(proxy)
                for proxy in sorted(
                    current_robot_proxies,
                    key=lambda item: item.source_path,
                )
            ],
            "current_environment_proxies": [
                _proxy_record(proxy)
                for proxy in sorted(
                    current_environment_proxies,
                    key=lambda item: item.source_path,
                )
            ],
        }
        if unresolved:
            result.update(
                {
                    "passed": False,
                    "failure": "live_geometry_unresolved",
                    "reason": (
                        "one or more live collision witnesses could not be "
                        "recaptured exactly"
                    ),
                }
            )
        elif not bool(robot_containment.get("passed", False)):
            result.update(
                {
                    "passed": False,
                    "failure": robot_containment.get("failure"),
                    "reason": robot_containment.get("reason"),
                }
            )
        elif not bool(environment_containment.get("passed", False)):
            result.update(
                {
                    "passed": False,
                    "failure": environment_containment.get("failure"),
                    "reason": (
                        "live environment/support collision geometry changed "
                        "outside its captured allowance: "
                        + str(environment_containment.get("reason"))
                    ),
                }
            )
        elif not combined_within_allowance:
            result.update(
                {
                    "passed": False,
                    "failure": "combined_dynamic_geometry_violation",
                    "reason": (
                        "summed robot/payload and environment/support escape "
                        "exceeded the single certified dynamic allowance"
                    ),
                }
            )
        return result
    except Exception as error:  # noqa: BLE001 - fail-closed runtime record
        return {
            **common,
            "reason": f"{type(error).__name__}: {error}",
        }


def enabled_dynamic_rigid_body_record(
    asset_root_path: str,
    bodies: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Pure fail-closed selection of exactly one enabled dynamic rigid body."""

    base = {
        "classification": "unique_enabled_dynamic_rigid_body_resolver",
        "asset_root_path": asset_root_path,
        "passed": False,
        "reason": None,
        "dynamic_rigid_body_path": None,
        "rigid_body_count": 0,
        "enabled_dynamic_rigid_body_count": 0,
        "bodies": [],
    }
    try:
        if (
            not isinstance(asset_root_path, str)
            or not asset_root_path.startswith("/")
            or asset_root_path == "/"
        ):
            raise ValueError("asset root path must be absolute and non-root")
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for supplied in bodies:
            path = str(supplied["prim_path"])
            if not _path_is_within_any(path, (asset_root_path,)):
                raise ValueError(
                    "rigid-body path is outside the requested asset root"
                )
            if path in seen:
                raise ValueError("rigid-body paths must be unique")
            seen.add(path)
            enabled = supplied["rigid_body_enabled"]
            kinematic = supplied["kinematic_enabled"]
            if not isinstance(enabled, bool) or not isinstance(
                kinematic, bool
            ):
                raise ValueError(
                    "rigid-body enabled/kinematic fields must be booleans"
                )
            normalized.append(
                {
                    "prim_path": path,
                    "rigid_body_enabled": enabled,
                    "kinematic_enabled": kinematic,
                }
            )
    except Exception as error:  # noqa: BLE001 - pure fail-closed record
        return {
            **base,
            "reason": f"{type(error).__name__}: {error}",
        }
    normalized.sort(key=lambda record: record["prim_path"])
    dynamic = [
        record
        for record in normalized
        if record["rigid_body_enabled"]
        and not record["kinematic_enabled"]
    ]
    record = {
        **base,
        "rigid_body_count": len(normalized),
        "enabled_dynamic_rigid_body_count": len(dynamic),
        "bodies": normalized,
    }
    if len(dynamic) != 1:
        return {
            **record,
            "reason": (
                "asset root must contain exactly one enabled dynamic rigid "
                f"body; found {len(dynamic)}"
            ),
        }
    return {
        **record,
        "passed": True,
        "reason": None,
        "dynamic_rigid_body_path": dynamic[0]["prim_path"],
    }


def resolve_enabled_dynamic_rigid_body_descendant(
    stage: Any,
    asset_root_path: str,
) -> dict[str, Any]:
    """Resolve one live dynamic body below an asset root using lazy pxr."""

    if (
        not isinstance(asset_root_path, str)
        or not asset_root_path.startswith("/")
        or asset_root_path == "/"
    ):
        return enabled_dynamic_rigid_body_record(asset_root_path, ())
    try:
        from pxr import Usd, UsdPhysics

        root = stage.GetPrimAtPath(asset_root_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"invalid asset root prim: {asset_root_path}")
        try:
            prims: Iterable[Any] = Usd.PrimRange(
                root, Usd.TraverseInstanceProxies()
            )
        except Exception:  # noqa: BLE001 - retain non-instance fallback
            prims = Usd.PrimRange(root)
        bodies: list[dict[str, Any]] = []
        seen: set[str] = set()
        for prim in prims:
            path = str(prim.GetPath())
            if path in seen:
                continue
            seen.add(path)
            api = UsdPhysics.RigidBodyAPI(prim)
            if not api:
                continue
            enabled_attr = api.GetRigidBodyEnabledAttr()
            kinematic_attr = api.GetKinematicEnabledAttr()
            enabled = (
                bool(enabled_attr.Get())
                if enabled_attr and enabled_attr.HasAuthoredValueOpinion()
                else True
            )
            kinematic = (
                bool(kinematic_attr.Get())
                if kinematic_attr
                and kinematic_attr.HasAuthoredValueOpinion()
                else False
            )
            bodies.append(
                {
                    "prim_path": path,
                    "rigid_body_enabled": enabled,
                    "kinematic_enabled": kinematic,
                }
            )
        return enabled_dynamic_rigid_body_record(
            asset_root_path, bodies
        )
    except Exception as error:  # noqa: BLE001 - explicit failed record
        record = enabled_dynamic_rigid_body_record(asset_root_path, ())
        return {
            **record,
            "reason": f"{type(error).__name__}: {error}",
        }


def _point3_tuple(point: Any) -> Point3:
    values = tuple(float(point[index]) for index in range(3))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("collision witness point is not finite")
    return values


def _unique_proxy_mapping(
    proxies: Iterable[PrismProxy],
) -> dict[str, PrismProxy]:
    result: dict[str, PrismProxy] = {}
    for proxy in proxies:
        proxy.validate()
        if proxy.source_path in result:
            raise ValueError("collision proxy source paths must be unique")
        result[proxy.source_path] = proxy
    return result


def _normalize_witness_keys(
    keys: Iterable[WitnessKey],
) -> tuple[WitnessKey, ...]:
    result: list[WitnessKey] = []
    for supplied in keys:
        if len(supplied) != 2:
            raise ValueError(
                "collision witness keys must contain collider and Boundable"
            )
        key = str(supplied[0]), str(supplied[1])
        if (
            not key[0].startswith("/")
            or not key[1].startswith("/")
            or not _path_is_within_any(key[1], (key[0],))
        ):
            raise ValueError("collision witness key paths are invalid")
        result.append(key)
    if len(result) != len(set(result)):
        raise ValueError("collision witness keys must be unique")
    return tuple(sorted(result))


def _inventory_witness_keys(
    inventory: dict[str, Any],
) -> tuple[WitnessKey, ...]:
    return tuple(
        sorted(
            (
                collider_path,
                str(geometry_prim.GetPath()),
            )
            for collider_path, geometry_prims in inventory[
                "geometry_by_candidate"
            ].items()
            for geometry_prim in geometry_prims
        )
    )


def _path_is_within_any(path: str, roots: Iterable[str]) -> bool:
    return any(
        path == root or path.startswith(f"{root}/")
        for root in roots
    )


def _local_geometry_corners(
    geometry_prim: Any,
    *,
    bbox_cache: Any,
    usd: Any,
    usd_geom: Any,
    gf: Any,
) -> tuple[Any, ...]:
    """Return conservative geometry corners in the prim's object space.

    Authored ``extent`` is the USD contract for a Boundable's local-space
    range, so it is preferred and is unaffected by visibility or purpose.
    When it is absent or invalid, ComputeUntransformedBound derives the
    geometry range while deliberately excluding the prim's own transform.
    The caller applies the prim's local-to-world transform exactly once.
    """

    errors: list[str] = []
    boundable = usd_geom.Boundable(geometry_prim)
    extent_attr = boundable.GetExtentAttr()
    if (
        extent_attr
        and extent_attr.IsValid()
        and extent_attr.HasAuthoredValueOpinion()
    ):
        try:
            extent = extent_attr.Get(usd.TimeCode.Default())
            return tuple(
                gf.Vec3d(*corner)
                for corner in _extent_corner_values(extent)
            )
        except Exception as error:  # noqa: BLE001 - try derived geometry
            errors.append(
                f"authored extent {type(error).__name__}: {error}"
            )

    try:
        local_bound = bbox_cache.ComputeUntransformedBound(geometry_prim)
        local_range = local_bound.GetRange()
        corners = tuple(
            local_bound.GetMatrix().Transform(gf.Vec3d(*corner))
            for corner in _extent_corner_values(
                (local_range.GetMin(), local_range.GetMax())
            )
        )
        if not corners:
            raise ValueError("derived bound produced no corners")
        return corners
    except Exception as error:  # noqa: BLE001 - report every safe source
        errors.append(f"derived bound {type(error).__name__}: {error}")

    detail = "; ".join(errors) if errors else "no bound source was available"
    raise ValueError(
        f"unresolved descendant collision bound at "
        f"{geometry_prim.GetPath()} ({detail})"
    )


def _validated_extent(
    extent: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Normalize a two-corner 3-D extent and reject unsafe values."""

    if extent is None:
        raise ValueError("extent is missing")
    try:
        if len(extent) != 2:
            raise ValueError("extent must contain exactly two corners")
        corners = tuple(
            tuple(float(corner[index]) for index in range(3))
            for corner in extent
        )
    except (IndexError, TypeError) as error:
        raise ValueError(
            "extent corners must each contain three coordinates"
        ) from error
    values = tuple(value for corner in corners for value in corner)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("extent contains a non-finite coordinate")
    local_min, local_max = corners
    if any(
        local_min[index] > local_max[index] for index in range(3)
    ):
        raise ValueError("extent minimum exceeds maximum")
    return local_min, local_max


def _extent_corner_values(
    extent: Any,
) -> tuple[tuple[float, float, float], ...]:
    local_min, local_max = _validated_extent(extent)
    return tuple(
        (x, y, z)
        for x in (local_min[0], local_max[0])
        for y in (local_min[1], local_max[1])
        for z in (local_min[2], local_max[2])
    )


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


def _distance_to_convex_polygon(
    point: tuple[float, float],
    vertices: tuple[tuple[float, float], ...],
) -> float:
    if _point_in_convex_polygon(point, vertices):
        return 0.0
    return min(
        _distance_to_segment(point, start, end)
        for start, end in zip(
            vertices, (*vertices[1:], vertices[0])
        )
    )


def _distance_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-24:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    fraction = min(
        1.0,
        max(
            0.0,
            (
                (point[0] - start[0]) * dx
                + (point[1] - start[1]) * dy
            )
            / denominator,
        ),
    )
    nearest = (
        start[0] + fraction * dx,
        start[1] + fraction * dy,
    )
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1])


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
