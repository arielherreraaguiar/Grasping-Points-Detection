# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0
#
# Baxter + Rectangular Cloth (Newton): VBD for cloth, Featherstone for Baxter.
# - Robot: Loads Baxter URDF from BAXTER_URDF_PATH using the SAME transform as the
#   "robot-only" demo (xform z = 0.0). No auto-grounding.
# - Cloth & table: Taken from the cloth_franka_rect example.
# - Control: Small sinusoidal joint velocities for "life" motion.
#
# Usage:
#   export BAXTER_URDF_PATH=/path/to/baxter_with_grippers_visual.urdf
#   python this_script.py
#
# Notes:
#   - If you want finger motion to be more visible, increase amplitudes for those
#     particular DOFs after identifying their indices.

from __future__ import annotations

import os
import numpy as np
import warp as wp
import warp.examples
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.utils
from newton import ModelBuilder
from newton.solvers import SolverFeatherstone, SolverVBD


class BaxterClothExample:
    def __init__(self, viewer):
        # ----- Simulation parameters -----
        self.add_cloth = True
        self.add_robot = True

        self.sim_substeps = 15
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # ----- Contact & materials -----
        self.cloth_particle_radius = 0.008
        self.cloth_body_contact_margin = 0.01
        self.self_contact_radius = 0.002
        self.self_contact_margin = 0.003

        self.soft_contact_ke = 100
        self.soft_contact_kd = 2e-3

        self.robot_friction = 1.0
        self.table_friction = 0.5
        self.self_contact_friction = 0.25

        # Cloth elasticity/stabilization
        self.tri_ke = 1e2
        self.tri_ka = 1e2
        self.tri_kd = 1.5e-6
        self.bending_ke = 1e-4
        self.bending_kd = 1e-3

        # ----- Scene -----
        self.scene = ModelBuilder()
        self.soft_contact_max = 1_000_000
        self.viewer = viewer

        # ----- Baxter (EXACT same transform as your robot-only demo) -----
        if self.add_robot:
            baxter_urdf = os.environ.get(
                "BAXTER_URDF_PATH",
                "/home/ariel/dev/newton/newton_assets/baxter/baxter_common/baxter_description/urdf/baxter_with_grippers_visual.urdf",
            )
            baxter_urdf = (baxter_urdf or "").strip()
            if not baxter_urdf or not os.path.exists(baxter_urdf):
                raise FileNotFoundError(
                    "Baxter URDF not found. Export the path before running:\n"
                    "  export BAXTER_URDF_PATH=/path/to/baxter_common/baxter_description/urdf/baxter_with_grippers_visual.urdf"
                )

            baxter = ModelBuilder()
            baxter.add_urdf(
                baxter_urdf,
                xform=wp.transform(
                    (-0.5, -0.5, 1),   # <<< SAME as your robot-only script
                    wp.quat_identity(),
                ),
                floating=False,
                scale=1.0,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
            )

            # Optional: keep counts if you want to inspect them
            self.bodies_per_world = baxter.body_count
            self.dof_q_per_world = baxter.joint_coord_count
            self.dof_qd_per_world = baxter.joint_dof_count

            self.scene.add_builder(baxter)

        # ----- Table (same as cloth example) -----
        # Box center at z=0.1 with half-height 0.1 -> table touches ground plane at z=0
        self.scene.add_shape_box(
            -1,
            wp.transform(
                wp.vec3(0.35, -0.5, 0.1),
                wp.quat_identity(),
            ),
            hx=0.4,
            hy=0.4,
            hz=0.9,
        )

        # ----- Rectangular cloth (USD) -----
        if self.add_cloth:
            usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
            usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/cloth/cloth"))
            mesh_points = np.array(usd_geom.GetPointsAttr().Get())
            mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
            vertices = [wp.vec3(v) for v in mesh_points]

            cloth_scale = 0.003  # ~15–20 cm
            self.scene.add_cloth_mesh(
                vertices=vertices,
                indices=mesh_indices,
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
                pos=wp.vec3(0.35, -0.50, 1.5),
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=0.2,
                scale=cloth_scale,
                tri_ke=self.tri_ke,
                tri_ka=self.tri_ka,
                tri_kd=self.tri_kd,
                edge_ke=self.bending_ke,
                edge_kd=self.bending_kd,
                particle_radius=self.cloth_particle_radius,
            )
            self.scene.color()

        # ----- Ground plane (z = 0) -----
        self.scene.add_ground_plane()

        # ----- Finalize combined model -----
        self.model = self.scene.finalize(requires_grad=False)
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # ----- Solvers -----
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.cloth_solver: SolverVBD | None = None
        if self.add_cloth:
            # Stabilization tweak from the cloth example
            self.model.edge_rest_angle.zero_()
            self.cloth_solver = SolverVBD(
                self.model,
                iterations=self.iterations,
                self_contact_radius=self.self_contact_radius,
                self_contact_margin=self.self_contact_margin,
                handle_self_contact=True,
                vertex_collision_buffer_pre_alloc=32,
                edge_collision_buffer_pre_alloc=64,
                integrate_with_external_rigid_solver=True,
                collision_detection_interval=-1,
            )

        # ----- Viewer -----
        if self.viewer:
            self.viewer.set_model(self.model)

        # ----- Gravity control: 0 for robot step, earth for cloth step -----
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)

        # ----- Initial FK -----
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ----- "Life" control: small sinusoidal velocities -----
        self.num_dofs = self.model.joint_dof_count
        rng = np.random.default_rng(42)
        base_omega = 0.5
        self.omega = rng.uniform(0.7 * base_omega, 1.3 * base_omega, size=self.num_dofs)
        self.phase = rng.uniform(0, 2 * np.pi, size=self.num_dofs)
        self.amp = np.full(self.num_dofs, 0.08, dtype=np.float32)
        self.target_joint_qd = wp.empty(self.num_dofs, dtype=float)

        # ----- CUDA Graph (optional) -----
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def step(self):
        # Generate sinusoidal joint velocities (same spirit as your robot-only demo)
        t = self.sim_time
        qd = (self.amp * np.sin(self.omega * t + self.phase)).astype(np.float32)
        self.target_joint_qd.assign(qd)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def simulate(self):
        # Coupled substeps: robot (rigid) + cloth (VBD)
        if self.add_cloth and self.cloth_solver is not None:
            self.cloth_solver.rebuild_bvh(self.state_0)

        for _ in range(self.sim_substeps):
            # Clear forces
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # Viewer interactive forces
            if self.viewer:
                self.viewer.apply_forces(self.state_0)

            # ----- Robot step (rigid-only) -----
            if self.add_robot:
                particle_count = self.model.particle_count
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)

                # Avoid shape contacts during rigid integration
                self.model.shape_contact_pair_count = 0
                self.state_0.joint_qd.assign(self.target_joint_qd)

                self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

                # Restore particles and gravity
                self.state_0.particle_f.zero_()
                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            # ----- Cloth step -----
            self.contacts = self.model.collide(self.state_0, soft_contact_margin=self.cloth_body_contact_margin)
            if self.add_cloth and self.cloth_solver is not None:
                self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def render(self):
        if not self.viewer:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=3850)  # ~64 s @ 60 FPS
    viewer, args = newton.examples.init(parser)

    example = BaxterClothExample(viewer)
    newton.examples.run(example, args)
