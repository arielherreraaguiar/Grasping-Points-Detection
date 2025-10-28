# newton_baxter_bridge.py
# ROS 2 bridge: FollowJointTrajectory action server + /joint_states publisher for MoveIt.
# It drives the Newton model by tracking the incoming trajectory with a simple PD loop.

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState

import numpy as np
import time

# Import your Newton example class (the combined Baxter+cloth script)
# Make sure this import path is correct in your workspace.
from your_pkg.baxter_cloth_combined import BaxterClothExample  # <-- adjust

class NewtonBaxterBridge(Node):
    def __init__(self, viewer=None, joint_names=None, rate_hz=100.0):
        super().__init__('newton_baxter_bridge')

        # --- Load your Newton simulation (without running a viewer loop here) ---
        self.example = BaxterClothExample(viewer=None)  # run headless for bridge
        self.model = self.example.model
        self.state = self.example.state_0
        self.rate_hz = rate_hz
        self.dt = 1.0 / self.rate_hz

        # Joint mapping (MoveIt joint order -> Newton joint indices)
        if joint_names is None:
            # You should pass the exact MoveIt joint order you will plan for.
            # Fallback: use all model joints in current order.
            joint_names = [self.model.joint_name[i] for i in range(self.model.joint_count)]
        self.joint_names = list(joint_names)

        # Map joint names to Newton DOF indices
        self.name_to_index = {}
        for i, name in enumerate(self.model.joint_name):
            self.name_to_index[name] = i

        self.indices = []
        for name in self.joint_names:
            if name not in self.name_to_index:
                raise RuntimeError(f"Joint name '{name}' not found in Newton model.")
            self.indices.append(self.name_to_index[name])
        self.indices = np.array(self.indices, dtype=np.int32)

        # PD gains
        self.kp = 40.0
        self.kd = 4.0

        # State targets
        self.q_target = self.model.joint_q.numpy().copy()
        self.qd_target = np.zeros_like(self.q_target)

        # Joint state publisher
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)

        # Action server
        self.server = ActionServer(
            self,
            FollowJointTrajectory,
            '/baxter_arm_controller/follow_joint_trajectory',  # match in moveit_controllers.yaml
            execute_callback=self.execute_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb
        )

        # Main timer loop
        self.timer = self.create_timer(self.dt, self.update_loop)

        self.get_logger().info('Newton-Baxter bridge started.')

    # ---- Action server callbacks ----

    def goal_cb(self, goal_request):
        # Validate joints
        recv_names = list(goal_request.goal.trajectory.joint_names)
        if recv_names != self.joint_names:
            # You can relax this check: reorder if names are a permutation
            self.get_logger().warn('Joint name list differs; attempting to proceed.')
        return GoalResponse.ACCEPT

    def cancel_cb(self, goal_handle):
        return CancelResponse.ACCEPT

    async def execute_cb(self, goal_handle):
        traj = goal_handle.request.goal.trajectory
        if len(traj.points) == 0:
            goal_handle.abort()
            return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.INVALID_GOAL)

        # Build a time-parameterized array of targets in our index order
        times = []
        Qs = []
        for pt in traj.points:
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            times.append(t)
            q = np.array(pt.positions, dtype=np.float32)
            if len(q) != len(self.joint_names):
                goal_handle.abort()
                return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.INVALID_GOAL)
            Qs.append(q)
        times = np.array(times, dtype=np.float64)
        Qs = np.vstack(Qs)

        t0 = time.time()
        ok = True
        # Track with PD until final time
        while rclpy.ok():
            t = time.time() - t0
            if t >= times[-1]:
                # Set final point
                self.set_partial_targets(Qs[-1])
                break
            # Linear segment interpolation
            i = np.searchsorted(times, t)
            i0 = max(0, i - 1)
            i1 = min(len(times) - 1, i)
            if times[i1] == times[i0]:
                q_ref = Qs[i1]
            else:
                alpha = (t - times[i0]) / (times[i1] - times[i0])
                q_ref = (1.0 - alpha) * Qs[i0] + alpha * Qs[i1]
            self.set_partial_targets(q_ref)
            await rclpy.sleep(0.0)  # yield control

        goal_handle.succeed()
        return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.SUCCESSFUL)

    # ---- Helpers ----

    def set_partial_targets(self, q_ref):
        """Set targets only for the planned joints; keep others unchanged."""
        q = self.q_target.copy()
        q[self.indices] = q_ref
        self.q_target = q

    def update_loop(self):
        """Run one Newton step and publish /joint_states."""
        # Current state from Newton
        q_now = self.model.joint_q.numpy()
        qd_now = self.model.joint_qd.numpy()

        # PD towards q_target
        err = self.q_target - q_now
        qd_cmd = self.kp * err - self.kd * qd_now
        # Limit command magnitude if needed
        qd_cmd = np.clip(qd_cmd, -1.0, 1.0)

        # Write velocities into the sim input and step once
        self.example.state_0.joint_qd.assign(qd_cmd.astype(np.float32))
        self.example.robot_solver.step(
            self.example.state_0, self.example.state_1, self.example.control, None, self.example.sim_dt
        )
        self.example.state_0, self.example.state_1 = self.example.state_1, self.example.state_0

        # Publish /joint_states
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [self.model.joint_name[i] for i in range(self.model.joint_count)]
        msg.position = list(self.model.joint_q.numpy())
        msg.velocity = list(self.model.joint_qd.numpy())
        self.js_pub.publish(msg)


def main():
    rclpy.init()
    node = NewtonBaxterBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
