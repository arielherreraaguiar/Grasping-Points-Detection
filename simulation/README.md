# 🧵 Baxter + Cloth (Newton)

This folder (`simulation/`) contains a **Newton-based simulation** of a Baxter robot manipulating a rectangular cloth on a table, plus an optional **ROS 2 + MoveIt 2** bridge to plan and execute trajectories with MoveIt and see them running in Newton.

---

## 📂 Folder Structure

```
simulation/
├── my_cloth_pick_place_2arms.py            # Standalone Newton simulation (Baxter + table + cloth)
├── my_cloth_pick_place_2arms_rosnode.py    # ROS 2 bridge node: MoveIt ↔ Newton
└── moveit_controllers.yaml                 # MoveIt controller config (FollowJointTrajectory)
```

---

## 🧩 Prerequisites

- **OS:** Ubuntu 22.04 (tested)
- **Newton:** install from [Newton GitHub repository](https://github.com/newton-physics/newton)
- **Python dependencies:**
  - `warp`, `pxr` (USD), `numpy`, and the `newton` Python bindings
- **Baxter assets (URDF + meshes):**
  - Download from [OneDrive link](https://uniluxembourg-my.sharepoint.com/:f:/g/personal/0230193007_uni_lu/Eo3DfZm19ClFkh91uz3xZFQBY19fTHO9m0ouJQryuLPQ0Q?e=NbchBf)
- **For MoveIt integration:**
  - ROS 2 Humble
  - MoveIt 2 (Humble)
  - A ROS 2 workspace (`colcon`) to host the bridge and MoveIt configuration

> ⚠️ **Important:** Keep the Baxter URDF identical between Newton and MoveIt (same joint names) to avoid mapping errors.

---

## ▶️ 1) Running the Standalone Newton Simulation

`my_cloth_pick_place_2arms.py` runs Baxter + a rectangular cloth on a table with small sinusoidal motions so you can see the robot moving.

1. Export the URDF path:

```bash
export BAXTER_URDF_PATH=/absolute/path/to/baxter_with_grippers_visual.urdf
```

2. Run the simulation:

```bash
cd simulation
python3 my_cloth_pick_place_2arms.py
```

---

## 🤖 2) Connecting Newton and MoveIt (ROS 2 Bridge)

The file `my_cloth_pick_place_2arms_rosnode.py` connects the Newton simulation with MoveIt:

- Publishes `/joint_states` from Newton
- Provides a `FollowJointTrajectory` action server (used by MoveIt)
- Applies incoming trajectories to the Newton model using a simple PD controller

### 🧱 Create a ROS 2 Workspace

```bash
mkdir -p ~/ws_baxter/src
cd ~/ws_baxter
source /opt/ros/humble/setup.bash
```

Create a package (e.g. `baxter_newton_bridge`) inside `src/` and add:

```
my_cloth_pick_place_2arms.py
my_cloth_pick_place_2arms_rosnode.py
moveit_controllers.yaml
```

Then edit the import line inside `my_cloth_pick_place_2arms_rosnode.py`:

```python
from baxter_newton_bridge.my_cloth_pick_place_2arms import BaxterClothExample
```

Build and source the workspace:

```bash
colcon build
source install/setup.bash
```

---

## 🧠 3) MoveIt Controller Configuration

Inside your MoveIt configuration package, include or reference this file:

```yaml
# moveit_controllers.yaml
moveit_controller_manager: moveit_simple_controller_manager/MoveItSimpleControllerManager

controller_names:
  - baxter_arm_controller

baxter_arm_controller:
  type: FollowJointTrajectory
  joints:
    - left_s0
    - left_s1
    - left_e0
    - left_e1
    - left_w0
    - left_w1
    - left_w2
    - right_s0
    - right_s1
    - right_e0
    - right_e1
    - right_w0
    - right_w1
    - right_w2
```

Make sure the action name matches:
```
/baxter_arm_controller/follow_joint_trajectory
```

---

## 🧵 4) Running the Full Setup

### **Terminal A — Newton Simulation**
```bash
export BAXTER_URDF_PATH=/path/to/baxter_with_grippers_visual.urdf
python3 my_cloth_pick_place_2arms.py
```

### **Terminal B — ROS 2 Bridge**
```bash
source ~/ws_baxter/install/setup.bash
ros2 run baxter_newton_bridge newton_baxter_bridge
```

### **Terminal C — MoveIt 2 + RViz**
```bash
ros2 launch your_baxter_moveit_config demo.launch.py
```

Then in RViz:
1. Add the **MotionPlanning** plugin  
2. Select a planning group (left arm, right arm, or both)  
3. Click **Plan** → **Execute**  
   The Baxter robot in Newton should follow the trajectory!

---

## ⚙️ Notes

- **Frame consistency:** use identical `world` / `base` transforms in `robot_state_publisher` and Newton.
- **Grippers:** avoid mimic joints; use independent left/right finger joints.
- **PD tuning:** adjust `kp` and `kd` in `my_cloth_pick_place_2arms_rosnode.py` if the motion is too slow or unstable.
- **Action name:** must match `/baxter_arm_controller/follow_joint_trajectory`.
- **Cloth stability:** if the cloth deforms or collapses, increase `iterations` or stiffness values in the script.

---

## ✅ Checklist

- [ ] Newton installed and `BAXTER_URDF_PATH` exported  
- [ ] `my_cloth_pick_place_2arms.py` runs and shows Baxter + table + cloth  
- [ ] ROS 2 Humble + MoveIt 2 installed  
- [ ] Bridge builds successfully with `colcon`  
- [ ] `moveit_controllers.yaml` configured correctly  
- [ ] MoveIt plans execute in Newton simulation  

---

🎯 **You’re ready to plan in MoveIt and see Baxter move in Newton!**
