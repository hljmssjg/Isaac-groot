# G1 真机控制环路详解

本文档梳理数据采集（xr_teleoperate）、ACT 评估（unitree_IL_lerobot）、GR00T 评估（eval_g1_0330.py）三个流程中，从"目标关节角"到"电机力矩"的完整控制链路。

---

## 1. 整体架构概览

三个流程共享同一套底层硬件接口：

```
┌─────────────────────────────────────────────────────────┐
│  上层决策（不同流程不同来源）                              │
│                                                         │
│  数采: XR控制器 → IK求解器 → (q_target, tauff)           │
│  ACT eval: ACT模型 → action → solve_tau → (q_target, tauff) │
│  GR00T eval: GR00T VLA → action → gravity_comp → (q_target, tauff) │
└──────────────────────────┬──────────────────────────────┘
                           │ ctrl_dual_arm(q_target, tauff)
                           ▼
┌─────────────────────────────────────────────────────────┐
│  G1_29_ArmController（250Hz 内部控制线程）                │
│                                                         │
│  1. 速度裁剪: cliped_q = clip(q_target, velocity_limit) │
│  2. 填充 DDS LowCmd:                                    │
│     motor_cmd[id].q   = cliped_q     （位置目标）        │
│     motor_cmd[id].dq  = 0            （速度目标=0）      │
│     motor_cmd[id].tau = tauff         （前馈力矩）       │
│  3. CRC校验 → DDS发布 LowCmd                            │
└──────────────────────────┬──────────────────────────────┘
                           │ DDS LowCmd @ 250Hz
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Unitree G1 电机控制器（板载固件）                        │
│                                                         │
│  tau_output = kp*(q_target - q_actual)                  │
│             + kd*(0 - dq_actual)                        │
│             + tau_feedforward                            │
│                                                         │
│  这是一个 混合PD+前馈 控制器 (mode=1)                    │
└─────────────────────────────────────────────────────────┘
```

关键点：**电机固件执行的是 PD 位置跟踪 + 前馈力矩**，软件层只负责计算 `q_target` 和 `tauff` 两个量。

---

## 2. 手臂 PD 增益参数

| 关节类型 | kp | kd | 备注 |
|---------|-----|-----|------|
| 肩/肘关节 (ShoulderPitch/Roll/Yaw, Elbow) | 80.0 | 3.0 | 低刚度，柔顺控制 |
| 腕关节 (WristRoll/Pitch/Yaw) | 40.0 | 1.5 | 更低刚度 |
| 非臂关节（腿、腰等） | 300.0 | 3.0 | 高刚度锁定 |

低 kp 意味着仅靠 PD 项不足以抵抗重力，**必须依赖前馈力矩 tauff 来补偿重力**，否则手臂会因重力而下垂。

---

## 3. 前馈力矩 tauff 的计算：RNEA 重力补偿

三个流程都（应该）通过 Pinocchio 的 **RNEA（递归牛顿-欧拉算法）** 计算前馈力矩：

```python
tauff = pinocchio.rnea(model, data, q, v=zeros, a=zeros)
```

参数含义：
- `q`：目标关节角度（14维，左臂7 + 右臂7）
- `v = zeros(14)`：关节速度设为 0
- `a = zeros(14)`：关节加速度设为 0

由逆动力学公式：

```
tau = M(q)*a + C(q,v)*v + g(q)
```

当 v=0, a=0 时，简化为：

```
tau = g(q)   ← 纯重力补偿项
```

也就是说，tauff 就是在目标构型 q 下**抵消重力所需的力矩**。

使用的 Pinocchio 模型是从完整 G1 URDF 中构建的 **reduced model**（锁定腿、腰、手指共 29 个关节，仅保留 14 个手臂关节）。

---

## 4. 三个流程的详细对比

### 4.1 数据采集（xr_teleoperate）

**文件**: `~/xr_teleoperate/teleop/teleop_hand_and_arm.py`

```
XR控制器输入 → 末端执行器目标位姿 (4x4 SE3)
    ↓
G1_29_ArmIK.solve_ik(left_pose, right_pose, current_q, current_dq)
    ├── CasADi 优化问题求解 IK → sol_q (14维关节角)
    ├── 平滑滤波: WeightedMovingFilter
    └── RNEA 重力补偿: sol_tauff = pin.rnea(model, data, sol_q, zeros, zeros)
    ↓
arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)  ← q 和 tauff 都非零
```

手控制：
```
XR 手骨骼数据 → HandRetargeting → left/right_q_target
    ↓
ctrl_dual_hand(left_q, right_q)  ← 纯PD (kp=1.5, kd=0.2, tau=0)
```

数据记录内容：
- `state`: current_arm_q(14) + hand_state(14) = 28 维关节角度
- `action`: sol_q(14) + hand_action(14) = 28 维目标关节角度
- 注意：**记录的 action 是关节角度，不是力矩**

### 4.2 ACT 评估（unitree_IL_lerobot）

**文件**: `~/unitree_IL_lerobot/unitree_lerobot/eval_robot/eval_g1.py`

```
观测 (图像 + 关节状态) → ACT 模型推理 → action_np (28维)
    ↓
arm_action = action_np[:14]           ← 模型直接预测关节角
tau = arm_ik.solve_tau(arm_action)    ← RNEA 重力补偿
arm_ctrl.ctrl_dual_arm(arm_action, tau)
```

`solve_tau` 的实现（`robot_arm_ik.py`）：
```python
def solve_tau(self, current_lr_arm_motor_q):
    return pin.rnea(model, data, current_lr_arm_motor_q, zeros(14), zeros(14))
```

手控制通过共享内存传给独立的手控制进程。

### 4.3 GR00T 评估（eval_g1_0330.py，升级后）

**文件**: `~/Isaac-GR00T/gr00t/eval/real_robot/G1/eval_g1_0330.py`

```
观测 (3路相机 + 28维关节状态) → GR00T VLA 推理服务器 → action_chunk (多步)
    ↓ 逐步执行
arm_action = action[:14]
tauff = gravity_comp.compute(arm_action)   ← RNEA 重力补偿（升级后新增）
arm_ctrl.ctrl_dual_arm(arm_action, tauff)
hand_ctrl.ctrl_dual_hand(left_hand, right_hand)
```

**升级前的问题**: `tauff = np.zeros(14)`，缺少重力补偿前馈，与数采时的控制特性不一致。

---

## 5. 手部控制（Dex3-1 灵巧手）

手部控制在三个流程中基本一致，都是 **纯 PD 位置控制**：

```
motor_cmd.q   = q_target    （位置目标）
motor_cmd.dq  = 0
motor_cmd.tau = 0            （无前馈力矩）
motor_cmd.kp  = 1.5
motor_cmd.kd  = 0.2
```

手指电机较小、重力影响可忽略，所以不需要重力补偿。

通信方式：DDS HandCmd 话题，左右手各一个 publisher：
- `rt/dex3/left/cmd`
- `rt/dex3/right/cmd`

---

## 6. 速度裁剪（安全机制）

`G1_29_ArmController` 内部有速度裁剪逻辑，防止目标位置突变导致危险运动：

```python
def clip_arm_q_target(self, target_q, velocity_limit=20.0):
    current_q = self.get_current_dual_arm_q()
    delta = target_q - current_q
    motion_scale = max(|delta|) / (velocity_limit * control_dt)
    clipped_q = current_q + delta / max(motion_scale, 1.0)
```

- `control_dt = 1/250` (内部控制线程 250Hz)
- `velocity_limit` 默认 20 rad/s，启动后 5 秒内从 20 渐增到 30
- 效果：单步最大移动量 = `velocity_limit * control_dt` ≈ 0.08 rad

---

## 7. 频率总结

| 层级 | 频率 | 说明 |
|------|------|------|
| 策略推理 | ~2-5 Hz | 每次推理返回 action_horizon 步动作 |
| 动作执行（外层循环） | 30 Hz | 逐步发送 action chunk 中的每一步 |
| 电机控制线程 | 250 Hz | G1_29_ArmController 内部，速度裁剪 + DDS 发布 |
| 手控制进程 | 100 Hz | Dex3_1_Controller 的 control_process |
| DDS 状态订阅 | ~500 Hz | lowstate 读取线程 (sleep 0.002s) |

外层 30Hz 循环调用 `ctrl_dual_arm` 只是更新目标值（写入共享变量），真正的 DDS 发布由 250Hz 的内部线程完成，中间会做速度裁剪平滑过渡。

---

## 8. 完整数据流图（以 eval_g1_0330.py 为例）

```
[ImageServer on Robot]                    [GR00T Inference Server on GPU]
  head + wrist + extra                         PolicyClient (ZMQ)
  → 拼接为单张 JPEG                                ↑ obs    ↓ action_chunk
  → ZMQ PUB                                        │        │
       │                                           │        │
       ▼                                           │        │
┌─── eval_g1_0330.py (主控脚本) ──────────────────────────────────────┐
│                                                                     │
│  1. ImageReceiver.receive()  → 拆分为 cam_head/cam_left_wrist/cam_extra │
│  2. robot.get_state()        → 28维关节角 (DDS lowstate)             │
│  3. adapter.get_action(state, images, lang)                         │
│     ├── obs_to_policy_inputs() → 打包为 GR00T 格式 (B=1,T=1)        │
│     ├── policy_client.get_action() → ZMQ 发给推理服务器              │
│     └── decode_action_chunk() → List[28维绝对关节角]                 │
│  4. for action in actions:                                          │
│     ├── arm_action = action[:14]                                    │
│     ├── tauff = gravity_comp.compute(arm_action)  ← RNEA 重力补偿   │
│     ├── arm_ctrl.ctrl_dual_arm(arm_action, tauff) ← 更新目标        │
│     ├── hand_ctrl.ctrl_dual_hand(left, right)     ← 更新手目标      │
│     └── sleep(1/30)  维持30Hz                                       │
│                                                                     │
│  内部线程 (250Hz):                                                   │
│     cliped_q = clip(q_target, velocity_limit)                       │
│     DDS LowCmd → motor_cmd = {q, dq=0, tau=tauff, kp, kd, mode=1}  │
└─────────────────────────────────────────────────────────────────────┘
       │                              │
       ▼ DDS LowCmd                   ▼ DDS HandCmd
  [G1 手臂电机固件]              [Dex3-1 手指电机固件]
  tau = kp*(q_t-q) + kd*(0-dq) + tauff    tau = kp*(q_t-q) + kd*(0-dq)
```
