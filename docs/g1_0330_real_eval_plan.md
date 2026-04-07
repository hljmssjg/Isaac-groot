# G1_0330 微调模型真机评估方案

## 1. 概述

- 微调模型: `outputs/g1_0330_refinetune/checkpoint-4000`
- 基座模型: GR00T N1.6-3B
- 训练数据: `cache/Jiangeng/G1_0330` (30 episodes, 39768 frames, 30 FPS)
- 机器人: Unitree G1 上半身 (28 DOF)
- Embodiment tag: `NEW_EMBODIMENT`

### 架构

```
[GPU 服务器]                                [G1 机器人]
GR00T 推理服务器                             eval_g1_0330.py
(run_gr00t_server.py)  <--- ZMQ REQ/REP --->  (直接 DDS 控制)
      |                                           |
      v                                           v
 checkpoint-4000                       ImageServer(ZMQ PUB) + DDS lowcmd
```

**注意：当前方案不使用 WBC（全身控制）管线。** 因为 G1 坐在凳子上只用上半身，不需要平衡/步态控制，直接通过 DDS 发送关节位置+前馈力矩即可，与数据采集时 xr_teleoperate 的控制方式完全一致。

---

## 2. 控制管线说明

### 2.1 为什么不用 WBC

旧版 plan 中考虑过 WBC 管线（`deploy_g1.py` → ROS topic → `CONTROL_GOAL_TOPIC`），但实际场景是：

- G1 **坐姿**，腿/腰关节锁定，不需要全身平衡
- 数采时用的是 xr_teleoperate 的 **直接 DDS 控制**（`G1_29_ArmController` + `Dex3_1_Controller`）
- eval 时保持**同样的控制栈**可以最大程度减少 sim-to-real gap

因此 `eval_g1_0330.py` 直接复用 xr_teleoperate 的底层控制器，通过 DDS 与电机通信，绕过 WBC。

### 2.2 控制链路

```
GR00T VLA 输出 action (28维绝对关节角)
    │
    ├── arm_action = action[:14]   (左臂7 + 右臂7)
    │     │
    │     ├── RNEA 重力补偿 → tauff = g(q)
    │     └── ctrl_dual_arm(arm_action, tauff)
    │           │
    │           └── 内部 250Hz 控制线程:
    │                 速度裁剪 → DDS LowCmd {q, dq=0, tau=tauff, kp, kd, mode=1}
    │
    └── hand_action = action[14:28]  (左手7 + 右手7)
          │
          └── ctrl_dual_hand(left_hand, right_hand)
                │
                └── DDS HandCmd {q, kp=1.5, kd=0.2, tau=0}
```

电机固件执行: `tau_output = kp*(q_target - q_actual) + kd*(0 - dq_actual) + tau_feedforward`

### 2.3 Modality 配置

| Modality | Key | DOF | Action 类型 |
|----------|-----|-----|-------------|
| video | `cam_head`, `cam_left_wrist`, `cam_extra` | - | - |
| state/action | `left_arm` | 7 | RELATIVE delta |
| state/action | `right_arm` | 7 | RELATIVE delta |
| state/action | `left_hand` | 7 | ABSOLUTE |
| state/action | `right_hand` | 7 | ABSOLUTE |
| language | `annotation.human.task_description` | - | - |

Action horizon: 16 步

> **注意**: arm 的 action 类型是 RELATIVE delta，但推理服务器内部的 `unapply_action` 会自动完成反归一化和 delta→绝对角度 的转换，`PolicyClient.get_action()` 返回的已经是绝对关节角。

---

## 3. 评估流程

### Step 1: 开环验证（可选，推荐）

部署到真机前，用训练数据验证模型输出质量：

```bash
python scripts/deployment/standalone_inference_script.py \
  --model-path outputs/g1_0330_refinetune/checkpoint-4000 \
  --dataset-path cache/Jiangeng/G1_0330 \
  --embodiment-tag NEW_EMBODIMENT \
  --traj-ids 0 1 2 \
  --inference-mode pytorch
```

> `--steps` 默认为 None,会自动跑完每条 episode 的全部帧。结果图保存到 `outputs/standalone_inference/traj_{id}.jpeg`。

### Step 2: 启动 GR00T 推理服务器（GPU 侧）

```bash
python gr00t/eval/run_gr00t_server.py \
  --model_path outputs/g1_0330_refinetune/checkpoint-4000 \
  --embodiment_tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555
```

### Step 3: 启动 ImageServer（机器人侧）

在机器人上运行图像服务，将 head + wrist + extra 三路相机拼接为单张 JPEG 通过 ZMQ PUB 发送：

```bash
# 在机器人上执行
python image_server.py --config camera_config.json
```

### Step 4: 运行评估脚本（机器人侧或同网络主机）

```bash
python gr00t/eval/real_robot/G1/eval_g1_0330.py \
  --policy_host <GPU_SERVER_IP> \
  --policy_port 5555 \
  --action_horizon 16 \
  --control_frequency 30 \
  --lang_instruction "<任务描述>" \
  --camera_host <ROBOT_IP> \
  --camera_port 5555 \
  --dataset_path cache/Jiangeng/G1_0330 \
  --urdf ~/xr_teleoperate/assets/g1/g1_body29_hand14.urdf \
  --mesh_dir ~/xr_teleoperate/assets/g1/
```

> 默认从 `--dataset_path` 指定的训练数据集中读取 episode 0 frame 0 作为 init state(28-DOF: arm14 + hand14)。可用 `--init_episode N` 选择别的 episode;也可改用 `--init_state path.txt` 直接传 28-DOF 文本文件(优先级高于 dataset)。

---

## 4. 已完成项

### 4.1 [P0] Init State 对齐 ✅

已实现完整的初始化流程，对齐 ACT eval (`unitree_IL_lerobot/eval_robot/eval_g1.py`):

1. **坐姿构型** — `G1RobotInterface.__init__` 中硬编码 `G1_29_INIT_BODY_STATE`（hip=-1.3, knee=1.3），始终作为 `preset_body_q` 传给 `G1_29_ArmController`，3 秒内 cosine 插值到位
2. **手臂+手指初始化** — `init_to_start_pose(init_state_28)` 将 28-DOF（arm14+hand14）移到训练数据 episode 0 起始位，带重力补偿前馈
3. **用户确认** — `input("Enter 's' ...")` 等待用户按键
4. **到位检测** — 循环检查手臂关节误差 < 0.05rad，超时 5s

### 4.2 [P1] Modality Config 注册验证 ✅

已确认 `checkpoint-4000/processor_config.json` 包含 `new_embodiment` 配置：
- video: `cam_head`, `cam_left_wrist`, `cam_extra`
- state/action: `left_arm`, `right_arm`, `left_hand`, `right_hand`
- action type: arm=RELATIVE, hand=ABSOLUTE
- `embodiment_id.json` 中 `new_embodiment: 10`

**结论**：checkpoint 已打包 modality config，推理服务器无需额外导入。

### 4.3 [P2] Eval 结束后恢复初始姿态 ✅

`main()` 的 `finally` 块中调用 `robot.restore_init_pose(init_state_28)`：
- 将手臂+手指恢复到起始位姿
- 等待到位（tolerance=0.05rad, timeout=5s）
- Hold 3 秒后退出

---

## 5. 任务清单

| 优先级 | 项目 | 状态 | 说明 |
|--------|------|------|------|
| P0 | `eval_g1_0330.py` 核心脚本 | ✅ 已完成 | 相机接收 + 机器人控制 + 推理适配 |
| P0 | 重力补偿前馈 | ✅ 已完成 | `G1GravityCompensator` 通过 RNEA 计算 tauff |
| P0 | Init State 对齐 | ✅ 已完成 | 坐姿构型 + 手臂手指初始化 + 用户确认 + 到位检测 |
| P1 | Modality Config 验证 | ✅ 已确认 | checkpoint 已包含 `new_embodiment` config |
| P2 | 结束恢复姿态 | ✅ 已完成 | finally 中恢复手臂+手指到初始位 + hold 3s |

---

## 6. 参考文件

| 文件 | 说明 |
|------|------|
| `gr00t/eval/real_robot/G1/eval_g1_0330.py` | 真机评估主脚本 |
| `gr00t/eval/run_gr00t_server.py` | GR00T 推理服务器 |
| `gr00t/policy/server_client.py` | PolicyClient (ZMQ) |
| `examples/G1_0330/g1_0330_modality_config.py` | Modality 配置 |
| `examples/G1_0330/finetune_g1_0330.sh` | 微调脚本 |
| `~/xr_teleoperate/teleop/robot_control/robot_arm.py` | G1_29_ArmController |
| `~/xr_teleoperate/teleop/robot_control/robot_arm_ik.py` | IK + RNEA 重力补偿 |
| `~/xr_teleoperate/teleop/robot_control/robot_hand_unitree.py` | Dex3-1 手控制器 |
| `~/unitree_IL_lerobot/unitree_lerobot/eval_robot/eval_g1.py` | ACT eval (参考) |
| `docs/g1_control_loop_explained.md` | 控制环路详解文档 |
