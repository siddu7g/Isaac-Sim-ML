#!/usr/bin/env python3
# ============================================================
# PX4 + MAVSDK + PPO TRAIN (150 eps) + SAVE .pt + INFERENCE (ep 151)
#
# GOAL (RELATIVE TO HOME): (-12, +1)
#
# FIXES INCLUDED (your requests):
#  1) CLEAN distance computation (always 2D in RELATIVE frame)
#       n_rel = st.n - HOME_N
#       e_rel = st.e - HOME_E
#       dist  = hypot(GOAL_DN - n_rel, GOAL_DE - e_rel)
#
#  2) Reward re-balance (less timestep pressure, more goal tolerance weight)
#
#  3) 15x15 plot/record constraint:
#       - We record points ONLY if inside [-15,15] x [-15,15] (relative to HOME)
#       - We ALSO filter again at PLOT TIME (hard guarantee)
#       - Plot axes are FORCED to [-15,15] (hard guarantee)
#
# NOTE:
#  - "Don't count beyond 15x15" is implemented as: don't record/plot those points.
#  - The UAV can still physically fly out there; we just won't store/plot it.
# ============================================================

import asyncio, math, time
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import matplotlib.pyplot as plt

from mavsdk import System
from mavsdk.offboard import PositionNedYaw, OffboardError

# ============================================================
# CONFIG
# ============================================================

SYSTEM_ADDR = "udp://:14540"

CTRL_DT = 0.1
EPISODE_TIMEOUT_S = 40.0

TRAIN_EPISODES = 100
INFER_EPISODE_ID = 101

TAKEOFF_ALT_M = 2.5

# -----------------------------
# GOAL relative to HOME (meters)
# -----------------------------
GOAL_DN = -11.0
GOAL_DE =  1.0
GOAL_TOL_M = 0.8

# Micro-waypoint step
STEP_M = 0.6

# Normalization
MAX_DIST = 20.0

# PPO
OBS_DIM = 4
ACT_DIM = 2
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 2e-4
PPO_EPOCHS = 4

# Exploration
ENT_COEF = 0.01
STD_MIN = 0.20
STD_MAX = 1.20

# Debug prints
PRINT_EVERY_STEPS = 20

# Plot outputs
PLOT_ALL_PNG = "trajectories.png"
PLOT_LAST50_PNG = "trajectories_last50.png"

# Checkpoint
CKPT_PATH = "ppo_px4_policy.pt"

# 15x15 bound (relative to HOME)
PLOT_BOUND_M = 15.0

# Reward knobs (rebalanced)
PROGRESS_W = 3.0
ACT_PEN_W = 0.003
TIME_PEN = -0.0001          # allow long rollouts
TOL_BAND_M = 2.0            # near-goal shaping radius
TOL_SHAPE_W = 2.0           # increase to 3–5 if it hovers near goal
SUCCESS_BONUS = 30.0

# ============================================================
# TELEMETRY
# ============================================================

@dataclass
class Telemetry:
    n: float = 0.0
    e: float = 0.0
    d: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    got_pos: bool = False
    got_att: bool = False

async def telemetry_task(drone: System, st: Telemetry):
    async for pv in drone.telemetry.position_velocity_ned():
        st.n = pv.position.north_m
        st.e = pv.position.east_m
        st.d = pv.position.down_m
        st.got_pos = True

async def attitude_task(drone: System, st: Telemetry):
    async for att in drone.telemetry.attitude_euler():
        st.roll = att.roll_deg
        st.pitch = att.pitch_deg
        st.got_att = True

# ============================================================
# PPO NETWORK
# ============================================================

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(OBS_DIM, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.mu = nn.Linear(64, ACT_DIM)
        self.v = nn.Linear(64, 1)

        # std ~ 0.67 initial
        self.log_std = nn.Parameter(torch.ones(ACT_DIM) * -0.4)

        # remove initial directional bias
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def forward(self, obs: torch.Tensor):
        x = self.base(obs)
        mu = torch.tanh(self.mu(x))
        std = torch.exp(self.log_std)
        std = torch.clamp(std, min=STD_MIN, max=STD_MAX)
        v = self.v(x).squeeze(-1)
        return mu, std, v

# ============================================================
# UTILS
# ============================================================

def rel_ne(st: Telemetry, home_n: float, home_e: float):
    return (st.n - home_n, st.e - home_e)

def in_bounds(n_rel: float, e_rel: float) -> bool:
    return (abs(n_rel) <= PLOT_BOUND_M) and (abs(e_rel) <= PLOT_BOUND_M)

def dist_to_goal_rel(n_rel: float, e_rel: float) -> float:
    # CLEAN FIX: true 2D distance in relative frame
    return math.hypot(GOAL_DN - n_rel, GOAL_DE - e_rel)

def make_obs(st: Telemetry, home_n: float, home_e: float) -> torch.Tensor:
    n_rel, e_rel = rel_ne(st, home_n, home_e)
    dn = (GOAL_DN - n_rel) / MAX_DIST
    de = (GOAL_DE - e_rel) / MAX_DIST
    return torch.tensor([
        float(np.clip(dn, -2.0, 2.0)),
        float(np.clip(de, -2.0, 2.0)),
        float(np.clip(st.roll / 30.0, -2.0, 2.0)),
        float(np.clip(st.pitch / 30.0, -2.0, 2.0)),
    ], dtype=torch.float32)

def compute_gae(rews, vals, dones):
    adv = []
    gae = 0.0
    vals = vals + [0.0]
    for i in reversed(range(len(rews))):
        delta = rews[i] + GAMMA * vals[i+1] * (1.0 - dones[i]) - vals[i]
        gae = delta + GAMMA * LAMBDA * (1.0 - dones[i]) * gae
        adv.insert(0, gae)
    return adv

def save_checkpoint(model: nn.Module, opt: optim.Optimizer, episode: int, successes: int, path: str):
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "episode": episode,
        "successes": successes,
        "config": {
            "goal": {"GOAL_DN": GOAL_DN, "GOAL_DE": GOAL_DE, "GOAL_TOL_M": GOAL_TOL_M},
            "reward": {
                "PROGRESS_W": PROGRESS_W, "ACT_PEN_W": ACT_PEN_W, "TIME_PEN": TIME_PEN,
                "TOL_BAND_M": TOL_BAND_M, "TOL_SHAPE_W": TOL_SHAPE_W, "SUCCESS_BONUS": SUCCESS_BONUS
            },
            "plot_bound": PLOT_BOUND_M
        }
    }, path)
    print(f"[checkpoint] saved -> {path} (episode={episode}, successes={successes})")

def load_checkpoint(model: nn.Module, path: str):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"[checkpoint] loaded <- {path} (saved_episode={ckpt.get('episode')}, successes={ckpt.get('successes')})")
    return ckpt

def plot_trajectories(all_trajs, out_png, last_n=None):
    """
    HARD-GUARANTEE plotting:
      - Filter points again at plot time
      - Force x/y limits to [-15, 15]
    all_trajs items:
      {"ep": int, "xy": [(n_rel,e_rel),...], "success": bool, "mode": "train"/"infer"}
    """
    trajs = all_trajs[-last_n:] if last_n is not None else all_trajs

    def _inb(p):
        n_rel, e_rel = p
        return in_bounds(n_rel, e_rel)

    plt.figure()
    for tr in trajs:
        xy = tr.get("xy", [])
        if not xy:
            continue

        # HARD filter at plot time
        xy_f = [p for p in xy if _inb(p)]
        if len(xy_f) < 2:
            continue

        xs = [p[1] for p in xy_f]  # East_rel
        ys = [p[0] for p in xy_f]  # North_rel

        if tr.get("mode") == "infer":
            lw, alpha = 3.0, 0.95
        else:
            lw = 2.5 if tr.get("success") else 1.0
            alpha = 0.85 if tr.get("success") else 0.45

        plt.plot(xs, ys, linewidth=lw, alpha=alpha)

    plt.scatter([0.0], [0.0], s=90, marker="o", label="Home (0,0)")
    plt.scatter([GOAL_DE], [GOAL_DN], s=130, marker="x", label="Goal")

    plt.xlabel("East_rel (m)")
    plt.ylabel("North_rel (m)")
    title = f"PPO Trajectories (Last {last_n})" if last_n else "PPO Trajectories (All Episodes)"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.axis("equal")

    # HARD force bounds
    plt.xlim(-PLOT_BOUND_M, PLOT_BOUND_M)
    plt.ylim(-PLOT_BOUND_M, PLOT_BOUND_M)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[plot] saved {out_png}")

async def hover_reset(drone: System, home_n: float, home_e: float, alt_m: float, steps: int = 25):
    for _ in range(steps):
        await drone.offboard.set_position_ned(PositionNedYaw(home_n, home_e, -alt_m, 0.0))
        await asyncio.sleep(CTRL_DT)

# ============================================================
# EPISODES
# ============================================================

async def run_training_episode(drone, st, model, opt, ep, home_n, home_e):
    await hover_reset(drone, home_n, home_e, TAKEOFF_ALT_M, steps=25)

    # trajectory storage (relative, filtered)
    ep_xy = []
    n_rel, e_rel = rel_ne(st, home_n, home_e)
    if in_bounds(n_rel, e_rel):
        ep_xy.append((n_rel, e_rel))

    ep_success = False

    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, val_buf, done_buf = [], [], []

    t0 = time.time()
    dist_prev = dist_to_goal_rel(n_rel, e_rel)
    first_print = False

    while time.time() - t0 < EPISODE_TIMEOUT_S:
        obs = make_obs(st, home_n, home_e)
        mu, std, v = model(obs)

        d = Normal(mu, std)
        act = d.sample()  # TRAIN: stochastic
        logp = d.log_prob(act).sum()

        ax, ay = act.detach().numpy()
        next_n = st.n + STEP_M * float(ax)
        next_e = st.e + STEP_M * float(ay)

        if not first_print:
            n0, e0 = rel_ne(st, home_n, home_e)
            yaw = math.degrees(math.atan2((GOAL_DE - e0), (GOAL_DN - n0)))
            print(f"\n[ep {ep:03d}] start N,E=({st.n:.2f},{st.e:.2f}) -> first cmd N,E=({next_n:.2f},{next_e:.2f}) yaw={yaw:.1f}")
            print(f"[ep {ep:03d}] mu={mu.detach().numpy()} std={std.detach().numpy()}")
            first_print = True

        await drone.offboard.set_position_ned(PositionNedYaw(next_n, next_e, -TAKEOFF_ALT_M, 0.0))
        await asyncio.sleep(CTRL_DT)

        # RELATIVE state + CLEAN dist
        n_rel, e_rel = rel_ne(st, home_n, home_e)
        dist_now = dist_to_goal_rel(n_rel, e_rel)

        # record ONLY inside 15x15
        if in_bounds(n_rel, e_rel):
            ep_xy.append((n_rel, e_rel))

        # reward (rebalanced)
        progress = (dist_prev - dist_now)
        reward = PROGRESS_W * progress - ACT_PEN_W * (ax * ax + ay * ay) + TIME_PEN

        if dist_now < TOL_BAND_M:
            closeness = (TOL_BAND_M - dist_now) / TOL_BAND_M
            reward += TOL_SHAPE_W * closeness

        done = 0.0
        if dist_now < GOAL_TOL_M:
            reward += SUCCESS_BONUS
            done = 1.0
            ep_success = True

        obs_buf.append(obs)
        act_buf.append(act)
        logp_buf.append(logp)
        rew_buf.append(float(reward))
        val_buf.append(float(v.item()))
        done_buf.append(float(done))

        dist_prev = dist_now

        if len(rew_buf) % PRINT_EVERY_STEPS == 0:
            dn = (GOAL_DN - n_rel)
            de = (GOAL_DE - e_rel)
            # Print RELATIVE N,E so you never confuse it with absolute
            print(f"[ep {ep:03d}] step={len(rew_buf):03d} dist={dist_now:.2f} dn={dn:+.2f} de={de:+.2f} "
                  f"N_rel,E_rel=({n_rel:+.2f},{e_rel:+.2f}) r={reward:+.3f}")

        if done:
            print(f"[ep {ep:03d}] SUCCESS ✅ steps={len(rew_buf)} dist={dist_now:.2f}")
            break

    # PPO update
    if len(rew_buf) < 10:
        print(f"[ep {ep:03d}] too-short rollout, skipping update")
        return {"success": ep_success, "reward_sum": float(sum(rew_buf)), "steps": len(rew_buf), "xy": ep_xy}

    adv = compute_gae(rew_buf, val_buf, done_buf)
    ret = [a + v for a, v in zip(adv, val_buf)]

    obs_t = torch.stack(obs_buf)
    act_t = torch.stack(act_buf)
    logp_old = torch.stack(logp_buf).detach()
    adv_t = torch.tensor(adv, dtype=torch.float32)
    ret_t = torch.tensor(ret, dtype=torch.float32)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        mu2, std2, v2 = model(obs_t)
        d2 = Normal(mu2, std2)
        logp2 = d2.log_prob(act_t).sum(-1)
        ratio = torch.exp(logp2 - logp_old)

        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
        pi_loss = -torch.min(surr1, surr2).mean()

        v_loss = (ret_t - v2).pow(2).mean()
        entropy = d2.entropy().sum(-1).mean()

        loss = pi_loss + 0.5 * v_loss - ENT_COEF * entropy

        opt.zero_grad()
        loss.backward()
        opt.step()

    return {"success": ep_success, "reward_sum": float(sum(rew_buf)), "steps": len(rew_buf), "xy": ep_xy}

async def run_inference_episode(drone, st, model, ep_label, home_n, home_e):
    await hover_reset(drone, home_n, home_e, TAKEOFF_ALT_M, steps=25)

    ep_xy = []
    n_rel, e_rel = rel_ne(st, home_n, home_e)
    if in_bounds(n_rel, e_rel):
        ep_xy.append((n_rel, e_rel))

    ep_success = False

    t0 = time.time()
    dist_prev = dist_to_goal_rel(n_rel, e_rel)
    first_print = False
    steps = 0
    reward_sum = 0.0

    model.eval()
    with torch.no_grad():
        while time.time() - t0 < EPISODE_TIMEOUT_S:
            obs = make_obs(st, home_n, home_e)
            mu, std, v = model(obs)

            # INFER: deterministic
            ax, ay = mu.detach().numpy()
            next_n = st.n + STEP_M * float(ax)
            next_e = st.e + STEP_M * float(ay)

            if not first_print:
                n0, e0 = rel_ne(st, home_n, home_e)
                yaw = math.degrees(math.atan2((GOAL_DE - e0), (GOAL_DN - n0)))
                print(f"\n[ep {ep_label:03d}][INFER] start -> first cmd N,E=({next_n:.2f},{next_e:.2f}) yaw={yaw:.1f}")
                print(f"[ep {ep_label:03d}][INFER] mu={mu.detach().numpy()} std={std.detach().numpy()} (mu-only)")
                first_print = True

            await drone.offboard.set_position_ned(PositionNedYaw(next_n, next_e, -TAKEOFF_ALT_M, 0.0))
            await asyncio.sleep(CTRL_DT)

            n_rel, e_rel = rel_ne(st, home_n, home_e)
            dist_now = dist_to_goal_rel(n_rel, e_rel)

            if in_bounds(n_rel, e_rel):
                ep_xy.append((n_rel, e_rel))

            steps += 1

            progress = (dist_prev - dist_now)
            reward = PROGRESS_W * progress - ACT_PEN_W * (ax * ax + ay * ay) + TIME_PEN
            if dist_now < TOL_BAND_M:
                closeness = (TOL_BAND_M - dist_now) / TOL_BAND_M
                reward += TOL_SHAPE_W * closeness
            if dist_now < GOAL_TOL_M:
                reward += SUCCESS_BONUS
                ep_success = True

            reward_sum += float(reward)
            dist_prev = dist_now

            if steps % PRINT_EVERY_STEPS == 0:
                dn = (GOAL_DN - n_rel)
                de = (GOAL_DE - e_rel)
                print(f"[ep {ep_label:03d}][INFER] step={steps:03d} dist={dist_now:.2f} dn={dn:+.2f} de={de:+.2f} "
                      f"N_rel,E_rel=({n_rel:+.2f},{e_rel:+.2f}) r={reward:+.3f}")

            if ep_success:
                print(f"[ep {ep_label:03d}][INFER] SUCCESS ✅ steps={steps} dist={dist_now:.2f}")
                break

    return {"success": ep_success, "reward_sum": float(reward_sum), "steps": steps, "xy": ep_xy}

# ============================================================
# MAIN
# ============================================================

async def main():
    print(f"[mavsdk] connecting via {SYSTEM_ADDR}")
    drone = System()
    await drone.connect(system_address=SYSTEM_ADDR)

    st = Telemetry()
    asyncio.create_task(telemetry_task(drone, st))
    asyncio.create_task(attitude_task(drone, st))

    while not (st.got_pos and st.got_att):
        await asyncio.sleep(0.1)

    print("[px4] arming...")
    await drone.action.arm()
    print("[px4] takeoff...")
    await drone.action.takeoff()
    await asyncio.sleep(6.0)

    # start offboard (must send one setpoint first)
    await drone.offboard.set_position_ned(PositionNedYaw(st.n, st.e, -TAKEOFF_ALT_M, 0.0))
    try:
        await drone.offboard.start()
        print("[px4] offboard started ✅")
    except OffboardError as e:
        raise RuntimeError(f"Offboard start failed: {e._result.result_str}") from e

    HOME_N, HOME_E = st.n, st.e
    print(f"[home] HOME_N={HOME_N:.2f} HOME_E={HOME_E:.2f} alt={TAKEOFF_ALT_M:.2f}")
    print(f"[goal] GOAL_REL_N={GOAL_DN:.2f} GOAL_REL_E={GOAL_DE:.2f} tol={GOAL_TOL_M:.2f}m")
    print(f"[plot] HARD window: N_rel,E_rel ∈ [-{PLOT_BOUND_M},{PLOT_BOUND_M}] meters")
    print(f"[reward] PROGRESS_W={PROGRESS_W} ACT_PEN_W={ACT_PEN_W} TIME_PEN={TIME_PEN} "
          f"TOL_BAND_M={TOL_BAND_M} TOL_SHAPE_W={TOL_SHAPE_W} SUCCESS_BONUS={SUCCESS_BONUS}")

    model = ActorCritic()
    opt = optim.Adam(model.parameters(), lr=LR)

    successes = 0
    all_trajs = []  # {"ep": int, "xy":[(n_rel,e_rel)...], "success": bool, "mode": "train"/"infer"}

    try:
        # TRAIN
        for ep in range(1, TRAIN_EPISODES + 1):
            res = await run_training_episode(drone, st, model, opt, ep, HOME_N, HOME_E)
            if res["success"]:
                successes += 1
            all_trajs.append({"ep": ep, "xy": res["xy"], "success": res["success"], "mode": "train"})
            print(f"[ep {ep:03d}] done steps={res['steps']} reward={res['reward_sum']:+.2f} succ={successes}/{ep}")

            if ep % 25 == 0:
                save_checkpoint(model, opt, ep, successes, CKPT_PATH)

        print("[train] Training done.")
        save_checkpoint(model, opt, TRAIN_EPISODES, successes, CKPT_PATH)

        # INFER (ep 151)
        print(f"\n[infer] loading checkpoint and running inference as episode {INFER_EPISODE_ID}...")
        load_checkpoint(model, CKPT_PATH)

        inf = await run_inference_episode(drone, st, model, INFER_EPISODE_ID, HOME_N, HOME_E)
        all_trajs.append({"ep": INFER_EPISODE_ID, "xy": inf["xy"], "success": inf["success"], "mode": "infer"})
        print(f"[ep {INFER_EPISODE_ID:03d}][INFER] done steps={inf['steps']} reward={inf['reward_sum']:+.2f} success={inf['success']}")

    finally:
        # Plot (HARD clamped and HARD filtered)
        if all_trajs:
            plot_trajectories(all_trajs, PLOT_ALL_PNG, last_n=None)
            plot_trajectories(all_trajs, PLOT_LAST50_PNG, last_n=50)

        try:
            await drone.offboard.stop()
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())

