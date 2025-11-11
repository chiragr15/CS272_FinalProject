import time
import math
import numpy as np
import pygame
import gymnasium as gym
from highway_env.road.lane import LineType

import narrow_street_env  # your custom env module

# --------------- Config knobs (tweak live) ----------------
FPS = 30
EPISODE_STEPS = 1500              
RANDOM_SEED = 0

# Rendering + env difficulty overrides for clarity
OVERRIDES = {
    "duration": 1000,                 # longer episode horizon (env still truncates internally)
    "screen_width": 1280,
    "screen_height": 320,
    "show_trajectories": True,
    "render_agent": True,
    "traffic_density": 0.08,
    "right_lane_narrow_factor": 0.80,
    "parked_spacing": 26.0,
    "parked_depth": 0.6,
}

# Colors for HUD text
WHITE = (240, 240, 240)
GREEN = (0, 255, 0)
RED = (255, 90, 90)
BLUE = (90, 180, 255)
YELLOW = (255, 220, 50)
GRAY = (160, 160, 160)

# --------------- Small helpers ----------------
def expected_parked_count(cfg):
    L = cfg["road_length"]
    start = cfg["parked_start"]
    spacing = cfg["parked_spacing"]
    return max(0, int(max(0, (L - 10.0 - start)) // spacing))

def lane_y_targets(cfg):
    """Approx expected y centers for right/left lanes."""
    w = cfg["lane_width"]
    return (0.0, w)

def find_front_same_lane(road, veh, lane_width):
    """Nearest object/vehicle ahead in same lane (simple heuristic)."""
    if veh is None:
        return None, None, None
    vx, vy = veh.position
    tol = lane_width * 0.7
    best_dx, front, front_speed = float("inf"), None, 0.0

    # moving vehicles
    for other in road.vehicles:
        if other is veh:
            continue
        dx = other.position[0] - vx
        if dx > 0 and abs(other.position[1] - vy) < tol and dx < best_dx:
            best_dx, front = dx, other
            front_speed = getattr(other, "speed", 0.0)

    # parked objects (if any)
    objs = getattr(road, "objects", [])
    for obj in objs:
        if not hasattr(obj, "position"):
            continue
        dx = obj.position[0] - vx
        if dx > 0 and abs(obj.position[1] - vy) < tol and dx < best_dx:
            best_dx, front = dx, obj
            front_speed = 0.0

    return front, front_speed, best_dx if front is not None else None

def ttc(ego_speed, front_speed, dx):
    rel_v = max(0.0, ego_speed - max(0.0, front_speed))
    if rel_v <= 1e-6:
        return math.inf
    return dx / rel_v if dx is not None and dx > 0 else math.inf

def count_parked(env):
    # Parked may live in road.objects or as zero-speed vehicles
    n_obj = len(getattr(env.unwrapped.road, "objects", []))
    n_zero_speed = sum(1 for v in env.unwrapped.road.vehicles if getattr(v, "speed", 0.0) == 0.0 and v is not env.unwrapped.vehicle)
    return n_obj if n_obj > 0 else n_zero_speed

def enforce_lane_markings(env):
    """Force visible lane markings regardless of env defaults."""
    try:
        rl = env.unwrapped._right_lane
        ll = env.unwrapped._left_lane
        rl.line_types = (LineType.CONTINUOUS, LineType.STRIPED)
        ll.line_types = (LineType.STRIPED, LineType.CONTINUOUS)
    except Exception:
        pass

def recolor_entities(env):
    """Make colors semantic: ego green, left-lane traffic blue, parked gray."""
    try:
        ego = env.unwrapped.vehicle
        ego.color = (0, 255, 0)

        lane_w = env.unwrapped.config["lane_width"]
        # Color left-lane movers
        for v in env.unwrapped.road.vehicles:
            if v is ego:
                continue
            # if near left-lane center -> blue; if zero-speed -> gray; else yellow (misc)
            if getattr(v, "speed", 0.0) == 0.0:
                v.color = (110, 110, 110)
            elif abs(v.position[1] - lane_w) < lane_w * 0.35:
                v.color = (80, 180, 255)  # blue
            elif abs(v.position[1]) < lane_w * 0.35:
                v.color = (255, 120, 120)  # red-ish for right-lane movers (if any)
            else:
                v.color = (240, 200, 60)   # yellow fallback

        # Color parked obstacles (if objects API exists)
        for o in getattr(env.unwrapped.road, "objects", []):
            if hasattr(o, "color"):
                o.color = (100, 100, 100)
    except Exception:
        pass

# --------------- Main viewer ----------------
def main():
    narrow_street_env.register_env()
    env = gym.make("NarrowStreetTwoLane-v0", render_mode="rgb_array")

    # Apply overrides
    if hasattr(env, "configure"):
        env.configure(OVERRIDES)
    else:
        env.unwrapped.config.update(OVERRIDES)

    # Reset and post-process visuals
    obs, info = env.reset(seed=RANDOM_SEED)
    enforce_lane_markings(env)
    recolor_entities(env)

    # --------- Sanity checks at reset ---------
    cfg = env.unwrapped.config
    exp_parked = expected_parked_count(cfg)
    got_parked = count_parked(env)
    right_y, left_y = lane_y_targets(cfg)

    ego = env.unwrapped.vehicle
    warnings = []
    if abs(ego.position[1] - right_y) > cfg["lane_width"] * 0.4:
        warnings.append(f"[WARN] Ego not in right lane at reset (y={ego.position[1]:.2f}).")

    if got_parked == 0 and exp_parked > 0:
        warnings.append(f"[WARN] Expected ~{exp_parked} parked cars, found {got_parked}.")

    movers = [v for v in env.unwrapped.road.vehicles if v is not ego and getattr(v, "speed", 0.0) > 0.2]
    offlane = sum(1 for v in movers if abs(v.position[1] - left_y) > cfg["lane_width"] * 0.5)
    if offlane > 0:
        warnings.append(f"[WARN] {offlane}/{len(movers)} moving vehicles not near left lane center.")

    print("---- NarrowStreet debug ----")
    print(f"Expected parked ~ {exp_parked}, actual {got_parked}")
    if warnings:
        for w in warnings:
            print(w)
    else:
        print("Sanity checks OK at reset.")

    # ---------- Pygame window for annotated frames ----------
    pygame.init()
    W, H = cfg["screen_width"], cfg["screen_height"]
    screen = pygame.display.set_mode((W, H))
    pygame.display.setcaption = pygame.display.set_caption  # compat alias
    pygame.display.setcaption("NarrowStreetTwoLane — Debug Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo", 16)

    step = 0
    ep_reward = 0.0
    running = True
    while running and step < EPISODE_STEPS:
        # Basic autoplay: KEEP_LANE most of the time; sometimes SLOWER to avoid crashes
        action = 0 if np.random.rand() > 0.1 else 4
        obs, r, term, trunc, info = env.step(action)
        ep_reward += r
        step += 1

        # Legend + numbers
        ego = env.unwrapped.vehicle
        front, fspd, dx = find_front_same_lane(env.unwrapped.road, ego, cfg["lane_width"])
        ttc_val = ttc(getattr(ego, "speed", 0.0), (0.0 if fspd is None else fspd), (dx if dx is not None else -1))

        # Recompute movers each frame for accurate count
        movers = [
            v for v in env.unwrapped.road.vehicles
            if v is not ego and getattr(v, "speed", 0.0) > 0.2
        ]

        # Grab rendered frame and blit
        frame = env.render()  # RGB array (H, W, 3)
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        # ---- HUD text overlay (fixed formatting) ----
        dx_str = "None" if dx is None else f"{dx:.1f}"
        fspd_val = 0.0 if fspd is None else float(fspd)
        fspd_str = f"{fspd_val:.1f}"
        ttc_str = "inf" if math.isinf(ttc_val) else f"{ttc_val:.1f}"

        lines = [
            "Legend:  Ego=Green   Left-lane traffic=Blue   Right-lane movers=Red   Parked=Gray   Other=Yellow",
            f"Step: {step}   EpReward: {ep_reward:.2f}   Parked: {got_parked}   Movers: {len(movers)}",
            f"Ego y: {ego.position[1]:.2f} (right lane ~ {right_y:.2f})",
            f"Front dx: {dx_str} m   Front speed: {fspd_str} m/s   TTC: {ttc_str} s",
        ]

        # Draw a semi-transparent HUD background bar
        hud_rect = pygame.Surface((W, 70), pygame.SRCALPHA)
        hud_rect.fill((0, 0, 0, 110))
        screen.blit(hud_rect, (0, 0))
        # Draw lines
        y = 4
        for s in lines:
            text = font.render(s, True, WHITE)
            screen.blit(text, (8, y))
            y += 18

        # Event handling + flip
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.display.flip()
        clock.tick(FPS)

        if term or trunc:
            banner = font.render(f"Episode ended — term={term} trunc={trunc}  Reward={ep_reward:.2f}", True, WHITE)
            screen.blit(banner, (8, H - 24))
            pygame.display.flip()
            time.sleep(0.8)

            obs, info = env.reset()
            enforce_lane_markings(env)
            recolor_entities(env)
            ep_reward = 0.0
            step = 0
            # refresh counts after reset
            got_parked = count_parked(env)

    env.close()
    pygame.quit()



if __name__ == "__main__":
    main()
