"""Soft comparisons, box helpers, the camera model and depth measurements.

Image convention: the gate works on the camera image flipped [::-1, ::-1] (what the detectors see, called the "F" image).
CameraModel converts between that image and the world frame (metres, z up).
"""
import math

import numpy as np

from .config import IMG_H, IMG_W, ENGINE_CFG

# result of calibrate_camera() on LIBERO's agentview camera in every one of our sessions
DEFAULT_FLIPS = (True, False)


# ---------- soft (graded) comparison helpers: return values in [0, 1] ----------
def ramp_down(x, lo, hi):
    """1 when x <= lo, 0 when x >= hi, linear in between."""
    if x <= lo:
        return 1.0
    if x >= hi:
        return 0.0
    return float((hi - x) / (hi - lo))


def ramp_up(x, lo, hi):
    """0 when x <= lo, 1 when x >= hi, linear in between."""
    return 1.0 - ramp_down(x, lo, hi)


def calibrate_conf(conf, cfg=None):
    """detector confidence -> grounding evidence g in [0, 1] (linear between conf_floor and conf_full)"""
    cfg = cfg or ENGINE_CFG
    return ramp_up(float(conf), cfg["conf_floor"], cfg["conf_full"])


# ---------- box helpers ----------
def box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def box_wh(b):
    return (max(b[2] - b[0], 1e-3), max(b[3] - b[1], 1e-3))


def box_diag(b):
    w, h = box_wh(b)
    return math.hypot(w, h)


def frac_inside(inner, outer):
    """fraction of the INNER box's area that lies inside OUTER"""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    w, h = box_wh(inner)
    return inter / (w * h)


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = box_wh(a)[0] * box_wh(a)[1] + box_wh(b)[0] * box_wh(b)[1] - inter
    return inter / ua if ua > 0 else 0.0


# ---------- camera ----------
class CameraModel:
    """Pinhole camera. K: 3x3 intrinsics, cam2world: 4x4 pose of the OpenCV camera (+z forward, +y down).
    flips = how the flipped detector image relates to the true camera image (found by calibrate_camera)."""

    def __init__(self, K, cam2world, H=IMG_H, W=IMG_W, flips=DEFAULT_FLIPS):
        self.K = np.asarray(K, dtype=float)
        self.cam2world = np.asarray(cam2world, dtype=float)
        self.world2cam = np.linalg.inv(self.cam2world)
        self.H, self.W = H, W
        self.flip_u, self.flip_v = flips

    @classmethod
    def from_mujoco(cls, env, camera_name="agentview", H=IMG_H, W=IMG_W, flips=None):
        """camera of a MuJoCo / robosuite environment (LIBERO)"""
        m, d = env.sim.model, env.sim.data
        cid = m.camera_name2id(camera_name)
        f = 0.5 * H / np.tan(np.deg2rad(float(m.cam_fovy[cid])) / 2.0)
        K = np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]])
        R_mj = np.array(d.cam_xmat[cid], dtype=float).reshape(3, 3)
        cam2world = np.eye(4)
        cam2world[:3, :3] = R_mj @ np.diag([1.0, -1.0, -1.0])   # MuJoCo cam (-z fwd, +y up) -> OpenCV (+z fwd, +y down)
        cam2world[:3, 3] = np.array(d.cam_xpos[cid], dtype=float)
        return cls(K, cam2world, H, W, DEFAULT_FLIPS if flips is None else flips)

    # flipped-image coords (what the detector sees) <-> true OpenCV pixel coords (an involution)
    def f_to_cv(self, u, v):
        return (self.W - u if self.flip_u else u, self.H - v if self.flip_v else v)

    def box_f_to_cv(self, b):
        x1, y1, x2, y2 = b
        if self.flip_u:
            x1, x2 = self.W - x2, self.W - x1
        if self.flip_v:
            y1, y2 = self.H - y2, self.H - y1
        return [x1, y1, x2, y2]

    def unproject(self, u_cv, v_cv, Z):
        X = (u_cv - self.K[0, 2]) * Z / self.K[0, 0]
        Y = (v_cv - self.K[1, 2]) * Z / self.K[1, 1]
        return (self.cam2world @ np.array([X, Y, Z, 1.0]))[:3]

    def project(self, p_world):
        pc = self.world2cam @ np.array([p_world[0], p_world[1], p_world[2], 1.0])
        if pc[2] <= 1e-6:
            return None
        return (self.K[0, 0] * pc[0] / pc[2] + self.K[0, 2], self.K[1, 1] * pc[1] / pc[2] + self.K[1, 2], pc[2])

    def world_to_f(self, p_world):
        pr = self.project(p_world)
        if pr is None:
            return None
        u, v = self.f_to_cv(pr[0], pr[1])
        return u, v, pr[2]

    def horizontal_axes(self):
        """camera 'right' and 'toward-scene' directions projected on the table plane (for left/right/front/behind)."""
        right = self.cam2world[:3, 0].copy(); right[2] = 0
        fwd = self.cam2world[:3, 2].copy(); fwd[2] = 0
        return right / (np.linalg.norm(right) + 1e-9), fwd / (np.linalg.norm(fwd) + 1e-9)


# ---------- depth ----------
def mujoco_depth_to_meters(env, depth_buffer):
    """MuJoCo depth buffer -> metres, flipped [::-1, ::-1] like the image the detectors see"""
    extent = env.sim.model.stat.extent
    near = env.sim.model.vis.map.znear * extent
    far = env.sim.model.vis.map.zfar * extent
    real_depth = near / (1.0 - depth_buffer * (1.0 - near / far))
    return real_depth[::-1, ::-1]


def depth_as_2d(depth_F):
    d = np.asarray(depth_F, dtype=float)
    return d[..., 0] if d.ndim == 3 else d


def box_depth(box_F, depth_F, shrink=0.4, max_valid=2.5):
    """Median depth of the central patch of a box (flipped-image coords)."""
    d2 = depth_as_2d(depth_F)
    H, W = d2.shape
    x1, y1, x2, y2 = box_F
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = max((x2 - x1) * shrink / 2.0, 1.0), max((y2 - y1) * shrink / 2.0, 1.0)
    c0, c1 = int(max(0, math.floor(cx - hw))), int(min(W, math.ceil(cx + hw)))
    r0, r1 = int(max(0, math.floor(cy - hh))), int(min(H, math.ceil(cy + hh)))
    patch = d2[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)]
    vals = patch[np.isfinite(patch) & (patch > 0) & (patch < max_valid)]
    if vals.size == 0:
        return float(d2[int(min(H - 1, max(0, cy))), int(min(W - 1, max(0, cx)))])
    return float(np.median(vals))


def measure_detection(det, depth_F, cam, cfg=None):
    """Detection (flipped-image box) -> record with calibrated confidence g, true-view box, 3D world position, size."""
    box_F = [float(v) for v in det["bbox"]]
    box = cam.box_f_to_cv(box_F)
    uc, vc = box_center(box)
    Z = box_depth(box_F, depth_F)
    return {
        "phrase": det.get("phrase_used", det.get("label")),
        "box_F": box_F, "box": box, "center": (uc, vc),
        "conf": float(det["confidence"]), "g": calibrate_conf(det["confidence"], cfg),
        "depth": Z, "pos": cam.unproject(uc, vc, Z),
        "size": float((box[2] - box[0]) * Z / cam.K[0, 0]),
        "height": float((box[3] - box[1]) * Z / cam.K[1, 1]),          # apparent height (m), used by 'on'
    }


def calibrate_camera(env, obs, inlier_tol=0.10, move_steps=12, verbose=True):
    """Finds how the flipped image relates to the real camera (4 flip hypotheses) and returns the best (flip_u, flip_v).
    Test 1 (decisive): move the gripper sideways and find WHERE in the depth image something new appeared.
    Test 2 (sanity): simulator body positions projected into the depth map should match its depth.
    Uses simulator positions ONLY for this one-time sensor calibration, never inside the evidence score.
    NOTE: this moves the robot -> set the episode up again afterwards."""
    cam = CameraModel.from_mujoco(env)
    hyps = [(fu, fv) for fu in (False, True) for fv in (False, True)]

    moves = []
    prev = obs
    for direction, n in ((+1, move_steps), (-1, 2 * move_steps)):
        cur = prev
        for _ in range(n):
            cur, _, _, _ = env.step([0, 0.8 * direction, 0, 0, 0, 0, -1])
        d_prev = depth_as_2d(mujoco_depth_to_meters(env, prev["agentview_depth"]))
        d_cur = depth_as_2d(mujoco_depth_to_meters(env, cur["agentview_depth"]))
        appeared = (d_prev - d_cur) > 0.03                      # something is now closer to the camera
        if appeared.sum() >= 5:
            vv, uu = np.nonzero(appeared)
            moves.append((np.array([uu.mean() + 0.5, vv.mean() + 0.5]), np.asarray(cur["robot0_eef_pos"], float)))
        prev = cur
    motion_err = {}
    for h in hyps:
        cam.flip_u, cam.flip_v = h
        errs = []
        for centroid, eef in moves:
            r = cam.world_to_f(eef)
            if r is not None:
                errs.append(float(np.linalg.norm(np.array(r[:2]) - centroid)))
        motion_err[h] = float(np.mean(errs)) if errs else float("inf")

    depth_F = depth_as_2d(mujoco_depth_to_meters(env, obs["agentview_depth"]))
    pts = [np.array(p, dtype=float) for p in np.asarray(env.sim.data.body_xpos)[1:]]
    pts += [np.array(obs[k], dtype=float) for k in obs
            if k.endswith("_pos") and "_to_" not in k and np.asarray(obs[k]).shape == (3,)]
    static = {}
    for h in hyps:
        cam.flip_u, cam.flip_v = h
        errs = []
        for p in pts:
            r = cam.world_to_f(p)
            if r is None:
                continue
            iu, iv = int(math.floor(r[0])), int(math.floor(r[1]))
            if 2 <= iu < cam.W - 2 and 2 <= iv < cam.H - 2:
                errs.append(abs(depth_F[iv, iu] - r[2]))
        errs = np.array(errs)
        static[h] = (float(np.mean(errs < inlier_tol)) if errs.size else 0.0, float(np.median(errs)) if errs.size else float("inf"))

    best = min(hyps, key=lambda h: (motion_err[h], -static[h][0]))
    if verbose:
        print("flip_u flip_v | motion_err(px)  static_inliers  static_median_err(m)")
        for h in hyps:
            print(f"{str(h[0]):6s} {str(h[1]):6s} | {motion_err[h]:14.1f}  {static[h][0]:14.2f}  {static[h][1]:18.3f}", "  <-- chosen" if h == best else "")
    return best, {"motion_err": motion_err, "static": static}
