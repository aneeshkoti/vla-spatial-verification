"""Synthetic table-top scenes for the tests: a camera above a table, objects with known positions, a depth map,
and fake detectors whose answers depend only on (scene, phrase). No simulator, no models, no GPU."""
import random
import types

import numpy as np

H = W = 256
TABLE_Z = 0.86
EXTENT, ZNEAR, ZFAR = 2.0, 0.01, 50.0

# class -> (half size x, half size y, height)
SIZES = {"bowl": (0.055, 0.055, 0.05), "plate": (0.09, 0.09, 0.02), "ramekin": (0.045, 0.045, 0.05),
         "cookie box": (0.07, 0.05, 0.08), "cabinet": (0.12, 0.10, 0.28), "stove": (0.10, 0.10, 0.06)}
WORDS = {  # which phrase finds which object class (first = main name)
    "bowl": ["black bowl", "bowl", "dark bowl"], "plate": ["plate", "white plate", "dish"],
    "ramekin": ["ramekin", "small white bowl", "white cup"], "cookie box": ["cookie box", "box", "carton"],
    "cabinet": ["wooden cabinet", "cabinet", "drawer"], "stove": ["stove", "cooktop", "burner"],
}


def look_at(cam_pos, target):
    z = np.asarray(target, float) - np.asarray(cam_pos, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)          # OpenCV camera axes in world coordinates (columns)


class FakeEnv:
    """just enough of a MuJoCo environment for CameraModel and the depth conversion"""

    def __init__(self, cam_pos, R_cv, fovy=45.0):
        R_mj = R_cv @ np.diag([1.0, -1.0, -1.0])
        model = types.SimpleNamespace(camera_name2id=lambda name: 0, cam_fovy=np.array([fovy]),
                                      stat=types.SimpleNamespace(extent=EXTENT),
                                      vis=types.SimpleNamespace(map=types.SimpleNamespace(znear=ZNEAR, zfar=ZFAR)))
        data = types.SimpleNamespace(cam_xmat=np.array([R_mj.reshape(-1)]), cam_xpos=np.array([cam_pos]))
        self.sim = types.SimpleNamespace(model=model, data=data)


class Scene:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.seed = seed
        cam_pos = np.array([0.66 + rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 1.61 + rng.uniform(-0.05, 0.05)])
        self.R = look_at(cam_pos, [-0.2, 0.0, TABLE_Z])
        self.cam_pos = cam_pos
        self.env = FakeEnv(cam_pos, self.R)
        f = 0.5 * H / np.tan(np.deg2rad(45.0) / 2.0)
        self.K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
        # objects: two bowls always; the others sometimes
        self.objects = []
        names = ["bowl", "bowl", "plate"] + [n for n in ("ramekin", "cookie box", "cabinet", "stove") if rng.random() < 0.6]
        placed = []
        for n in names:
            for _ in range(200):
                p = np.array([rng.uniform(-0.40, 0.15), rng.uniform(-0.30, 0.30)])
                if all(np.linalg.norm(p - q) > 0.12 for q in placed):
                    break
            placed.append(p)
            hx, hy, hz = SIZES[n]
            base = TABLE_Z
            self.objects.append({"cls": n, "xy": p, "z0": base, "size": (hx, hy, hz)})
        # sometimes stack a bowl on the ramekin / cookie box / cabinet / plate (the 'on' relation)
        supports = [o for o in self.objects if o["cls"] in ("ramekin", "cookie box", "cabinet", "plate")]
        if supports and rng.random() < 0.5:
            s = supports[int(rng.integers(len(supports)))]
            b = self.objects[0]
            b["xy"] = s["xy"] + rng.normal(0, 0.01, 2)
            b["z0"] = s["z0"] + s["size"][2]
        self.depth_F = self._render_depth()
        for o in self.objects:
            o["box_F"] = self._box_F(o)
        raw = self.depth_F[::-1, ::-1]
        near, far = ZNEAR * EXTENT, ZFAR * EXTENT
        self.obs = {"agentview_image": rng.integers(0, 255, (H, W, 3), dtype=np.uint8),
                    "agentview_depth": ((1.0 - near / raw) / (1.0 - near / far))[..., None].astype(np.float64)}

    # ---- geometry ----
    def _to_cam(self, p):
        return self.R.T @ (np.asarray(p, float) - self.cam_pos)

    def _project_cv(self, p):
        pc = self._to_cam(p)
        return self.K[0, 0] * pc[0] / pc[2] + self.K[0, 2], self.K[1, 1] * pc[1] / pc[2] + self.K[1, 2], pc[2]

    def _box_F(self, o):
        hx, hy, hz = o["size"]
        x, y = o["xy"]
        us, vs = [], []
        for dx in (-hx, hx):
            for dy in (-hy, hy):
                for z in (o["z0"], o["z0"] + hz):
                    u, v, _ = self._project_cv([x + dx, y + dy, z])
                    us.append(W - u)      # flips (True, False): u_F = W - u_cv, v_F = v_cv
                    vs.append(v)
        return [float(min(us)), float(min(vs)), float(max(us)), float(max(vs))]

    def _render_depth(self):
        d = np.zeros((H, W))
        vv, uu = np.mgrid[0:H, 0:W]
        u_cv = W - (uu + 0.5)
        v_cv = vv + 0.5
        dirs = np.stack([(u_cv - self.K[0, 2]) / self.K[0, 0], (v_cv - self.K[1, 2]) / self.K[1, 1], np.ones_like(u_cv)], -1)
        wdir = dirs @ self.R.T
        t = (TABLE_Z - self.cam_pos[2]) / wdir[..., 2]
        d[:] = np.where(t > 0, t, 3.0)
        # objects, far to near
        objs = sorted(self.objects, key=lambda o: -self._to_cam([*o["xy"], o["z0"] + o["size"][2]])[2])
        for o in objs:
            b = self._box_F(o)
            z = self._to_cam([*o["xy"], o["z0"] + o["size"][2]])[2]
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                d[y1:y2, x1:x2] = np.minimum(d[y1:y2, x1:x2], z)
        return d

    # ---- fake detectors: the answer depends only on (scene seed, phrase, object) ----
    def _detect(self, phrase, lo, hi, box_threshold, false_rate):
        p = phrase.lower().strip()
        out = []
        for j, o in enumerate(self.objects):
            r = random.Random(f"{self.seed}|{p}|{j}")
            if p in WORDS[o["cls"]]:
                c = r.uniform(lo, hi) if WORDS[o["cls"]].index(p) == 0 or r.random() < 0.7 else r.uniform(lo, (lo + hi) / 2)
            elif r.random() < false_rate:                # a wrong object matched with a lower score
                c = r.uniform(lo * 0.6, (lo + hi) / 2)
            else:
                continue
            jit = [r.uniform(-2, 2) for _ in range(4)]
            b = [v + e for v, e in zip(o["box_F"], jit)]
            if c >= box_threshold:
                out.append({"bbox": [round(v, 1) for v in b], "confidence": round(c, 3), "label": phrase})
        if p == "table":
            r = random.Random(f"{self.seed}|table")
            c = r.uniform(lo, hi)
            if c >= box_threshold:
                out.append({"bbox": [10.0, 90.0, 246.0, 250.0], "confidence": round(c, 3), "label": "table"})
        r = random.Random(f"{self.seed}|{p}|spurious")
        if r.random() < 0.15:                            # a box on nothing in particular
            x, y = r.uniform(0, 220), r.uniform(60, 220)
            c = r.uniform(lo * 0.6, hi)
            if c >= box_threshold:
                out.append({"bbox": [round(x, 1), round(y, 1), round(x + r.uniform(8, 40), 1), round(y + r.uniform(8, 40), 1)],
                            "confidence": round(c, 3), "label": phrase})
        return out

    def gdino(self, image_pil, phrase, box_threshold=0.35, text_threshold=0.25):
        return self._detect(phrase, 0.18, 0.75, box_threshold, 0.25)

    def owl(self, image_pil, phrase, box_threshold=0.10, text_threshold=None):
        p = phrase.lower().strip()
        if p in WORDS["cabinet"] + WORDS["stove"]:       # like OWLv2: misses the cabinet and the stove
            return []
        return self._detect(phrase, 0.05, 0.50, box_threshold, 0.10)


LIBERO_INSTRUCTIONS = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
]
EXTRA_INSTRUCTIONS = [
    "pick up the black bowl to the left of the plate and place it on the plate",
    "pick up the black bowl right of the ramekin and place it on the plate",
    "pick up the black bowl behind the plate and place it on the plate",
    "pick up the black bowl in front of the ramekin and place it on the plate",
    "pick up the black bowl far from the plate and place it on the plate",
    "pick up the black bowl near the cookie box and place it on the plate",
    "pick up the black bowl under the stove and place it on the plate",
    "pick up the black bowl on top of the plate and place it on the plate",
    "put the black bowl next to the plate",
    "pick up the plate",
]
