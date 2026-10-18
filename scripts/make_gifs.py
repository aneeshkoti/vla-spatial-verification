"""Records the robot clips shown in the README (assets/gifs/*.gif) and writes the clip gallery into README.md.

    python scripts/make_gifs.py              # about 10 minutes on an 8 GB GPU

Every clip is a real LIBERO-Spatial episode run with this repository's code. The gate checks the scene first (boxes and the
score C); only after EXECUTE does the policy act. The scenes are ones from our attached run (Table VII of the paper), and the
captions are written from what actually happens in your run.
"""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vla_gate import SpatialGate, flipped_image  # noqa: E402
from vla_gate.geometry import CameraModel  # noqa: E402
from vla_gate.sim import SCENE_PERTURBATIONS, make_case, make_env, setup_episode, true_keys  # noqa: E402
from vla_gate.viz import scene_boxes  # noqa: E402

GIF_DIR = os.path.join(ROOT, "assets", "gifs")
README = os.path.join(ROOT, "README.md")
START, END = "<!-- gallery:start -->", "<!-- gallery:end -->"
SIZE, FPS, EVERY = 320, 10, 2            # GIF size (pixels), frames per second, one frame every 2 simulator steps

# expected = (decision, C, policy outcome) in our attached run with the hybrid gate (initial state 5)
CLIPS = [
    dict(name="execute_correct", task=1, state=5, kind="clean", policy="smolvla", title="Correct instruction",
         expected=("EXECUTE", 1.00, "success")),
    dict(name="execute_occlusion", task=8, state=5, kind="occlusion", policy="smolvla", title="Valid, part of the scene covered",
         expected=("EXECUTE", 0.97, "success")),
    dict(name="execute_distractor_xvla", task=0, state=5, kind="distractor", policy="xvla", title="Valid, extra bowl added, X-VLA",
         expected=("EXECUTE", 1.00, "success")),
    dict(name="stop_absent_reference", task=6, state=5, kind="absent_reference", policy="smolvla",
         title="Impossible: the reference is not on the table", expected=("ABSTAIN", 0.00, None)),
    dict(name="stop_relation_violated", task=1, state=5, kind="relation_violation", policy="smolvla",
         title="Impossible: the bowl was moved away", expected=("ABSTAIN", 0.25, None)),
    dict(name="limitation_moved_plate", task=0, state=2, kind="position_change", policy="smolvla", gate=False,
         title="Limitation: the plate was moved", expected=None),
]

WHITE, GREEN, RED, GREY, ORANGE, BLUE = (255, 255, 255), (46, 158, 107), (200, 50, 40), (110, 110, 110), (235, 104, 52), (42, 120, 214)
ROLE_COL = {"target": (220, 40, 40), "reference": (0, 170, 200), "goal": (40, 170, 70)}


def font(size, bold=False):
    for p in (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
              f"C:/Windows/Fonts/arial{'bd' if bold else ''}.ttf", "/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_TAG, F_BIG, F_SUB, F_BOX = font(14, True), font(28, True), font(13, True), font(12, True)


def up(frame):
    return Image.fromarray(np.asarray(frame)).resize((SIZE, SIZE), Image.BICUBIC)


def tag(im, text):
    """small label in the top-left corner"""
    im = im.copy()
    d = ImageDraw.Draw(im, "RGBA")
    w = d.textlength(text, font=F_TAG)
    d.rectangle([6, 6, 18 + w, 30], fill=(20, 24, 32, 200))
    d.text((12, 9), text, font=F_TAG, fill=WHITE)
    return im


def banner(im, text, color, sub=""):
    """verdict bar at the bottom"""
    im = im.copy()
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, SIZE - 56, SIZE, SIZE], fill=color + (215,))
    d.text(((SIZE - d.textlength(text, font=F_BIG)) / 2, SIZE - 54), text, font=F_BIG, fill=WHITE)
    if sub:
        d.text(((SIZE - d.textlength(sub, font=F_SUB)) / 2, SIZE - 20), sub, font=F_SUB, fill=WHITE)
    return im


def draw_boxes(im, scene):
    """the boxes the gate used: target (red), reference (cyan), goal (green)"""
    im = im.copy()
    d = ImageDraw.Draw(im)
    s = SIZE / 256.0
    for role, phrase, b in scene_boxes(scene):
        c = ROLE_COL[role]
        x1, y1, x2, y2 = [v * s for v in b]
        d.rectangle([x1, y1, x2, y2], outline=c, width=3)
        w = d.textlength(phrase, font=F_BOX)
        ty = max(0, y1 - 17)
        d.rectangle([x1, ty, x1 + w + 8, ty + 16], fill=c)
        d.text((x1 + 4, ty + 1), phrase, font=F_BOX, fill=WHITE)
    return im


def circle(im, uv, color, dashed=False, r=20):
    if uv is None:
        return im
    im = im.copy()
    d = ImageDraw.Draw(im)
    s = SIZE / 256.0
    u, v = uv[0] * s, uv[1] * s
    box = [u - r, v - r, u + r, v + r]
    if dashed:
        for a in range(0, 360, 30):
            d.arc(box, start=a, end=a + 18, fill=color, width=4)
    else:
        d.ellipse(box, outline=color, width=4)
    return im


def hold(im, seconds):
    return [im] * max(1, int(round(seconds * FPS)))


def write_gif(frames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        tmp = tempfile.mkdtemp(prefix="gif_")
        try:
            for i, f in enumerate(frames):
                f.convert("RGB").save(os.path.join(tmp, f"{i:05d}.png"))
            cmd = [ffmpeg, "-v", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(tmp, "%05d.png"), "-vf",
                   "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                   "-loop", "0", path]
            subprocess.run(cmd, check=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:                                                    # no ffmpeg: Pillow's own GIF writer (larger file)
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    return os.path.getsize(path)


def rollout(policy_name, policy, env, clip, instr, on_frame):
    from vla_gate.policies import rollout_smolvla, rollout_xvla
    pert = clip["kind"] if clip["kind"] in SCENE_PERTURBATIONS else None
    if policy_name == "xvla":
        return rollout_xvla(env, policy, clip["task"], clip["state"], instr, perturb=pert, on_frame=on_frame)
    return rollout_smolvla(env, policy, clip["task"], clip["state"], instr, perturb=pert, on_frame=on_frame)


def record(clip, gate, policy):
    """runs one clip; returns (frames, caption html, report line)"""
    pname = "X-VLA" if clip["policy"] == "xvla" else "SmolVLA"
    env, _, _ = make_env(clip["task"])
    try:
        cam = CameraModel.from_mujoco(env)
        old_uv = None
        if clip["kind"] == "position_change":                 # where the plate stands in the original layout
            o0 = setup_episode(env, clip["task"], clip["state"])
            _, gk0 = true_keys(env, o0)
            old_pos = np.asarray(o0[gk0], float).copy()
            old_uv = cam.world_to_f(old_pos)
        obs, instr = make_case(env, clip["task"], clip["state"], clip["kind"])
        first = up(flipped_image(obs))
        frames, report = [], f"task {clip['task']} state {clip['state']} {clip['kind']}"
        quote = f"<i>“{instr}”</i>"
        if clip.get("gate", True):
            res = gate.check_libero(env, obs, instr)
            boxed = draw_boxes(first, res.scene)
            verdict = "EXECUTE" if res.execute else "STOP"
            frames += hold(tag(first, "the gate checks the scene"), 0.6)
            frames += hold(tag(boxed, f"G {res.G:.2f}   S {res.S:.2f}   D {res.D:.2f}"), 1.6)
            frames += hold(banner(boxed, verdict, GREEN if res.execute else RED, f"C = {res.C:.2f}"), 1.4)
            exp = clip["expected"]
            report += f" | gate: {verdict} C={res.C:.2f}" + (" (our run: same)" if exp and res.decision == exp[0] and abs(res.C - exp[1]) < 0.006
                                                               else f" (our run: {exp[0]} C={exp[1]:.2f})" if exp else "")
            if not res.execute:
                frames += hold(banner(boxed, "STOP", RED, "the robot does not move"), 2.4)
                return frames, f"<b>{clip['title']}</b><br>{quote}<br>C = {res.C:.2f} → <b>STOP</b>: the robot does not move.", report
        raw, obs_log = [], []

        def on_frame(o):
            raw.append(flipped_image(o))
            if not obs_log:
                obs_log.append(o)
            obs_log[1:] = [o]
        row = rollout(clip["policy"], policy, env, clip, instr, on_frame)
        new_uv = tkey = gkey = None
        if old_uv is not None:
            tkey, gkey = true_keys(env, obs_log[0])
            new_uv = cam.world_to_f(np.asarray(obs_log[0][gkey], float))
        for i, f in enumerate(raw[::EVERY]):
            im = up(f)
            if old_uv is not None:
                im = circle(circle(im, old_uv, ORANGE, dashed=True), new_uv, BLUE)
            frames.append(tag(im, f"{pname}{' alone' if not clip.get('gate', True) else ''} acting · step {i * EVERY}"))
        last = frames[-1]
        if old_uv is not None:                                # moved plate: where did the bowl end up?
            bowl = np.asarray(obs_log[-1][tkey], float)
            d_old = float(np.linalg.norm(bowl[:2] - old_pos[:2]))
            d_new = float(np.linalg.norm(bowl[:2] - np.asarray(obs_log[0][gkey], float)[:2]))
            if d_old < d_new:
                frames += hold(banner(last, "OLD SPOT", ORANGE, "where the plate used to be"), 2.2)
                text = (f"{pname} alone carries the bowl to where the plate used to be (orange, {100 * d_old:.0f} cm away) instead of "
                        f"the plate (blue, {100 * d_new:.0f} cm). A check before motion cannot see this.")
            else:
                frames += hold(banner(last, "NOT DONE" if not row["success"] else "SUCCESS", GREY if not row["success"] else GREEN), 2.2)
                text = f"{pname} alone, plate moved: {'success' if row['success'] else 'task not completed'}."
            report += f" | bowl ends {100 * d_old:.0f} cm from the old spot, {100 * d_new:.0f} cm from the plate"
            return frames, f"<b>{clip['title']}</b><br>{quote}<br>{text}", report
        ok = bool(row["success"])
        frames += hold(banner(last, "SUCCESS" if ok else "NOT DONE", GREEN if ok else GREY, f"{pname}, {row['episode_length']} steps"), 2.2)
        report += f" | {pname}: {'success' if ok else row['failure_reason']} in {row['episode_length']} steps"
        caption = (f"<b>{clip['title']}</b><br>{quote}<br>C = {res.C:.2f} → <b>EXECUTE</b>, then {pname} "
                   f"{'completes the task' if ok else 'acts but does not complete the task'}.")
        return frames, caption, report
    finally:
        env.close()


def update_readme(done):
    """writes the clip table between the gallery markers of README.md"""
    cells = [f'<td align="center" width="33%"><img src="assets/gifs/{name}.gif" width="260"><br><sub>{cap}</sub></td>'
             for name, cap in done]
    rows = ["<tr>\n" + "\n".join(cells[i:i + 3]) + "\n</tr>" for i in range(0, len(cells), 3)]
    block = (f"{START}\n## Robot clips\n\nReal LIBERO-Spatial episodes. First the gate checks the scene (boxes: target red, reference cyan, "
             f"goal green); only after EXECUTE does the policy move. Recorded with `scripts/make_gifs.py`.\n\n<table>\n"
             + "\n".join(rows) + f"\n</table>\n{END}")
    text = open(README, encoding="utf-8").read()
    if START in text and END in text:
        a, b = text.index(START), text.index(END) + len(END)
        text = text[:a] + block + text[b:]
    else:
        text = text.replace("\n## The problem", "\n" + block + "\n\n## The problem", 1)
    open(README, "w", encoding="utf-8").write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="clip names to record (default: all)")
    ap.add_argument("--no-readme", action="store_true", help="only write the GIFs")
    args = ap.parse_args()
    clips = [c for c in CLIPS if not args.only or c["name"] in args.only]
    import torch
    print("loading the gate (OWLv2 + Grounding DINO) ...")
    gate = SpatialGate.load("hybrid")
    results, policies = {}, {}
    for clip in sorted(clips, key=lambda c: c["policy"] == "xvla"):          # SmolVLA clips first, then X-VLA
        t0 = time.time()
        try:
            if clip["policy"] not in policies:
                for k in list(policies):                                       # only one policy on the GPU at a time
                    del policies[k]
                gc.collect()
                torch.cuda.empty_cache()
                from vla_gate.policies import XVLA, SmolVLA
                print(f"loading {'X-VLA' if clip['policy'] == 'xvla' else 'SmolVLA'} ...")
                policies[clip["policy"]] = XVLA() if clip["policy"] == "xvla" else SmolVLA()
            frames, caption, report = record(clip, gate, policies[clip["policy"]])
            size = write_gif(frames, os.path.join(GIF_DIR, clip["name"] + ".gif"))
            results[clip["name"]] = caption
            print(f"OK   {clip['name']:26s} {report} | {len(frames)} frames, {size / 1e6:.1f} MB, {time.time() - t0:.0f} s")
        except Exception as e:
            print(f"FAILED {clip['name']}: {type(e).__name__}: {e}")
    cap_file = os.path.join(GIF_DIR, "captions.json")       # captions of every clip recorded so far (also earlier runs)
    captions = json.load(open(cap_file, encoding="utf-8")) if os.path.exists(cap_file) else {}
    captions.update(results)
    if results:
        json.dump(captions, open(cap_file, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    done = [(c["name"], captions[c["name"]]) for c in CLIPS
            if c["name"] in captions and os.path.exists(os.path.join(GIF_DIR, c["name"] + ".gif"))]
    if done and not args.no_readme:
        update_readme(done)
        print(f"\nREADME.md gallery updated with {len(done)} clips -> {GIF_DIR}")


if __name__ == "__main__":
    main()
