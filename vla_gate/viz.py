"""Pictures of what the gate saw: the boxes it chose (target, reference, goal) on the camera image."""
import numpy as np

ROLE_COLORS = {"target": "red", "reference": "cyan", "goal": "lime", "candidate": "yellow"}


def scene_boxes(scene):
    """[(role, phrase, box_F)] for the boxes the gate used"""
    out = []

    def add(rec, role):
        if isinstance(rec, dict) and rec.get("box_F") is not None:
            out.append((role, str(rec.get("phrase") or role), [float(v) for v in rec["box_F"]]))
    add(scene.get("target"), "target")
    for r in scene.get("refs") or []:
        add(r, "reference")
    add(scene.get("goal"), "goal")
    return out


def show_result(image_F, result, title="", ax=None):
    """image_F: the flipped camera image (numpy or PIL); result: GateResult"""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    own = ax is None
    if own:
        fig, ax = plt.subplots(1, figsize=(4.6, 5.0))
    ax.imshow(np.asarray(image_F))
    for c in (result.scene.get("candidates") or [])[1:]:          # other bowl candidates
        b = c["box"]
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], lw=1, edgecolor=ROLE_COLORS["candidate"],
                                       facecolor="none", ls="--"))
    for role, phrase, b in scene_boxes(result.scene):
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], lw=2, edgecolor=ROLE_COLORS[role], facecolor="none"))
        ax.text(b[0], max(b[1] - 3, 6), f"{role}: {phrase}", color=ROLE_COLORS[role], fontsize=7, weight="bold")
    verdict = "EXECUTE" if result.execute else "STOP"
    ax.set_title((title + "\n" if title else "") + f"G={result.G:.2f}  S={result.S:.2f}  D={result.D:.2f}  C={result.C:.2f}  ->  {verdict}",
                 fontsize=8)
    ax.axis("off")
    if own:
        plt.show()
    return ax
