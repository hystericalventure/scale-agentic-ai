"""
Regenerate results/figures/diagram2_prac_loop.png to match the paper's
Figure 2: the five-stage Consent-Aware Perception-Reasoning-Action-Consent
loop, run independently by each agent, with two stages beyond the usual
three (Consent Check, Trust Model Update).
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.35, 1.35)
ax.set_aspect('equal')
ax.axis('off')

ax.text(0, 1.28, "The Consent-Aware Perception–Reasoning–Action–Consent Loop",
        ha='center', fontsize=13, fontweight='bold', color='#16212b')
ax.text(0, 1.16, "Run independently by each agent; repeats every interaction",
        ha='center', fontsize=9.5, color='#4a5a68')

stages = [
    ("1. Perception", "behavioural signals,\nprofile, task context", "#eef0f2"),
    ("2. Reasoning", "ZPD estimate +\noverload risk score", "#ece9f9"),
    ("3. Proposed\nAction", "scaffold change, content\nswap, break", "#e3f2ee"),
    ("4. Consent\nCheck", "student accepts /\nrejects / tunes", "#fbe9e3"),
    ("5. Trust Model\nUpdate", "accepted -> executed; either way\ntrust/consent profile updated", "#f3e6ea"),
]

n = len(stages)
R = 0.92
angles = [np.pi/2 - 2*np.pi*i/n for i in range(n)]  # start at top, go clockwise

box_w, box_h = 0.66, 0.44
centers = []
for (title, sub, fc), ang in zip(stages, angles):
    cx, cy = R*np.cos(ang), R*np.sin(ang)
    centers.append((cx, cy))
    b = FancyBboxPatch((cx-box_w/2, cy-box_h/2), box_w, box_h,
                        boxstyle="round,pad=0.05,rounding_size=0.08",
                        linewidth=1.3, edgecolor="#333333", facecolor=fc)
    ax.add_patch(b)
    ax.text(cx, cy+0.10, title, ha='center', va='center', fontsize=10,
            fontweight='bold', color='#16212b')
    ax.text(cx, cy-0.11, sub, ha='center', va='center', fontsize=7.3, color='#4a5a68')

# curved arrows between consecutive stages, going around the circle
for i in range(n):
    x1, y1 = centers[i]
    x2, y2 = centers[(i+1) % n]
    a = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle="arc3,rad=0.18",
                         arrowstyle='-|>', mutation_scale=18, linewidth=1.6,
                         color='#4a5a68', shrinkA=26, shrinkB=26)
    ax.add_patch(a)

# center label: "repeats every interaction"
c = Circle((0, 0), 0.28, facecolor='#f7f9fb', edgecolor='#d3d9e0', linewidth=1)
ax.add_patch(c)
ax.text(0, 0.05, "loop", ha='center', fontsize=9.5, fontweight='bold', color='#2f8f7f')
ax.text(0, -0.08, "repeats", ha='center', fontsize=8, color='#4a5a68')

ax.text(0, -1.28,
        "No scaffold change is made without student consent or a pre-authorised safety override.\n"
        "A rejection still moves trust, by a smaller fixed step, and shapes every later proposal.",
        ha='center', fontsize=8.8, color='#4a5a68')

plt.tight_layout()
plt.savefig("results/figures/diagram2_prac_loop.png", dpi=170, bbox_inches='tight')
print("wrote results/figures/diagram2_prac_loop.png")
