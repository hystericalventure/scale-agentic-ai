"""
Regenerate results/figures/diagram1_architecture.png to match the paper's
current Figure 1: five processing layers + a persistent IEP/LMS/Behavioural
Data Store (six parts total), with the data store read by every layer
(dashed arrows) and written only by the Consent Ledger (solid arrow).
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 12))
ax.set_xlim(0, 10)
ax.set_ylim(3.6, 15)
ax.axis('off')

def box(x, y, w, h, text, sub=None, fc="#eef0f2", ec="#333333"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                        linewidth=1.3, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    if sub:
        ax.text(x + w/2, y + h*0.62, text, ha='center', va='center',
                 fontsize=12, fontweight='bold', color='#16212b')
        ax.text(x + w/2, y + h*0.28, sub, ha='center', va='center',
                 fontsize=9.5, color='#4a5a68')
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                 fontsize=12, fontweight='bold', color='#16212b')

def solid_arrow(x1, y1, x2, y2, color='#4a5a68'):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=16,
                         linewidth=1.6, color=color)
    ax.add_patch(a)

def dashed_arrow(x1, y1, x2, y2, color='#9aa7b2'):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12,
                         linewidth=1.1, color=color, linestyle=(0, (4, 3)))
    ax.add_patch(a)

# Title
ax.text(5, 14.55, "SCALE Layered Architecture", ha='center', fontsize=17,
        fontweight='bold', color='#16212b')
ax.text(5, 14.1, "Six parts once the persistent data store is counted",
        ha='center', fontsize=10.5, color='#4a5a68')

# Layer boxes (top to bottom): L1 .. L5, then Data Store at bottom
W, H = 6.6, 1.05
X = (10 - W) / 2

y1 = 12.6
box(X, y1, W, H, "L1: Interface", "IN: student/teacher input   OUT: raw request", fc="#f3f4f6")

y2 = 11.15
box(X, y2, W, H, "L2: LLM Interpretation", "IN: raw request   OUT: parsed intent  (designed, not implemented)",
    fc="#ece9f9")

y3 = 9.7
box(X, y3, W, H, "L3: MCP Routing", "IN: parsed intent   OUT: routed request", fc="#ece9f9")

y4 = 8.0
b4 = FancyBboxPatch((X, y4), W, 1.65, boxstyle="round,pad=0.08,rounding_size=0.12",
                     linewidth=1.3, edgecolor="#333333", facecolor="#e3f2ee")
ax.add_patch(b4)
ax.text(X + W/2, y4 + 1.65 - 0.28, "L4: Multi-Agent Layer", ha='center', fontsize=12,
        fontweight='bold', color='#16212b')
ax.text(X + W/2, y4 + 1.65 - 0.5, "IN: routed request + learner state   OUT: 4 candidate actions",
        ha='center', fontsize=9, color='#4a5a68')

# four agent sub-boxes inside L4
agent_names = ["ZPD\nScaffolding", "Consent\n& Trust", "Overload\nPrediction", "Teacher\nCo-Pilot"]
aw = W / 4 - 0.12
for i, name in enumerate(agent_names):
    ax_x = X + 0.1 + i * (aw + 0.12)
    box(ax_x, y4 + 0.12, aw, 0.62, name, fc="#d7ece5")

y5 = 6.35
box(X, y5, W, H, "L5: Consent Ledger",
    "IN: 4 candidate actions   OUT: 1 consent-checked action", fc="#fbe9e3")

y6 = 4.6
box(X, y6, W, 1.2, "IEP / LMS / Behavioural Data Store",
    "persistent record — not a processing stage", fc="#eef0f2")

# Solid arrows down the pipeline
solid_arrow(5, y1, 5, y2 + H, )
solid_arrow(5, y2, 5, y3 + H)
solid_arrow(5, y3, 5, y4 + 1.65)
solid_arrow(5, y4, 5, y5 + H)
solid_arrow(5, y5, 5, y6 + 1.2, color='#c9603f')  # written only by Consent Ledger -> data store

# Label the write arrow
ax.text(5.9, (y5 + y6 + 1.2) / 2, "writes\n(solid)", fontsize=8.5, color='#c9603f', ha='left', va='center')

# Dashed "read by every layer" arrows from data store up to each layer
read_x = X - 0.55
ax.text(read_x - 0.15, (y1 + y6) / 2, "dashed = read\nby every layer",
        rotation=90, ha='center', va='center', fontsize=8.5, color='#7a8895')
for (yy, hh) in [(y1, H), (y2, H), (y3, H), (y4, 1.65), (y5, H)]:
    dashed_arrow(X - 0.15, y6 + 0.6, X - 0.15, yy + hh/2)

# Output label
ax.text(5, 3.9, "output: consent-checked action shown to the learner",
        ha='center', fontsize=10, style='italic', color='#4a5a68')

plt.tight_layout()
plt.savefig("results/figures/diagram1_architecture.png", dpi=170, bbox_inches='tight')
print("wrote results/figures/diagram1_architecture.png")
