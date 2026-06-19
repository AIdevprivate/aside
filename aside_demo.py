"""
Faithful, runnable demo of EXACTLY what ASIDE rotates.

Mirrors:
  - experiments/embeddings_init.py : generate_isoclinic_rotation_matrix(),
                                      rotate_embeddings_in_multiple_planes()
  - experiments/model.py (ForwardRotMixin.forward): rotation applied ONLY where
                                      segment_ids == 1 (data tokens), via
                                      new_embeds[mask] = inputs_embeds[mask] @ R
"""
import numpy as np

# ---- 1. The isoclinic rotation matrix (verbatim logic from embeddings_init.py) ----
def generate_isoclinic_rotation_matrix(dim, alpha):
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    M = np.eye(dim)
    for i in range(0, dim, 2):          # split dims into pairs (0,1),(2,3),...
        M[i,   i]   =  cos_a
        M[i,   i+1] = -sin_a
        M[i+1, i]   =  sin_a
        M[i+1, i+1] =  cos_a
    return M

# ---- 2. Fast pair-wise version actually used on the embedding matrix ----
def rotate_embeddings_in_multiple_planes(embeds, alpha):
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    even = embeds[:, 0::2]
    odd  = embeds[:, 1::2]
    out = embeds.copy()
    out[:, 0::2] = even * cos_a - odd * sin_a
    out[:, 1::2] = even * sin_a + odd * cos_a
    return out

np.set_printoptions(precision=3, suppress=True)
ALPHA = np.pi / 2   # the paper's default: rotation_alpha = 1.57079633 (90 degrees)
DIM   = 8           # tiny embedding dim for readability (real models: 3584, 4096, ...)

print("="*70)
print("ASIDE rotation angle alpha = pi/2 (90 deg), embedding dim =", DIM)
print("="*70)

R = generate_isoclinic_rotation_matrix(DIM, ALPHA)
print("\n[1] Isoclinic rotation matrix R (block-diagonal 2x2 rotations):")
print(R)
print("\n  -> orthogonal? R @ R.T == I :", np.allclose(R @ R.T, np.eye(DIM)))
print("  -> det(R) =", round(float(np.linalg.det(R)), 6), "(proper rotation)")

# ---- 3. A toy prompt: 3 instruction tokens, 4 data tokens ----
# segment_ids: 0 = instruction (NOT rotated), 1 = data (rotated)
rng = np.random.default_rng(0)
tokens      = ["Translate", "to", "German", "Who", "is", "Albert", "Einstein"]
segment_ids = np.array([0,            0,     0,        1,     1,    1,        1])
embeds      = rng.standard_normal((len(tokens), DIM))

print("\n[2] Prompt tokens and their roles (segment_ids):")
for t, s in zip(tokens, segment_ids):
    print(f"    {t:10s} -> segment {s}  ({'DATA  -> ROTATED' if s==1 else 'INSTR -> unchanged'})")

# ---- 4. ForwardRot forward pass: rotate ONLY where segment_ids == 1 ----
mask = segment_ids == 1
new_embeds = embeds.copy()
new_embeds[mask] = embeds[mask] @ R     # exactly model.py line 451-452 (rotation_direction='right')

print("\n[3] Per-token change after the ASIDE forward pass:")
for i, t in enumerate(tokens):
    moved = np.linalg.norm(new_embeds[i] - embeds[i])
    print(f"    {t:10s} seg={segment_ids[i]}  ||new-old||={moved:8.4f}  "
          f"{'<-- ROTATED' if moved>1e-6 else '<-- untouched'}")

print("\n[4] Sanity checks:")
print("    instruction rows identical to original :",
      np.allclose(new_embeds[~mask], embeds[~mask]))
print("    norms preserved on data rows (rotation) :",
      np.allclose(np.linalg.norm(new_embeds[mask],axis=1),
                  np.linalg.norm(embeds[mask],axis=1)))
print("    fast pairwise == matrix form on data    :",
      np.allclose(rotate_embeddings_in_multiple_planes(embeds[mask], ALPHA),
                  embeds[mask] @ R))

# ---- 5. Show the literal numbers for ONE data token ----
i = 3  # "Who"
print(f"\n[5] Concrete numbers for data token '{tokens[i]}' (pairs swap & negate at 90 deg):")
print("    original :", embeds[i])
print("    rotated  :", new_embeds[i])
print("    note: (x0,x1)->(-x1,x0), (x2,x3)->(-x3,x2), ... i.e. a 90-deg turn in each plane")
