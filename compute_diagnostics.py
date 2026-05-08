"""
compute_diagnostics.py
======================

Trains the CTE encoder on the paper's training split (10 contexts in
[5.0, 9.74] m/s^2), collects held-out trajectories at low-g {3, 4} and
high-g {12, 15} contexts, encodes them, and computes the diagnostics
the FAGEN reviewer asked for:

  1. Manifold-projection error: project each held-out embedding onto the
     1-D line fit (PCA component 1) through training-context centroids
     in the 64-D latent space; report L2 residual to the predicted
     centroid for each held-out g, plus within-training mean residual.

  2. Real t-SNE figure of train + held-out embeddings, color-coded by
     gravity (replaces the simulated fig5_latent_space.png used in the
     paper).

  3. JSON dump of all numbers so the notebook / paper can read them back.

Output directory: ./diagnostics_outputs/
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore")

# Local imports (the script assumes you run it from the repo root)
from src.data import TrajectoryCollector, TrajectoryDataset
from src.models import LSTMEncoder, SupConLoss
from src.training import EncoderTrainer

# -----------------------------------------------------------------------------
# Configuration --- mirrors the paper's experimental setup
# -----------------------------------------------------------------------------
ENV_NAME = "pendulum"
SEGMENT_LENGTH = 32
NUM_SEGMENTS_PER_CONTEXT_TRAIN = 100
NUM_SEGMENTS_PER_CONTEXT_HELDOUT = 30
LATENT_DIM = 64
HIDDEN_DIM = 256
NUM_EPOCHS = 50          # Bumped during testing if you want sharper embeddings
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
TEMPERATURE = 0.1
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

TRAIN_GRAVITIES = np.linspace(5.0, 9.74, 10).tolist()
HELDOUT_LOW_G   = [3.0, 4.0]
HELDOUT_HIGH_G  = [12.0, 15.0]
ALL_HELDOUT     = HELDOUT_LOW_G + HELDOUT_HIGH_G

OUTPUT_DIR = Path("diagnostics_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
def collect_segments(gravities, n_per_context, seed):
    contexts = [{"g": float(g)} for g in gravities]
    collector = TrajectoryCollector(
        env_name=ENV_NAME,
        contexts=contexts,
        segment_length=SEGMENT_LENGTH,
        num_segments_per_context=n_per_context,
        policy="random",
        seed=seed,
    )
    return collector.collect(verbose=False)


def train_encoder(train_segments):
    first_seg = train_segments[0]
    obs_dim = first_seg.observations.shape[-1]
    action_dim = first_seg.actions.shape[-1] if len(first_seg.actions.shape) > 1 else 1
    input_dim = obs_dim + action_dim

    print(f"[encoder] input_dim={input_dim}, device={DEVICE}")

    dataset = TrajectoryDataset(train_segments, augmentation="noise")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
    )

    encoder = LSTMEncoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        latent_dim=LATENT_DIM,
        dropout=0.2,
        bidirectional=True,
    )
    print(f"[encoder] params={sum(p.numel() for p in encoder.parameters()):,}")

    loss_fn = SupConLoss(temperature=TEMPERATURE)
    trainer = EncoderTrainer(
        encoder=encoder,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        weight_decay=1e-4,
        device=DEVICE,
        log_dir=None,
        checkpoint_dir=None,
    )
    trainer.train(num_epochs=NUM_EPOCHS, eval_every=max(1, NUM_EPOCHS // 5))
    return encoder


def embed_segments(encoder, segments, device=DEVICE):
    encoder.eval()
    encoder.to(device)
    embs, ctx_ids, gs = [], [], []
    with torch.no_grad():
        for seg in segments:
            obs = seg.observations
            actions = seg.actions
            if len(actions.shape) == 1:
                actions = actions[:, None]
            traj = np.concatenate([obs, actions], axis=-1)
            t = torch.FloatTensor(traj).unsqueeze(0).to(device)
            z = encoder(t).cpu().numpy().squeeze(0)
            embs.append(z)
            ctx_ids.append(seg.context_id)
            # context_id maps to gravity via segment metadata
            gs.append(seg.context_params.get("g") if hasattr(seg, "context_params") else None)
    return np.stack(embs), np.array(ctx_ids), np.array(gs, dtype=float)


def manifold_projection_error(train_emb, train_g, heldout_emb, heldout_g):
    """
    Fit a 1-D line through training-context centroids in 64-D and compute
    residual L2 distance for held-out embeddings.

    Returns dict with per-g residual and within-training mean residual.
    """
    # Centroid of each training context
    train_g_unique = np.array(sorted(set(train_g.tolist())))
    centroids = np.stack([
        train_emb[train_g == g].mean(axis=0) for g in train_g_unique
    ])
    # Fit line: principal axis through centroids
    centered = centroids - centroids.mean(axis=0)
    # SVD: principal direction = first right-singular vector
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0] / np.linalg.norm(Vt[0])
    origin = centroids.mean(axis=0)

    # Map each centroid to a scalar coordinate along the line
    centroid_coords = (centroids - origin) @ direction
    # Linear regression: scalar coord -> gravity (so we can predict
    # the line position for any held-out g)
    A = np.stack([centroid_coords, np.ones_like(centroid_coords)], axis=1)
    slope, intercept = np.linalg.lstsq(A, train_g_unique, rcond=None)[0]
    # Inverse mapping g -> coord:  coord = (g - intercept) / slope
    def g_to_point(g):
        coord = (g - intercept) / slope
        return origin + coord * direction

    # Within-training residual: mean L2 distance from training embeddings
    # to the line position predicted for their gravity
    train_residuals = []
    for g in train_g_unique:
        mask = train_g == g
        target = g_to_point(g)
        d = np.linalg.norm(train_emb[mask] - target, axis=1)
        train_residuals.append(d.mean())
    within_train_mean = float(np.mean(train_residuals))

    # Held-out residuals
    heldout_per_g = {}
    heldout_g_unique = np.array(sorted(set(heldout_g.tolist())))
    for g in heldout_g_unique:
        mask = heldout_g == g
        target = g_to_point(g)
        d = np.linalg.norm(heldout_emb[mask] - target, axis=1)
        heldout_per_g[float(g)] = {
            "mean_l2_residual": float(d.mean()),
            "std": float(d.std()),
            "n_segments": int(mask.sum()),
        }

    return {
        "within_train_mean_residual": within_train_mean,
        "heldout": heldout_per_g,
        "line_direction_norm": float(np.linalg.norm(direction)),
        "g_coord_slope": float(slope),
        "g_coord_intercept": float(intercept),
        "n_train_contexts": int(len(train_g_unique)),
    }


def make_real_tsne_figure(all_emb, all_g, out_path):
    from sklearn.manifold import TSNE
    print("[tsne] running on", len(all_emb), "embeddings")
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED)
    z2 = tsne.fit_transform(all_emb)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    sc = ax.scatter(z2[:, 0], z2[:, 1], c=all_g, cmap="viridis",
                    alpha=0.75, s=22, edgecolor="white", linewidth=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Gravity g (m/s$^2$)")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("Real CTE latent embeddings (train + held-out)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print("[tsne] wrote", out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("CTE diagnostics pipeline (FAGEN reproduction)")
    print(f"  device              : {DEVICE}")
    print(f"  train gravities     : {[round(g,2) for g in TRAIN_GRAVITIES]}")
    print(f"  held-out (low-g)    : {HELDOUT_LOW_G}")
    print(f"  held-out (high-g)   : {HELDOUT_HIGH_G}")
    print("=" * 70)

    # 1. Collect data
    print("\n[1/5] collecting training segments ...")
    train_segments = collect_segments(TRAIN_GRAVITIES,
                                      NUM_SEGMENTS_PER_CONTEXT_TRAIN,
                                      seed=SEED)
    print(f"      {len(train_segments)} training segments")

    print("[1/5] collecting held-out segments ...")
    heldout_segments = collect_segments(ALL_HELDOUT,
                                        NUM_SEGMENTS_PER_CONTEXT_HELDOUT,
                                        seed=SEED + 1)
    print(f"      {len(heldout_segments)} held-out segments")

    # 2. Train encoder
    print("\n[2/5] training encoder ...")
    encoder = train_encoder(train_segments)

    # Save the trained encoder so the policy phase / future runs can reuse it
    ckpt_path = OUTPUT_DIR / "encoder.pt"
    torch.save(encoder.state_dict(), ckpt_path)
    print(f"      checkpoint saved to {ckpt_path}")

    # 3. Embed both sets
    print("\n[3/5] embedding train + held-out segments ...")
    train_emb, train_ctx, train_g = embed_segments(encoder, train_segments)
    heldout_emb, heldout_ctx, heldout_g = embed_segments(encoder, heldout_segments)

    # If gravity wasn't surfaced via segment.context_params, fall back to mapping
    # context_id -> gravity manually using the order they were collected.
    if np.isnan(train_g).any():
        train_g = np.array([TRAIN_GRAVITIES[c] for c in train_ctx])
    if np.isnan(heldout_g).any():
        heldout_g = np.array([ALL_HELDOUT[c] for c in heldout_ctx])

    np.savez(OUTPUT_DIR / "embeddings.npz",
             train_emb=train_emb, train_g=train_g,
             heldout_emb=heldout_emb, heldout_g=heldout_g)
    print(f"      embeddings saved to {OUTPUT_DIR / 'embeddings.npz'}")

    # 4. Manifold-projection error
    print("\n[4/5] computing manifold-projection error ...")
    diag = manifold_projection_error(train_emb, train_g, heldout_emb, heldout_g)
    print("\n--- Manifold-projection error (real, single-seed) ---")
    print(f"  within-training mean residual : {diag['within_train_mean_residual']:.4f}")
    for g, info in diag["heldout"].items():
        print(f"  g={g:>5.1f}  L2 residual = {info['mean_l2_residual']:.4f} "
              f"(n={info['n_segments']}, std={info['std']:.4f})")
    print()

    with open(OUTPUT_DIR / "projection_error.json", "w") as f:
        json.dump(diag, f, indent=2)
    print(f"      JSON saved to {OUTPUT_DIR / 'projection_error.json'}")

    # 5. Real t-SNE figure
    print("\n[5/5] generating real t-SNE figure ...")
    all_emb = np.concatenate([train_emb, heldout_emb], axis=0)
    all_g   = np.concatenate([train_g,   heldout_g],   axis=0)
    make_real_tsne_figure(all_emb, all_g,
                          OUTPUT_DIR / "fig5_latent_space_real.png")

    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)
    print(f"  outputs in: {OUTPUT_DIR}/")
    for p in sorted(OUTPUT_DIR.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"    {p.name}  ({size_kb:.1f} KB)")
    print("=" * 70)


if __name__ == "__main__":
    main()