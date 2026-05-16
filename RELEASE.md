# Publishing pretrained weights

Riffly resolves `Riffly("<alias>")` to a checkpoint downloaded from a GitHub
Release the first time it is used. This document is the runbook for
publishing a new checkpoint asset. The registry that maps aliases to URLs
and sha256 hashes lives in [`riffly/weights.py`](riffly/weights.py).

## Current default — `pop`

| field | value |
|---|---|
| alias | `pop` |
| local artefact | `runs/pop_v3/model.pt` |
| size | 6.12 MB |
| sha256 | `b78a5d29c082510bc1757d5c7bc15050ab2baeceddc99fd99a419a283647cfb9` |
| release tag | `weights-v1` |
| asset filename | `pop.pt` |
| URL | `https://github.com/YannickGibson/riffly/releases/download/weights-v1/pop.pt` |
| training data | ADL Piano MIDI / Pop (CC BY 4.0) — 1,215 train / 522 val segments |
| recipe | ConvVAE 32×32, latent_dim=128, base_channels=48, dropout=0.1, EMA=0.999, AMP fp16, 300 epochs |
| metrics | F1 = 0.65 at τ=0.30 (best epoch 210) |

## One-time release steps

The asset name (`pop.pt`) and release tag (`weights-v1`) must match the
URL hard-coded in `riffly/weights.py`. If you change either, update the
registry and bump the alias' sha256 in lock-step.

```bash
# 1. Verify the checkpoint hash matches the registry
sha256sum runs/pop_v3/model.pt
# expected: b78a5d29c082510bc1757d5c7bc15050ab2baeceddc99fd99a419a283647cfb9

# 2. Stage the asset under the release filename
cp runs/pop_v3/model.pt runs/pop_v3/pop.pt

# 3. Create the GitHub Release and upload the asset
gh release create weights-v1 \
    runs/pop_v3/pop.pt \
    --title "Pretrained weights v1" \
    --notes "ConvVAE 32x32 trained on ADL Piano MIDI / Pop (CC BY 4.0). F1=0.65 @ tau=0.30."

# 4. Smoke-test the public URL from a clean cache
rm -rf ~/.cache/riffly/weights
uv run python -c "from riffly import Riffly; m = Riffly('pop'); print(m)"
# Should print:  riffly: downloading 'pop' from https://...
# and then:      Riffly(arch='convvae', shape=32x32, params=1,600,001, threshold=0.3)
```

## Updating an existing alias

`pop` is hash-suffixed in the cache (`pop-<sha8>.pt`), so a new checkpoint
lands in a fresh file and users transparently pick it up. The two changes
that have to ship together:

1. Re-upload the asset to the release (or cut a new tag — both work, see
   below). `gh release upload <tag> <file> --clobber` overwrites.
2. Update both `url` and `sha256` in
   [`riffly/weights.py`](riffly/weights.py); bump the package version.

If you change the **tag** (e.g. `weights-v2`) you only need to update
`url`. If you keep the tag and replace the asset, you must update
`sha256`. The integrity check refuses to load a file whose hash doesn't
match the registry, so a stale shasum is a hard failure, not a silent
one — that is the intent.

## Adding a new alias

Edit `WEIGHTS` in `riffly/weights.py` with a new entry:

```python
WEIGHTS["pop-mini"] = {
    "url": "https://github.com/YannickGibson/riffly/releases/download/weights-v1/pop-mini.pt",
    "sha256": "<sha256 of the file>",
    "description": "Smaller ConvVAE 22x32 for CPU inference benchmarks.",
}
```

The dispatch in `Riffly.__init__` resolves any string that (a) is not a
known architecture name (`vae`, `convvae`, `transformer`) and (b) is not a
local file path through the registry.

## Local override

For private / experimental weights, point `Riffly` at the file directly
or pre-seed the cache:

```python
Riffly("path/to/local.pt")            # bypasses the registry entirely
```

The cache directory honours `RIFFLY_HOME` and `XDG_CACHE_HOME`; setting
`RIFFLY_HOME=/tmp/r` puts weights under `/tmp/r/weights/`. Useful for CI
runs that must not litter `~/.cache`.
