JAX implementation of elastic latent recurrence, a looped-block training recipe (arXiv:2610.12582).

```bash
pip install -e .
python -m pytest
```

This is a toy looped block with vocab 32 and width 32, not the 4B/30B hybrid, and rank overrides are a batch split that forces a fraction of examples to one loop. It does not use Nemotron data; a 30-step CPU Adam smoke run on synthetic tokens has loss 5.715847015380859 at step 0 and loss 5.236719608306885 at step 30.
