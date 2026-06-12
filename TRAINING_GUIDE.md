# Training Guide

Practical guide to preparing data, choosing hyperparameters, reading loss curves, and knowing when to stop training.

---

## Preparing Training Data

### Format

Every training example is an `(input, output)` pair.

```csv
input_text,expected_output
"Translate to French: The weather is nice today","Le temps est agréable aujourd'hui"
"Summarize in one sentence: [long article]","A one-sentence summary."
```

**Rules:**
- Minimum 10 pairs (model will overfit immediately on fewer)
- Both fields must be non-empty strings
- UTF-8 encoding (the loader handles NFKC normalization and strips null bytes)
- No hard limit on rows — more data is almost always better

### Quality checklist

| Check | Why it matters |
|---|---|
| Representative samples | If all inputs are similar, the model won't generalize |
| Correct outputs | A single wrong label can corrupt a cluster of similar examples |
| No duplicates | Duplicate pairs inflate the effective dataset and bias the model |
| Consistent format | Mixing "Translate:" with "Translation:" confuses the model |
| Balanced classes | For classification, >10:1 class ratio usually needs oversampling |

### Recommended dataset sizes

| Task | Minimum | Good | Excellent |
|---|---|---|---|
| Simple classification | 100 | 1,000 | 10,000+ |
| Text generation | 500 | 5,000 | 50,000+ |
| Instruction following | 1,000 | 10,000 | 100,000+ |

---

## Hyperparameter Tuning Guide

### Learning rate — the most important setting

```
Too high: loss oscillates or diverges
          step: 1e-3 ──→ loss goes up and down wildly

Correct:  loss decreases smoothly
          step: 2e-5 ──→ steady descent with slight noise

Too low:  loss barely moves
          step: 1e-7 ──→ flat line, model barely learns
```

**Recommended starting points for fine-tuning:**

| Base model size | Learning rate |
|---|---|
| Small (GPT-2, T5-small) | `2e-5` – `5e-5` |
| Medium (GPT-2-medium) | `1e-5` – `3e-5` |
| Large (7B+: LLaMA, Mistral) | `5e-6` – `2e-5` |

**How to tune:** Start at `2e-5`. If val loss diverges after epoch 1, halve it. If training is too slow, double it.

### Batch size

| Setting | Effect |
|---|---|
| Larger batch | More stable gradient estimates, faster wall-clock time, needs more GPU RAM |
| Smaller batch | Noisier gradients (sometimes helps escape local minima), fits in less RAM |

Use **gradient accumulation** to simulate large batches on limited hardware:
```
effective_batch = batch_size × gradient_accumulation_steps
```
Example: `batch_size=4, gradient_accumulation_steps=8` → effective batch of 32.

**Recommended:** effective batch of 16–32 for fine-tuning.

### Epochs

- Fine-tuning typically needs **3–5 epochs** on a good dataset.
- More epochs → higher risk of overfitting.
- Use early stopping (save best checkpoint, not the final epoch).

### Warmup steps

Warmup prevents large gradient updates at the start before the optimizer has good momentum estimates.

**Rule of thumb:** `warmup_steps = total_steps × 0.06` (6% of training).

### Max sequence length

Set to the 95th percentile of your token lengths, not the maximum. Longer sequences = quadratically more memory.

```bash
# Check average token length via the data_stats endpoint:
curl http://localhost:8000/jobs/1/data_stats
```

---

## Understanding Loss Curves

### Healthy training

```
Loss
│
3 ┤ ●                    train_loss
  │   ●
2 ┤     ●   ○            val_loss
  │       ● ○
1 ┤         ●○
  │           ●○
0 └──────────────── Epoch
```

Both losses decrease together. Validation slightly above training is normal.

### Overfitting

```
Loss
│
3 ┤ ●              ○ ○ ○    val_loss climbing
  │   ●  ○ ○
2 ┤     ●  ●            ← train keeps going down
  │          ●  ●
1 ┤               ● ●
0 └──────────────────── Epoch
```

**Signs:** `val_loss` plateaus or increases while `train_loss` keeps falling.

**Fix:** Use the checkpoint from the epoch with the lowest `val_loss` (already implemented — the trainer saves the best model automatically). Add more data or increase weight decay.

### Underfitting

```
Loss
│
3 ┤ ● ● ● ● ● ●     loss barely moves
  │
2 ┤
  │
0 └──────────────── Epoch
```

**Fix:** Increase learning rate, train longer, use a larger base model.

### Loss spike

```
Loss
│
3 ┤ ●
  │   ●
2 ┤     ●
  │       ●  ←  sudden spike
3 ┤              ● ●   
```

**Cause:** A batch of unusually long/difficult examples, or learning rate too high.
**Fix:** Gradient clipping (already applied at `max_norm=1.0`). If spikes persist, lower the learning rate.

---

## When to Stop Training (Early Stopping)

The system automatically saves a new `ModelVersion` whenever `val_loss` improves. You don't need to stop training manually — just use the best saved version.

**Manual signal to stop:**
1. `val_loss` has not improved for 2+ consecutive epochs — further training is just overfitting.
2. `val_loss` is increasing — the model is memorising training data.
3. The improvement per epoch is < 0.001 — diminishing returns.

**Practical approach:**
- Set `epochs=10` and let early stopping (via best-checkpoint saving) do the work.
- After training, call `GET /jobs/{id}/metrics` and pick the version with the lowest `val_loss`.

---

## MLflow Experiment Tracking

Every training run is logged automatically. View all runs:

```bash
# Start the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open: http://localhost:5000
```

**What to look at:**
- **Parallel coordinates:** see which learning rate + batch size combinations produced the lowest val_loss.
- **Metric curves:** compare val_loss across runs over epochs.
- **Artifacts:** download any saved checkpoint directly from the UI.

---

## Common Issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch too large | Reduce `batch_size`, increase `gradient_accumulation_steps` |
| Loss = `NaN` from epoch 1 | Learning rate too high | Set `learning_rate=1e-5` |
| Val loss never improves | Too little data, LR too low | Add more data; try `learning_rate=5e-5` |
| Training very slow | Batch too small, no GPU | Increase batch size; use `fp16=True` on GPU |
| `KeyError: input_ids`| Tokenizer mismatch | Ensure tokenizer `pad_token` is set (done automatically in trainer) |
