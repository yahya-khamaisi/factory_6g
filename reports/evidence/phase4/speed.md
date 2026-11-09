# PyTorch stack speed (Phase 4)

Measured 9 Nov 2026 on the 10-core arm64 Mac (Docker sees 10 CPUs), with two
TensorFlow Phase 1 lanes running at the same time, so absolute times are
inflated for both stacks. Both images are native arm64.

## Whole runs

| Run C (estimators, Rayleigh, 200 batches) | Wall clock |
|:--|--:|
| TensorFlow, Sionna 1.2.1 (Phase 1 seeds 42–46) | 1 h 30 min – 1 h 57 min |
| PyTorch, Sionna 2.2, seed 42, 10 threads, eager decoder | 7 h 06 min |

## Per batch (Run C config, 55 method × Eb/N0 points)

| Setting | Seconds per batch |
|:--|--:|
| PyTorch, 10 threads | about 145 |
| PyTorch, 4 threads | about 95–100 |
| PyTorch, 2 threads | about 110 |
| PyTorch, 4 threads, `torch.compile` decoder | about 70–74 (first batch adds about 100 s of compiling) |
| TensorFlow lane running alongside | about 41 |

## Where the time goes

`cProfile` of one PyTorch batch (4 threads): 94 s in `run_batch`, of which 72 s
is the LDPC decoder (20 belief-propagation iterations × 55 calls). Inside it the
time is in elementwise and indexing ops: the boxplus-phi check-node update
(`log`, `exp`, `clamp`), and the gathers and scatters between variable and
check nodes. The neural estimator's convolutions take about 11 s and LMMSE
equalisation about 5 s.

On the same machine, `torch.log` and `torch.exp` on 2 × 10⁶ floats take about
twice as long as `tf.math.log` and `tf.math.exp`, and TensorFlow runs the
decoder as one traced graph (`tf.function`), where the PyTorch port runs it
op by op.

## What changed

- `F6G_COMPILE_DECODER=1` wraps the decoder in `torch.compile` (inductor; the
  image now has gcc/g++). On a 3-batch check its BER matches the eager decoder
  except for floating-point ordering (adaptive estimator: 11 225 against
  11 224.9 mean bit errors per point), the same caveat the TensorFlow graph
  has.
- The gate script uses 4 threads and the compiled decoder by default
  (`OMP_NUM_THREADS`, `F6G_COMPILE_DECODER` override it). The Run C gate
  above ran eager with 10 threads; the Run B gate runs compiled with 4.

The PyTorch stack stays about 1.7× slower than TensorFlow per batch after
these changes. Changing the check-node rule (for example min-sum) would be
faster but changes the results, so it is not used.
