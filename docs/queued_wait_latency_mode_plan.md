# Tritonbench Queued Wait Latency Mode Plan

## Objective
Introduce a new latency measurement mode for TritonBench that mitigates CPU kernel-launch overhead without relying on CUDA graphs by using a Triton wait/signal barrier. This mode should retain cache-clearing behavior, avoid CUDA-specific APIs, and integrate cleanly with existing latency statistics.

## Key Concepts
- **Wait kernel**: A Triton kernel that spins on a device-side flag using `tl.atomic_add(flag_ptr, 0)` until the flag becomes non-zero, ensuring the GPU stalls until all work is enqueued.
- **Signal kernel**: A Triton kernel (or asynchronous device write) that sets the flag once the host has queued all benchmark work.
- **Streams & events**: Use `triton.runtime.driver.active.get_device_interface()` to obtain portable stream/event primitives (no direct `torch.cuda` usage).
- **Watchdog**: Optional host-side timeout using Python `signal` to break potential deadlocks.

## Implementation Steps
1. **Helper kernels**
   - Add `wait_for_flag` and `signal_flag` Triton kernels (single CTA, `num_warps=1`).
   - Ensure they accept a pointer to a 1-element int32 tensor on the active device.

2. **Latency helper**
   - Create `do_bench_wait_signal` alongside existing helpers in `tritonbench/components/do_bench/run.py`.
   - Flow per measurement:
     1. Allocate flag tensor (zero-initialized) and obtain `device_iface.stream()` instances for measurement and signaling.
     2. Launch `wait_for_flag` on the measurement stream.
     3. Record start event, queue repeated benchmark iterations (`cache.zero_()`, grad clears, `fn()`), record end event — all on the measurement stream.
     4. After queuing, launch `signal_flag` on the signaling stream to release the wait.
     5. Synchronize on the end event to collect timing; repeat for `n_retries` and summarize with `_summarize_statistics`.
   - Include optional watchdog (SIGALRM) to abort hangs and clean up signal state afterwards.

3. **CLI integration**
   - Extend `--latency-measure-mode` choices with `queued_wait`.
   - Update help text and any validation logic.
   - Modify `do_bench_wrapper` to dispatch to the new helper when this mode is selected (ignore `--cudagraph` flag in this case).

4. **Cache-clearing compatibility**
   - Reuse `triton.runtime.driver.active.get_empty_cache_for_benchmark()` inside the timed loop to maintain cold-cache measurements.
   - Ensure gradient reset semantics match existing helpers.

5. **Watchdog & cleanup**
   - Use `signal.signal` to install an alarm around the timing loop, cancel the alarm after successful completion.
   - Provide clear error messaging if the watchdog triggers.

6. **Testing / validation**
   - Run `python -m compileall tritonbench/components/do_bench/run.py` to ensure syntax correctness.
   - Optionally execute a small benchmark locally with `--latency-measure-mode queued_wait` to verify runtime behavior (manual step).

## Deliverables
- Updated `tritonbench/components/do_bench/run.py` with new helper and kernels.
- Updated `tritonbench/utils/parser.py` to expose the new CLI option.
- Any supporting documentation comments explaining the new mode.
- Confirmation that compile-time checks succeed.

## Current Status (Deadlock Investigation)
- Implementation of the queued-wait latency mode is in place, but the wait kernel still deadlocks.
- Minimal repro that reproduces the hang:
  ```bash
  python /tmp/test_wait_signal.py
  ```
- The repro uses `_wait_for_flag` and `_signal_flag` from `tritonbench/components/do_bench/run.py` with a host watchdog (SIGALRM) yet still times out, indicating the wait kernel never observes the flag flip.
