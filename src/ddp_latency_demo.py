import argparse, os, time
import torch, torch.distributed as dist

def setup(backend):
    # torchrun 会传入以下 env（LOCAL_RANK、RANK、WORLD_SIZE、MASTER_ADDR、MASTER_PORT）
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank(); world = dist.get_world_size()
    return rank, world

def tensor_mb(mb, device):
    numel = (mb * 1024 * 1024) // 4  # float32 4 bytes
    return torch.ones(numel, dtype=torch.float32, device=device)

def benchmark(mb=64, iters=50, delay_ms=0, device="cpu"):
    t = tensor_mb(mb, device)
    # 预热
    for _ in range(5):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if device != "cpu": torch.cuda.synchronize()
    # 正式计时
    start = time.perf_counter()
    for _ in range(iters):
        if delay_ms > 0:
            # 模拟跨地域时延（比如 200ms RTT）
            time.sleep(delay_ms/1000.0)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if device != "cpu": torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=64, help="每次 all_reduce 的 tensor 大小(MB)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--delay_ms", type=float, default=0.0, help="每次通信前注入的延迟(毫秒)")
    args = ap.parse_args()

    # 设备与后端判定：Windows/无多卡 → gloo+CPU；Linux+>=2GPU → nccl+CUDA
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= int(os.environ.get("WORLD_SIZE","1"))
    backend = "nccl" if (use_cuda and os.name != "nt") else "gloo"
    rank, world = setup(backend)

    if use_cuda and backend == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    if rank == 0:
        print(f"[Setup] backend={backend} | world={world} | device={device}")

    # 基线（无延迟）与跨地域延迟（有 delay_ms）对比
    base = benchmark(args.mb, args.iters, delay_ms=0, device=device)
    dist.barrier()
    cross = benchmark(args.mb, args.iters, delay_ms=args.delay_ms, device=device)
    dist.barrier()

    if rank == 0:
        print(f"[Result] tensor={args.mb}MB, iters={args.iters}")
        print(f"  No-delay: {base:.3f}s total  | {base/args.iters*1000:.1f} ms/iter")
        print(f"  +{args.delay_ms:.0f}ms: {cross:.3f}s total | {cross/args.iters*1000:.1f} ms/iter")
        print(f"  Δ per-iter ≈ {(cross-base)/args.iters*1000:.1f} ms  (延迟开销直观化)")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
