"""Run with python tests/test_swap.py . from the repository root. CUDA is optional."""
import sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else '..'
import itertools, re, sys, torch
from torch import nn

src = open(ROOT + '/library/sdxl_original_unet.py', encoding='utf-8').read()
a = src.index('_ARENA_ALIGN = 512'); b = src.index('def _move_to_device')
ns = {'torch': torch, 'itertools': itertools, 'nn': nn}
real_empty = torch.empty
def empty(*args, **kw):
    kw.pop('pin_memory', None); return real_empty(*args, **kw)
torch.empty = empty
exec(src[a:b], ns)
SwapPack, _SwapArena = ns['SwapPack'], ns['_SwapArena']

def make_block(dtype, seed):
    g = torch.Generator().manual_seed(seed)
    m = nn.ModuleList([nn.Linear(37, 53), nn.LayerNorm(53), nn.Linear(53, 11)])
    for p in m.parameters():
        p.data = torch.randn(p.shape, generator=g).to(dtype); p.requires_grad_(False)
    m.register_buffer('idx', torch.arange(7))
    return m

def snap(m): return [t.data.clone() for t in itertools.chain(m.parameters(), m.buffers())]
def same(m, ref): return all(torch.equal(t.data, r) for t, r in zip(itertools.chain(m.parameters(), m.buffers()), ref))
def fwd(m, x): return m[2](m[1](m[0](x)))

dev = torch.device('cpu')
ns['_HOST_CHUNK_BYTES'] = 1024
arena = _SwapArena()
A, B = make_block(torch.float32, 1), make_block(torch.float32, 2)
ref_A, ref_B = snap(A), snap(B)
x = torch.randn(4, 37)
yA, yB = fwd(A, x), fwd(B, x)
pa, pb = SwapPack(A, arena), SwapPack(B, arena)
arena.ensure(max(pa.total, pb.total), dev)
ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}")

check("Packing preserves weight values", same(A, ref_A) and same(B, ref_B))
check("CPU weights share the chunk storage", len(pa.cpu_chunks) > 1 and all(
    any(chunk.data_ptr() <= t.data_ptr() < chunk.data_ptr() + chunk.numel() for _, chunk in pa.cpu_chunks)
    for t in itertools.chain(A.parameters(), A.buffers())))
pa.load(dev)
check("Loaded A preserves weights and forward output", same(A, ref_A) and torch.allclose(fwd(A, x), yA))
check("Loaded A shares arena storage", all(arena.tensor.data_ptr() <= t.data_ptr() < arena.tensor.data_ptr() + arena.tensor.numel() for t in itertools.chain(A.parameters(), A.buffers())))
pa.offload()
check("Offloaded A preserves weights", same(A, ref_A))
pb.load(dev)
check("B reuses the arena and preserves weights and output", same(B, ref_B) and torch.allclose(fwd(B, x), yB))
pb.offload(); pa.load(dev)
check("A can reload after B", same(A, ref_A) and torch.allclose(fwd(A, x), yA))
pa.load(dev)
check("Repeated load is idempotent", same(A, ref_A))
pa.offload(); pa.offload()
check("Repeated offload is idempotent", same(A, ref_A))
A.to(torch.float64)
pa.offload()
check("Offload preserves an externally changed dtype", all(t.dtype == torch.float64 for t in A.parameters()))
pa.load(dev)
ref64 = [r.to(torch.float64) if r.is_floating_point() else r for r in ref_A]
check("Load rebuilds the buffer for the changed dtype", same(A, ref64) and all(t.dtype == torch.float64 for t in A.parameters()))
pa.offload()
check("Offload preserves the rebuilt weights", same(A, ref64))
sys.exit(0 if ok else 1)
