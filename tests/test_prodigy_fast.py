"""Compare the fast and stock optimizers with python -m tests.test_prodigy_fast."""
import torch

try:
    from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
except ImportError:
    raise SystemExit("Install prodigy-plus-schedule-free to run this comparison")
from library.prodigy_fast import FastProdigyPlusScheduleFree

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

SHAPES = [(16, 1280), (1280, 16), (16, 640), (640, 16), (16, 2048), (5120, 16), (16, 1280), (1280, 16), (16, 640), (1280, 16)]


def make_params():
    base = [torch.randn(s, device=dev) * (0.02 if i % 2 == 0 else 0.0) for i, s in enumerate(SHAPES)]
    a = [torch.nn.Parameter(t.clone()) for t in base]
    b = [torch.nn.Parameter(t.clone()) for t in base]
    return a, b


def run(label, steps=40, none_grad_at=None, roundtrip_at=None, **kw):
    pa, pb = make_params()
    opt_a = ProdigyPlusScheduleFree(pa, lr=1.0, **kw)
    opt_b = FastProdigyPlusScheduleFree(pb, lr=1.0, **kw)
    worst_p = worst_d = 0.0
    for step in range(steps):
        gs = [torch.randn(s, device=dev) * 0.01 for s in SHAPES]
        for p, q, g in zip(pa, pb, gs):
            p.grad, q.grad = g.clone(), g.clone()
        if step == none_grad_at:
            pa[3].grad = None
            pb[3].grad = None
        opt_a.step()
        opt_b.step()
        if step == roundtrip_at:
            opt_a.eval(); opt_b.eval()
            for p, q in zip(pa, pb):
                worst_p = max(worst_p, ((p - q).abs().max() / (p.abs().max() + 1e-12)).item())
            opt_a.train(); opt_b.train()
        for p, q in zip(pa, pb):
            worst_p = max(worst_p, ((p - q).abs().max() / (p.abs().max() + 1e-12)).item())
        da, db = opt_a.param_groups[0]["d"], opt_b.param_groups[0]["d"]
        worst_d = max(worst_d, abs(da - db) / max(abs(da), 1e-30))
    print(f"{label:34s} max rel diff params {worst_p:.2e} | d {worst_d:.2e} | d_a={da:.4e} d_b={db:.4e}")


if __name__ == "__main__":
    run("default")
    run("eval()/train() round trip", roundtrip_at=10)
    run("weight_decay=0.01", weight_decay=0.01)
    run("prodigy_steps=10 (freeze)", prodigy_steps=10)
    run("use_stableadamw=False", use_stableadamw=False)
    run("one grad is None at step 7", none_grad_at=7)
