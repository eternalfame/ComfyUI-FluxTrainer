"""Run with python tests/test_resume.py . from the repository root. CUDA is optional."""
import sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else '..'
import copy, importlib.util, tempfile, torch
from pathlib import Path
from accelerate import Accelerator
from safetensors.torch import save_file
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

spec = importlib.util.spec_from_file_location("prodigy_fast", ROOT + "/library/prodigy_fast/__init__.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
Fast = mod.FastProdigyPlusScheduleFree

torch.manual_seed(0)
torch.set_num_threads(1)
SHAPES = [(16, 1280), (1280, 16), (16, 640), (640, 16), (16, 1280), (1280, 16), (16, 2048), (5120, 16)]
def params():
    return [torch.nn.Parameter(torch.randn(s) * (0.02 if i % 2 == 0 else 0.0)) for i, s in enumerate(SHAPES)]
def grads(): return [torch.randn(s) * 0.01 for s in SHAPES]
def feed(ps, gs):
    for p, g in zip(ps, gs): p.grad = g.clone()

ok = True
def rel(a, b): return max(((x - y).abs().max() / (x.abs().max() + 1e-12)).item() for x, y in zip(a, b))

for label, kw in [("default", {}), ("prodigy_steps=12", {"prodigy_steps": 12}), ("weight_decay", {"weight_decay": 0.01})]:
    p0 = params()
    ps = [torch.nn.Parameter(p.detach().clone()) for p in p0]
    opt = ProdigyPlusScheduleFree(ps, lr=1.0, **kw)
    gseq = [grads() for _ in range(20)]
    for i in range(10):
        feed(ps, gseq[i]); opt.step()
    saved_state = copy.deepcopy(opt.state_dict()); saved_params = [p.detach().clone() for p in ps]

    def resume(cls):
        q = [torch.nn.Parameter(t.clone()) for t in saved_params]
        o = cls(q, lr=1.0, **kw)
        o.load_state_dict(copy.deepcopy(saved_state))
        return q, o
    qa, oa = resume(ProdigyPlusScheduleFree)
    qb, ob = resume(Fast)
    for i in range(10, 20):
        feed(qa, gseq[i]); oa.step(); feed(qb, gseq[i]); ob.step()
        if i == 14:
            oa.eval(); ob.eval(); oa.train(); ob.train()
    dp = rel(qa, qb); da, db = oa.param_groups[0]['d'], ob.param_groups[0]['d']
    dd = abs(da - db) / max(abs(da), 1e-30)
    good = dp < 1e-4 and dd < 1e-3
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  resume [{label}]: params rel diff {dp:.2e}, d rel diff {dd:.2e}")


def accelerated_round_trip():
    model = torch.nn.ParameterList(params())
    stock = Fast(model.parameters(), lr=1.0)
    accelerator = Accelerator(cpu=True)
    model, opt = accelerator.prepare(model, stock)
    def prepare_model_state(models, weights, output_dir):
        for weights_dict in weights:
            stock.prepare_model_state_for_save(weights_dict)

    accelerator.register_save_state_pre_hook(prepare_model_state)
    feed(model, grads())
    opt.step()
    opt.eval()

    with tempfile.TemporaryDirectory() as checkpoint:
        accelerator.save_state(checkpoint)
        opt.train()
        next_grads = grads()
        feed(model, next_grads)
        opt.step()
        expected = [param.detach().clone() for param in model]
        expected_state = copy.deepcopy(opt.state_dict())
        accelerator.free_memory()

        # Reload into a fresh optimizer, then reload again after it already has packs.
        resumed_model = torch.nn.ParameterList(params())
        resumed = Fast(resumed_model.parameters(), lr=1.0)
        resumed_model, wrapped = accelerator.prepare(resumed_model, resumed)

        def load_model(models, input_dir):
            if resumed.load_checkpoint_model(models[0], input_dir, None):
                models.pop()

        accelerator.register_load_state_pre_hook(load_model)
        for _ in range(2):
            accelerator.load_state(checkpoint)
            assert not resumed.param_groups[0]["train_mode"]
            assert all(pack.group is resumed.param_groups[0] for pack in resumed._packs)
            assert all(resumed.state[param]["s"].dtype == torch.bfloat16 for param in resumed_model)
            wrapped.train()
            feed(resumed_model, next_grads)
            wrapped.step()
            assert all(torch.equal(actual, target) for actual, target in zip(resumed_model, expected))
            loaded = wrapped.state_dict()
            for key, state in expected_state["state"].items():
                for field, value in state.items():
                    other = loaded["state"][key][field]
                    assert torch.equal(value, other) if torch.is_tensor(value) else value == other
            assert loaded["param_groups"][0]["k"] == expected_state["param_groups"][0]["k"]
    print("PASS  Accelerate save/load, eval-mode checkpoint, and reload with live packs")


accelerated_round_trip()


def legacy_round_trip():
    model = torch.nn.ParameterList(params())
    optimizer = Fast(model.parameters(), lr=1.0, d0=0.01)
    feed(model, grads())
    optimizer.step()
    expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "adapter-step00001-state"
        checkpoint.mkdir()
        torch.save(optimizer.state_dict(), checkpoint / "optimizer.bin")
        save_file({"0": expected["0"]}, checkpoint / "model.safetensors")
        optimizer.eval()
        save_file(
            {name: tensor.clone() for name, tensor in model.state_dict().items()},
            checkpoint.with_name("adapter-step00001.safetensors"),
            metadata={"ss_steps": "1"},
        )
        resumed = torch.nn.ParameterList(params())
        resumed_optimizer = Fast(resumed.parameters(), lr=1.0, d0=0.01)
        assert resumed_optimizer.load_checkpoint_model(resumed, checkpoint, 1)
        for name, param in resumed.state_dict().items():
            torch.testing.assert_close(param, expected[name], atol=1e-8, rtol=1e-6)
    print("PASS  Recover legacy missing weights from the matching eval-mode LoRA export")


legacy_round_trip()
sys.exit(0 if ok else 1)
