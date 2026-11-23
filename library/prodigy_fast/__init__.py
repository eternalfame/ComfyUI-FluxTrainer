"""Batch supported FP32 schedule-free Prodigy updates by optimizer group and tensor shape.

Parameters and optimizer state share storage with the packed tensors. Unsupported settings use the stock optimizer.
Checkpoint loading restores the packs. Keep parameters on their optimizer device after initialization."""
import torch
from pathlib import Path
from types import SimpleNamespace

try:
    from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
except ImportError:
    from .prodigy_plus_schedulefree import ProdigyPlusScheduleFree

SLICE = 11


class FastProdigyPlusScheduleFree(ProdigyPlusScheduleFree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._packs = None
        self._fast_ok = None

    @staticmethod
    def prepare_model_state_for_save(weights):
        # Safetensors removes tensors sharing a storage, even for disjoint packed slices.
        # Copy those slices independently to CPU without adding a CUDA allocation.
        for name, tensor in weights.items():
            if tensor.untyped_storage().nbytes() > tensor.numel() * tensor.element_size():
                weights[name] = tensor.to(device="cpu", copy=True)

    @torch.no_grad()
    def load_checkpoint_model(self, model, input_dir, steps):
        from safetensors import safe_open

        directory = Path(input_dir)
        model_file = directory / "model.safetensors"
        if not model_file.is_file():
            return False
        params = dict(model.named_parameters())
        with safe_open(model_file, framework="pt", device="cpu") as saved:
            saved_keys = set(saved.keys())
            missing = params.keys() - saved_keys
            if not missing:
                # Copy into existing parameters without safetensors inspecting their packed storage.
                model.load_state_dict({name: saved.get_tensor(name) for name in saved_keys})
                return True
            adapter_file = directory.with_name(directory.name.removesuffix("-state") + ".safetensors")
            if not adapter_file.is_file():
                raise RuntimeError(
                    f"Fast Prodigy checkpoint is missing {len(missing)} adapter weights. "
                    f"Recovery requires the matching LoRA export: {adapter_file}"
                )
            with safe_open(adapter_file, framework="pt", device="cpu") as adapter:
                metadata = adapter.metadata() or {}
                if steps is None or metadata.get("ss_steps") != str(steps) or missing - set(adapter.keys()):
                    raise RuntimeError(f"LoRA export does not match the incomplete Fast Prodigy checkpoint: {adapter_file}")
                if any(adapter.get_slice(name).get_shape() != list(params[name].shape) for name in missing):
                    raise RuntimeError(f"LoRA export has incompatible adapter shapes: {adapter_file}")

                optimizer_state = torch.load(directory / "optimizer.bin", map_location="cpu", weights_only=True, mmap=True)
                names = {id(param): name for name, param in params.items()}
                model.load_state_dict({name: saved.get_tensor(name) for name in saved_keys}, strict=False)
                for group, saved_group in zip(self.param_groups, optimizer_state["param_groups"], strict=True):
                    stored = optimizer_state["state"]
                    group_params = {
                        names[id(param)]: stored[param_id]["z"]
                        for param, param_id in zip(group["params"], saved_group["params"], strict=True)
                    }
                    missing_in_group = missing & group_params.keys()
                    if not missing_in_group:
                        continue
                    beta1 = saved_group["betas"][0]
                    to_train = 1 - beta1
                    to_eval = 1 - 1 / beta1
                    target_train = saved_group["train_mode"]
                    # Step exports may have been saved in eval mode while the state is in train
                    # mode. Use the weights that survived serialization to identify the mode.
                    direct_error = converted_error = norm = 0.0
                    for name in group_params.keys() & saved_keys & set(adapter.keys()):
                        weight = saved.get_tensor(name).float()
                        exported = adapter.get_tensor(name).float()
                        converted = weight.lerp(group_params[name].float(), to_eval if target_train else to_train)
                        direct_error += (weight - exported).square().sum().item()
                        converted_error += (converted - exported).square().sum().item()
                        norm += exported.square().sum().item()
                    if not norm or min(direct_error, converted_error) > norm * 1e-4:
                        raise RuntimeError(f"Cannot verify that this LoRA export matches the checkpoint: {adapter_file}")

                    for name in missing_in_group:
                        weight = adapter.get_tensor(name).float()
                        if converted_error < direct_error:
                            weight.lerp_(group_params[name].float(), to_train if target_train else to_eval)
                        params[name].copy_(weight)

        self.log(f"Recovered {len(missing)} missing weights from {adapter_file.name} at the export's precision.")
        return True

    def _supported(self, groups=None):
        for g, current in zip(self.param_groups if groups is None else groups, self.param_groups):
            if (not g["use_schedulefree"] or g["use_speed"] or g["use_adopt"] or g["use_bias_correction"]
                    or g["use_cautious"] or g["use_grams"] or g["use_orthograd"] or g["use_focus"]
                    or g["fused_back_pass"] or g["eps"] is None):
                return False
            for p in current["params"]:
                if p.dtype != torch.float32:
                    return False
        return True

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        self._packs = None
        self._fast_ok = None
        self.parameters_to_process = None

        groups = state_dict["param_groups"]
        states = state_dict["state"]
        param_ids = {param_id for group in groups for param_id in group["params"]}
        complete = all(states.get(param_id) for param_id in param_ids)
        if not states or not self._supported(groups):
            return super().load_state_dict(state_dict)
        if not complete or self._optimizer_load_state_dict_pre_hooks or self._optimizer_load_state_dict_post_hooks:
            # The stock optimizer can initialize only the parameters that receive gradients.
            self._fast_ok = False
            return super().load_state_dict(state_dict)

        # Let PyTorch validate and restore the parameter groups. Leave per-parameter tensors
        # on their saved device until packing so CUDA never holds both copies of the state.
        super().load_state_dict({**state_dict, "state": {key: val for key, val in states.items() if key not in param_ids}})
        for group, saved_group in zip(self.param_groups, groups):
            for param, param_id in zip(group["params"], saved_group["params"]):
                self.state[param] = states[param_id].copy()
        self._packs = self._build_packs()
        self._fast_ok = True

    @torch.no_grad()
    def _build_packs(self):
        packs = []
        for group in self.param_groups:
            by_shape = {}
            for p in group["params"]:
                by_shape.setdefault((tuple(p.shape), p.device), []).append(p)

            for (shape, device), params in by_shape.items():
                n = len(params)
                numel = 1
                for s in shape:
                    numel *= s
                states = [self.state[p] for p in params]
                have = [len(st) > 0 for st in states]
                if any(have) and not all(have):
                    raise RuntimeError("Fast Prodigy: optimizer state is missing for some parameters.")
                resumed = all(have)

                Y = torch.stack([p.detach() for p in params])
                dims = self.factored_dims(shape, factored=group["factored"], min_dim_size_to_factor=32)
                pk = SimpleNamespace(group=group, params=params, shape=shape, n=n, numel=numel,
                                     Y=Y, Z=None, P0=None, S=None, dims=dims, ROW=None, COL=None, V=None)
                sliced_len = len(range(0, numel, SLICE))

                if resumed:
                    pk.Z = torch.stack([st["z"] for st in states]).to(device=device, dtype=torch.float32)
                    if dims is not None:
                        dc, dr = dims
                        for st in states:
                            meta = st.get("exp_avg_sq_metadata")
                            if meta is None or tuple(meta) != (dr, dc):
                                raise RuntimeError("Fast Prodigy: the saved state does not match the factorization for "
                                                   f"parameter shape {shape}. Check the factored setting.")
                        pk.ROW = torch.stack([st["exp_avg_sq_row"] for st in states]).to(device=device, dtype=torch.float32)
                        pk.COL = torch.stack([st["exp_avg_sq_col"] for st in states]).to(device=device, dtype=torch.float32)
                    else:
                        if any("exp_avg_sq" not in st for st in states):
                            raise RuntimeError(f"Fast Prodigy: the saved state has no exp_avg_sq for shape {shape}.")
                        pk.V = torch.stack([st["exp_avg_sq"] for st in states]).to(device=device, dtype=torch.float32)
                    has_p0 = ["p0" in st and "s" in st for st in states]
                    if all(has_p0):
                        p0s = []
                        for st in states:
                            p0 = st["p0"]
                            if p0.ndim == 0:
                                p0 = p0.new_zeros(sliced_len)
                            p0s.append(p0)
                        pk.P0 = torch.stack(p0s).to(device=device, dtype=torch.bfloat16)
                        pk.S = torch.stack([st["s"] for st in states]).to(device=device, dtype=torch.bfloat16)
                    elif any(has_p0):
                        raise RuntimeError("Fast Prodigy: p0 or s is missing for some parameters.")
                else:
                    pk.Z = Y.clone()
                    pk.P0 = Y.reshape(n, -1)[:, ::SLICE].to(torch.bfloat16).contiguous()
                    pk.S = torch.zeros_like(pk.P0)  # s (bf16)
                    if dims is not None:
                        dc, dr = dims
                        row_shape = list(shape)
                        row_shape[dr] = 1
                        col_shape = list(shape)
                        col_shape[dc] = 1
                        pk.ROW = torch.zeros((n, *row_shape), dtype=torch.float32, device=device)
                        pk.COL = torch.zeros((n, *col_shape), dtype=torch.float32, device=device)
                    else:
                        pk.V = torch.zeros_like(Y)

                for i, p in enumerate(params):
                    p.data = Y[i]
                    st = self.state[p]
                    st["z"] = pk.Z[i]
                    if dims is not None:
                        st["exp_avg_sq_row"] = pk.ROW[i]
                        st["exp_avg_sq_col"] = pk.COL[i]
                        st["exp_avg_sq_metadata"] = (dr, dc)
                    else:
                        st["exp_avg_sq"] = pk.V[i]
                    if pk.P0 is not None:
                        st["p0"] = pk.P0[i]
                        st["s"] = pk.S[i]
                packs.append(pk)
        return packs

    def _group_scalars(self, group):
        if not group["train_mode"]:
            raise Exception("Not in train mode!")

        beta1, beta2, beta3 = self.get_betas(group)
        d = group["d"]
        dlr = self.get_dlr(group)

        weight = dlr ** 2
        weight_sum = group["running_weight_sum"] = group.get("weight_sum", 0) + weight
        ckp1 = weight / weight_sum if weight_sum else 0.0
        c = group.get("schedulefree_c", 0)
        if c > 0:
            ckp1 = min(1.0, ckp1 * (1 - beta1) * c)
        xy_step = 1 - beta1 * (1 - ckp1)
        group["effective_lr"] = group["lr"] * xy_step

        k, ps = group["k"], group["prodigy_steps"]
        return SimpleNamespace(
            beta1=beta1, beta2=beta2, beta3=beta3, d=d, dlr=dlr, ckp1=float(ckp1), xy_step=xy_step,
            decay=self.get_weight_decay(group), prodigy=(ps <= 0 or k < ps),
            d_update=(d ** 2) / (group["d0"] ** 0.5),
        )

    @torch.no_grad()
    def _step_pack(self, pk, sc):
        group, n, shape = pk.group, pk.n, pk.shape
        G = torch.stack([p.grad for p in pk.params])

        if sc.prodigy:
            run_num, run_den = self.get_running_values_for_group(group)
            Gs = G.reshape(n, -1)[:, ::SLICE]
            Zs = pk.Z.reshape(n, -1)[:, ::SLICE]
            x0 = pk.P0 - Zs
            run_num.add_((Gs * x0).sum(dim=1).sum(), alpha=sc.d_update)
            pk.S.mul_(sc.beta3).add_(Gs, alpha=sc.d_update)
            run_den.add_(pk.S.abs().sum(dim=1).float().sum())
            del x0
        elif pk.S is not None:
            pk.P0 = pk.S = None
            for p in pk.params:
                self.state[p].pop("p0", None)
                self.state[p].pop("s", None)

        a = (1 - sc.beta2) * sc.d ** 2
        if pk.dims is not None:
            dc, dr = pk.dims
            pk.ROW.mul_(sc.beta2).add_(G.norm(dim=dr + 1, keepdim=True).square_().mul_(1 / shape[dr]), alpha=a)
            pk.COL.mul_(sc.beta2).add_(G.norm(dim=dc + 1, keepdim=True).square_().mul_(1 / shape[dc]), alpha=a)
            reduce_dc = (dc - 1 if dc > dr else dc) + 1
            row_col_mean = pk.ROW.mean(dim=reduce_dc, keepdim=True).add_(1e-30)
            denom = (pk.ROW.div(row_col_mean) * pk.COL).sqrt_()
            del row_col_mean
        else:
            pk.V.mul_(sc.beta2).addcmul_(G, G, value=a)
            denom = pk.V.sqrt()

        G.mul_(sc.d)
        denom.add_(sc.d * group["eps"])
        G.div_(denom)
        del denom

        if group["use_stableadamw"]:
            rms = G.flatten(1).norm(dim=1).div(pk.numel ** 0.5).clamp_min(1.0).rsqrt()
            G.mul_(rms.view(-1, *([1] * len(shape))))

        Y, Z = pk.Y, pk.Z
        y_wd = None
        if sc.decay != 0:
            if group["weight_decay_by_lr"]:
                G.add_(Y, alpha=sc.decay)
            else:
                y_wd = Y.clone()
        Y.lerp_(Z, sc.ckp1)
        Z.sub_(G, alpha=sc.dlr)
        Y.sub_(G, alpha=sc.dlr * sc.xy_step)
        if y_wd is not None:
            Z.sub_(y_wd, alpha=sc.decay)
            Y.sub_(y_wd, alpha=sc.decay * sc.xy_step)
            del y_wd
        del G

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, closure=None):
        if self._fast_ok is None:
            self._fast_ok = self._supported()
            if not self._fast_ok:
                self.log("Fast Prodigy: using the stock step for this configuration.")
        if not self._fast_ok:
            return super().step(closure)

        self.try_unhook_kohya_fbp()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._packs is None:
            self._packs = self._build_packs()

        if any(p.grad is None for pk in self._packs for p in pk.params):
            super().step()
            return loss

        self.on_start_step()
        scalars = {}
        for pk in self._packs:
            sc = scalars.get(id(pk.group))
            if sc is None:
                sc = scalars[id(pk.group)] = self._group_scalars(pk.group)
            self._step_pack(pk, sc)

        self.parameters_to_process = 1
        self.on_end_step()
        return loss
