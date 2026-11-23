"""Encode live SDXL captions on CUDA using pinned host copies of frozen text encoders.

The originals remain available for validation. GPU copies are released after encoding."""
import contextlib
import copy
import itertools

import torch


class TransientGpuTextEncoders:
    def __init__(self, text_encoders, device, dtype=torch.float16):
        self.device = torch.device(device)
        self.models = []
        self._tensors = []
        self._pinned = []
        for te in text_encoders:
            m = copy.deepcopy(te).to("cpu").to(dtype=dtype)
            m.requires_grad_(False)
            m.eval()
            tensors = list(itertools.chain(m.parameters(), m.buffers()))
            for t in tensors:
                t.data = t.data.pin_memory()
            self.models.append(m)
            self._tensors.append(tensors)
            self._pinned.append([t.data for t in tensors])

    @contextlib.contextmanager
    def on_device(self):
        try:
            for tensors, pinned in zip(self._tensors, self._pinned):
                for t, p in zip(tensors, pinned):
                    t.data = p.to(self.device, non_blocking=True)
            yield self.models
        finally:
            for tensors, pinned in zip(self._tensors, self._pinned):
                for t, p in zip(tensors, pinned):
                    t.data = p
