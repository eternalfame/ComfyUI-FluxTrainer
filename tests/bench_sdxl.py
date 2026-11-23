"""Time SDXL U-Net LoRA updates on nine portraits at 512 or 1024 pixels."""

import argparse
from contextlib import nullcontext
import hashlib
import importlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Local SDXL checkpoint")
    parser.add_argument("--comfy-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--trainer-root", type=Path, default=ROOT, help="Trainer checkout or git archive")
    parser.add_argument("--resolution", type=int, choices=(512, 1024), default=1024)
    parser.add_argument("--shuffle-captions", action="store_true", help="Shuffle caption tags and encode them on every batch")
    parser.add_argument("--capture-update", action="store_true", help="Save the first forward inputs, prediction, unscaled gradients, and LoRA deltas")
    parser.add_argument("--detach-latents", action="store_true", help="Test dropping the input-latent gradient before the U-Net")
    parser.add_argument("--output-only", action="store_true", help="Train LoRA adapters only in the U-Net output blocks")
    parser.add_argument("--initial-weights-from", type=Path, help="Initialize shared adapter tensors from a --capture-update run")
    parser.add_argument("--sample-adapters", action="store_true", help="Save the evaluated adapter and render two fixed-seed portraits")
    parser.add_argument("--preview-adapter", type=Path, action="append", help="Render a saved adapter without training; repeat to compare adapters")
    parser.add_argument("--preview-prompt", action="append", help="Override the fixed preview prompts; repeat for multiple prompts")
    parser.add_argument("--compare-time-to", type=Path, help="Extend training to approximately match a reference result.json's timed duration")
    checkpoint_scope = parser.add_mutually_exclusive_group()
    checkpoint_scope.add_argument("--uncheckpoint-middle-attention", action="store_true", help="Disable checkpointing for the middle block's transformers")
    checkpoint_scope.add_argument(
        "--uncheckpoint-deep-attention", nargs="?", type=int, choices=(1, 2), const=2, default=0,
        help="Disable checkpointing for the middle and one or two deepest input transformer groups",
    )
    parser.add_argument("--keep-middle-resident", action="store_true", help="Exclude the middle U-Net block from swapping")
    parser.add_argument("--profile", action="store_true", help="Save a Chrome trace of the timed updates")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--blocks-to-swap", type=int, default=12)
    parser.add_argument("--no-swap-arena", action="store_true")
    parser.add_argument("--attention", choices=("xformers", "sdpa"), default="xformers")
    parser.add_argument("--clean-every", type=int, default=10)
    parser.add_argument("--latent-cache-dir", type=Path, default=OUTPUT / "sdxl_latent_cache")
    parser.add_argument("--output", type=Path, default=OUTPUT / "sdxl_bench")
    cli = parser.parse_args()
    if not cli.checkpoint.is_file():
        parser.error("checkpoint must be a local file")
    if cli.steps < 1 or cli.warmup < 0 or cli.clean_every < 0 or not 0 <= cli.blocks_to_swap <= 20:
        parser.error("invalid step count, cleanup interval, or block count")
    if cli.keep_middle_resident and not cli.blocks_to_swap:
        parser.error("--keep-middle-resident requires block swapping")
    if cli.compare_time_to and (cli.profile or not cli.sample_adapters):
        parser.error("--compare-time-to requires --sample-adapters and unprofiled timings")
    reference_seconds = None
    if cli.compare_time_to:
        reference = json.loads(cli.compare_time_to.read_text(encoding="utf-8"))
        reference_seconds = reference["timed_seconds"]
        if reference["timed_steps"] != cli.steps or reference["warmup_steps"] != cli.warmup:
            parser.error("time comparison requires the same initial timed and warm-up step counts")

    trainer_root = cli.trainer_root.resolve()
    if not (trainer_root / "sdxl_train_network.py").is_file():
        parser.error("trainer root must contain sdxl_train_network.py")
    archive_revision = trainer_root / ".benchmark-revision"
    if archive_revision.is_file():
        revision = archive_revision.read_text(encoding="utf-8").strip()
    else:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=trainer_root, capture_output=True, text=True)
        revision = git.stdout.strip() if git.returncode == 0 else None

    import torch
    from PIL import Image
    from transformers import CLIPTokenizer

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if torch.distributed.is_initialized():
        parser.error("run directly in a single process")

    files = []
    dataset = OUTPUT / "datasets" / ("sdxl_baseline_512" if cli.resolution == 512 else "sdxl_baseline")
    resolution = (cli.resolution, cli.resolution)
    for index in range(1, 10):
        image = dataset / f"portrait_{index:02d}.png"
        caption = image.with_suffix(".txt")
        with Image.open(image) as tile:
            if tile.size != resolution or tile.mode != "RGB":
                parser.error(f"expected a {cli.resolution}×{cli.resolution} RGB image in {image}")
        if not caption.read_text(encoding="utf-8").strip():
            parser.error(f"empty caption in {caption}")
        files.extend((image, caption))
    if set(dataset.glob("*.png")) != set(files[::2]):
        parser.error("the benchmark dataset must contain exactly the nine reference PNGs")

    dataset_hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    checkpoint = cli.checkpoint.resolve()
    checkpoint_stat = checkpoint.stat()
    cache_inputs = {
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "images": {image.name: dataset_hashes[image.name] for image in files[::2]},
        "resolution": resolution,
        "vae_dtype": "fp16",
        "vae_batch_size": 1,
        "vae_attention": "xformers" if cli.attention == "xformers" else "default",
        "torch": str(torch.__version__),
        "diffusers": version("diffusers"),
        "xformers": version("xformers") if cli.attention == "xformers" else None,
        "seed": 42,
    }
    cache_key = hashlib.sha256(json.dumps(cache_inputs, sort_keys=True).encode()).hexdigest()
    cached_dataset = cli.latent_cache_dir.resolve() / f"{cli.resolution}-{cache_key[:20]}"
    cached_dataset.mkdir(parents=True, exist_ok=True)
    for file in files[::2]:
        staged = cached_dataset / file.name
        if not staged.is_file() or hashlib.sha256(staged.read_bytes()).hexdigest() != dataset_hashes[file.name]:
            shutil.copyfile(file, staged)
    # The trainer expects caption sidecars beside the images.
    for file in files[1::2]:
        staged = cached_dataset / file.name
        if not staged.is_file() or staged.read_bytes() != file.read_bytes():
            shutil.copyfile(file, staged)
    latent_files = list(cached_dataset.glob("*_sdxl.npz"))

    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    comfy_root = cli.comfy_root.resolve()
    sys.path[:0] = [str(comfy_root), str(trainer_root.parent)]
    sys.argv[1:] = []  # ComfyUI imports parse the host process command line.
    sdxl = importlib.import_module(f"{trainer_root.name}.sdxl_train_network")

    tokenizer = CLIPTokenizer.from_pretrained(str(comfy_root / "comfy" / "sd1_tokenizer"), local_files_only=True)
    tokenizer.model_max_length = 77
    tokenizer_root = output / "tokenizers"
    for model_id in (sdxl.strategy_sdxl.TOKENIZER1_PATH, sdxl.strategy_sdxl.TOKENIZER2_PATH):
        tokenizer.save_pretrained(tokenizer_root / model_id.replace("/", "_"))

    dataset_config = output / "dataset.json"
    dataset_config.write_text(json.dumps({
        "general": {"caption_extension": ".txt", "shuffle_caption": cli.shuffle_captions},
        "datasets": [{
            "resolution": list(resolution),
            "batch_size": 1,
            "enable_bucket": False,
            "subsets": [{"image_dir": str(cached_dataset), "num_repeats": 1}],
        }],
    }, indent=2), encoding="utf-8")

    training_args = [
        "--pretrained_model_name_or_path", str(cli.checkpoint.resolve()),
        "--dataset_config", str(dataset_config),
        "--tokenizer_cache_dir", str(tokenizer_root),
        "--output_dir", str(output),
        "--network_module", ".networks.lora",
        "--network_train_unet_only", "--network_dim", "16", "--network_alpha", "16",
        "--mixed_precision", "fp16", "--gradient_checkpointing",
        "--cache_latents", "--cache_latents_to_disk",
        "--optimizer_type", "ProdigyPlusScheduleFree", "--learning_rate", "1",
        "--lr_scheduler", "constant", "--max_grad_norm", "0", "--min_snr_gamma", "5",
        "--seed", "42", "--max_data_loader_n_workers", "0",
        "--max_train_steps", str(cli.warmup + cli.steps * (4 if cli.compare_time_to else 1)),
        "--blocks_to_swap", str(cli.blocks_to_swap),
        f"--{cli.attention}",
    ]
    if not cli.shuffle_captions:
        training_args.append("--cache_text_encoder_outputs")
    if not cli.no_swap_arena and cli.blocks_to_swap:
        training_args.append("--swap_arena")
    training_parser = sdxl.train_network.setup_parser()
    if hasattr(training_parser.parse_args([]), "clean_memory_every_n_steps"):
        training_args.extend(("--clean_memory_every_n_steps", str(cli.clean_every)))
    args = training_parser.parse_args(training_args)
    args.output_only_lora = cli.output_only
    args.keep_middle_resident = cli.keep_middle_resident
    trainer = sdxl.SdxlNetworkTrainer()
    latent_images_encoded = 0
    get_latent_strategy = trainer.get_latents_caching_strategy

    def count_latent_encodes(args):
        strategy = get_latent_strategy(args)
        cache_batch = strategy.cache_batch_latents

        def encode_batch(vae, image_infos, *options, **kwargs):
            nonlocal latent_images_encoded
            cache_batch(vae, image_infos, *options, **kwargs)
            latent_images_encoded += len(image_infos)

        strategy.cache_batch_latents = encode_batch
        return strategy

    trainer.get_latents_caching_strategy = count_latent_encodes
    encode_seconds = []
    encode_devices = set()
    if cli.shuffle_captions:
        get_strategy = trainer.get_text_encoding_strategy

        def timed_strategy(args):
            strategy = get_strategy(args)
            encode = strategy.encode_tokens

            def timed_encode(tokenize_strategy, models, tokens_list):
                torch.cuda.synchronize()
                start = perf_counter()
                outputs = encode(tokenize_strategy, models, tokens_list)
                torch.cuda.synchronize()
                encode_seconds.append(perf_counter() - start)
                encode_devices.update(str(model.device) for model in models[:2])
                return outputs

            strategy.encode_tokens = timed_encode
            return strategy

        trainer.get_text_encoding_strategy = timed_strategy
    torch.cuda.synchronize()
    start = perf_counter()
    loop = trainer.init_train(args)
    torch.cuda.synchronize()
    init_seconds = perf_counter() - start
    unet = trainer.accelerator.unwrap_model(trainer.unet)
    swapped_indices = sorted(getattr(unet, "swap_block_indices", ()))
    if cli.keep_middle_resident:
        resident_device = next(unet.time_embed.parameters()).device
        middle_on_device = all(param.device == resident_device for param in unet.middle_block.parameters())
        if len(unet.input_blocks) in swapped_indices or resident_device.type != "cuda" or not middle_on_device:
            raise RuntimeError("middle block did not remain on the accelerator device")
    uncheckpointed_layers = []
    if cli.uncheckpoint_middle_attention or cli.uncheckpoint_deep_attention:
        selected_blocks = {"middle_block": unet.middle_block}
        if cli.uncheckpoint_deep_attention:
            for index in range(len(unet.input_blocks) - cli.uncheckpoint_deep_attention, len(unet.input_blocks)):
                selected_blocks[f"input_blocks.{index}"] = unet.input_blocks[index]
        transformer_block = sdxl.sdxl_model_util.sdxl_original_unet.BasicTransformerBlock
        for block_name, block in selected_blocks.items():
            for name, module in block.named_modules():
                if isinstance(module, transformer_block) and module.gradient_checkpointing:
                    module.gradient_checkpointing = False
                    uncheckpointed_layers.append(f"{block_name}.{name}")
        if not uncheckpointed_layers:
            raise RuntimeError("no checkpointed transformers were found in the selected blocks")
    if not hasattr(trainer, "current_real_epoch"):
        trainer.comfy_pbar = sdxl.train_network.ProgressBar(cli.warmup + cli.steps)
    trainer.optimizer_train_fn()
    # Encoding samples from the VAE posterior. Start training with the same RNG on a cache hit or miss.
    sdxl.train_network.set_seed(args.seed)
    optimizer_updates = 0
    optimizer = trainer.optimizer.optimizer
    network = trainer.accelerator.unwrap_model(trainer.network)
    parameters = dict(network.named_parameters())
    if cli.output_only and (not parameters or any(not name.startswith("lora_unet_output_blocks_") for name in parameters)):
        raise RuntimeError("output-only run has an unexpected adapter layout")
    if cli.initial_weights_from:
        reference_weights = torch.load(cli.initial_weights_from, map_location="cpu", weights_only=True)
        with torch.no_grad():
            for name, param in parameters.items():
                param.copy_(reference_weights[name])

    def sample(steps, prompts):
        args.sample_sampler = "euler"
        args.sample_prompts = prompts
        return trainer.sample_images(
            trainer.accelerator, args, None, steps, trainer.accelerator.device, trainer.vae,
            trainer.tokenizers, trainer.text_encoder, trainer.unet,
        )

    preview_prompts = cli.preview_prompt or [
        "1girl, solo, sdxlref, portrait, looking_at_viewer, upper_body, gray_background, "
        "simple_background, realistic --w 512 --h 512 --s 16 --d 1401 --l 5 "
        "--n camera, holding_camera, window, architecture, face_out_of_frame",
        "1girl, solo, sdxlref, portrait, from_side, looking_back, outdoors, sunset, realistic "
        "--w 512 --h 512 --s 16 --d 1402 --l 5 --n camera, holding_camera, face_out_of_frame",
    ]
    if cli.preview_adapter:
        unet.eval()
        network.eval()
        initial = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
        for index, adapter in enumerate(cli.preview_adapter, start=1):
            network.load_state_dict(initial)
            incompatible = network.load_weights(str(adapter.resolve()))
            if incompatible.unexpected_keys:
                raise RuntimeError(f"adapter contains unexpected tensors: {incompatible.unexpected_keys}")
            args.output_name = f"{index:02d}-{adapter.stem}"
            sample(0, preview_prompts)
        return

    if cli.capture_update or cli.detach_latents:
        call_unet = trainer.call_unet

        def inspect_unet(args, accelerator, unet, latents, timesteps, text_conds, batch, dtype, **kwargs):
            if cli.detach_latents and not cli.output_only:
                latents = latents.detach()
            prediction = call_unet(args, accelerator, unet, latents, timesteps, text_conds, batch, dtype, **kwargs)
            if cli.capture_update and trainer.global_step == 0:
                torch.save({
                    "latents": latents.detach().cpu(),
                    "timesteps": timesteps.cpu(),
                    "text_conds": [cond.detach().cpu() for cond in text_conds],
                    "sizes": {key: batch[key].cpu() for key in ("original_sizes_hw", "crop_top_lefts", "target_sizes_hw")},
                    "prediction": prediction.detach().cpu(),
                }, output / "forward.pt")
            return prediction

        trainer.call_unet = inspect_unet

    if cli.capture_update:
        predict = trainer.get_noise_pred_and_target

        def capture_target(*args, **kwargs):
            result = predict(*args, **kwargs)
            if trainer.global_step == 0:
                torch.save(result[1].detach().cpu(), output / "target.pt")
            return result

        trainer.get_noise_pred_and_target = capture_target
        initial = {name: param.detach().cpu().clone() for name, param in parameters.items()}

        def capture_gradients(optimizer, args, kwargs):
            if trainer.global_step == 0:
                torch.save({
                    name: None if param.grad is None else param.grad.detach().cpu()
                    for name, param in parameters.items()
                }, output / "gradients.pt")

        def capture_deltas(optimizer, args, kwargs):
            if trainer.global_step == 0:
                torch.save({
                    name: param.detach().cpu() - initial[name] for name, param in parameters.items()
                }, output / "deltas.pt")

        optimizer.register_step_pre_hook(capture_gradients)
        optimizer.register_step_post_hook(capture_deltas)
        torch.save(initial, output / "initial.pt")

    def count_update(optimizer, args, kwargs):
        nonlocal optimizer_updates
        optimizer_updates += 1

    optimizer.register_step_post_hook(count_update)

    def train_until(target):
        while trainer.global_step < target:
            previous = trainer.global_step
            loop(target, trainer.current_epoch.value)
            if trainer.global_step == previous:
                raise RuntimeError("training loop made no optimizer progress")

    train_until(cli.warmup)
    warmup_updates = optimizer_updates
    encode_seconds.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    first_loss = len(trainer.loss_recorder.global_loss_list)
    if cli.profile:
        checkpoint_fn = torch.utils.checkpoint.checkpoint

        def label_recompute(*args, **kwargs):
            if kwargs.get("use_reentrant") is False and "context_fn" not in kwargs:
                kwargs["context_fn"] = lambda: (nullcontext(), torch.profiler.record_function("sdxl_checkpoint_recompute"))
            return checkpoint_fn(*args, **kwargs)

        torch.utils.checkpoint.checkpoint = label_recompute
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                train_until(cli.warmup + cli.steps)
        finally:
            torch.utils.checkpoint.checkpoint = checkpoint_fn
        profile.export_chrome_trace(str(output / "trace.json"))
    else:
        train_until(cli.warmup + cli.steps)
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    losses = trainer.loss_recorder.global_loss_list[first_loss:]
    if len(losses) != cli.steps or not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError(f"expected {cli.steps} finite losses, got {losses}")
    timed_updates = optimizer_updates - warmup_updates
    if timed_updates == 0:
        raise RuntimeError("all timed optimizer updates were skipped")
    if cli.shuffle_captions and len(encode_seconds) != cli.steps:
        raise RuntimeError(f"expected {cli.steps} live text encodes, got {len(encode_seconds)}")
    report = {
        "git_commit": revision,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint_stat.st_size,
        "resolution": list(resolution),
        "dataset_sha256": dataset_hashes,
        "latent_cache_dir": str(cached_dataset),
        "latent_cache_files_before_init": len(latent_files),
        "latent_cache_files_after_init": len(list(cached_dataset.glob("*_sdxl.npz"))),
        "latent_images_encoded_during_init": latent_images_encoded,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "accelerate": version("accelerate"),
        "transformers": version("transformers"),
        "xformers": version("xformers") if cli.attention == "xformers" else None,
        "warmup_optimizer_updates": warmup_updates,
        "training_args": vars(args),
        "warmup_steps": cli.warmup,
        "timed_steps": cli.steps,
        "timed_optimizer_updates": timed_updates,
        "shuffle_caption": cli.shuffle_captions,
        "detach_latents": cli.detach_latents or cli.output_only,
        "output_only": cli.output_only,
        "adapter_modules": len(network.unet_loras),
        "trainable_parameters": sum(param.numel() for param in parameters.values() if param.requires_grad),
        "uncheckpointed_layers": uncheckpointed_layers,
        "keep_middle_resident": cli.keep_middle_resident,
        "swapped_block_indices": swapped_indices,
        "profiled": cli.profile,
        "text_encoder_devices": sorted(encode_devices),
        "text_encoder_seconds_per_image": sum(encode_seconds) / cli.steps if cli.shuffle_captions else None,
        "init_seconds": init_seconds,
        "timed_seconds": seconds,
        "seconds_per_image": seconds / cli.steps,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "median_loss": median(losses),
        "losses": losses,
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"{report['seconds_per_image']:.3f} s/image | {report['peak_allocated_mib']:.0f} MiB peak allocated | {result_path}")

    if cli.sample_adapters:
        args.output_name = "portrait"
        def save_and_sample():
            steps = trainer.global_step
            unet_training, network_training = unet.training, network.training
            trainer.optimizer_eval_fn()
            try:
                unet.eval()
                network.eval()
                trainer.save_model(f"step-{steps:04d}.safetensors", network, steps, trainer.checkpoint_epoch())
                sample(steps, preview_prompts)
            finally:
                unet.train(unet_training)
                network.train(network_training)
                trainer.optimizer_train_fn()

        save_and_sample()
        if reference_seconds is not None:
            additional = round((reference_seconds - seconds) / report["seconds_per_image"])
            if additional > 0:
                torch.cuda.synchronize()
                start = perf_counter()
                train_until(min(args.max_train_steps, trainer.global_step + additional))
                torch.cuda.synchronize()
                report["matched_time_seconds"] = seconds + perf_counter() - start
                report["matched_time_updates"] = optimizer_updates
                result_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
                save_and_sample()


if __name__ == "__main__":
    main()
