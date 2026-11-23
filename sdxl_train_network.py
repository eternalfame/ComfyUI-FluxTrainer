import argparse
from typing import List, Optional

import torch
from accelerate import Accelerator
from .library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from .library import sdxl_model_util, sdxl_train_util, strategy_base, strategy_sd, strategy_sdxl, train_util
from . import train_network
from .library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def make_offload_hooks(unet, min_numel=1 << 18):
    param_ids = {id(p) for p in unet.parameters()}
    def pack(t):
        if (t.device.type != "cuda" or t.numel() < min_numel
                or isinstance(t, torch.nn.Parameter) or id(t) in param_ids):
            return t
        buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        buf.copy_(t, non_blocking=True)
        ev = torch.cuda.Event(); ev.record()
        return (t.device, buf, ev)
    def unpack(x):
        if isinstance(x, torch.Tensor):
            return x
        dev, buf, ev = x
        ev.synchronize()
        return buf.to(dev, non_blocking=True)
    return pack, unpack


class SdxlNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self._size_emb_cache = {}
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True

    def init_train(self, args):
        legacy_gpu_encode = vars(args).pop("te_gpu_encode", False)
        if getattr(args, "text_encoder_device", None) is None:
            args.text_encoder_device = "transient_gpu" if legacy_gpu_encode else "cpu"

        if getattr(args, "output_only_lora", False):
            if args.network_module != ".networks.lora" or not args.network_train_unet_only or args.network_train_text_encoder_only:
                raise ValueError("Output-only SDXL LoRA requires the standard LoRA network and frozen text encoders.")
            if args.dim_from_weights:
                raise ValueError("Output-only SDXL LoRA requires ranks from the node. Remove --dim_from_weights.")

            ranks = [0] * 13 + [args.network_dim] * 9 + [0]
            block_dims = "block_dims=" + ",".join(map(str, ranks))
            network_args = list(args.network_args or [])
            if any(arg.partition("=")[0] in {"block_dims", "conv_dim", "conv_block_dims"} and arg != block_dims
                   for arg in network_args):
                raise ValueError("Output-only SDXL LoRA sets the block ranks. Remove custom block or convolution ranks.")
            if block_dims not in network_args:
                args.network_args = [*network_args, block_dims]
            logger.info("Experimental output-only SDXL LoRA enabled.")
        return super().init_train(args)

    def assert_extra_args(self, args, train_dataset_group):
        super().assert_extra_args(args, train_dataset_group)
        sdxl_train_util.verify_sdxl_training_args(args)

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        train_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_train_util.load_target_model(args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        # モデルに xformers とか memory efficient attention を組み込む
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        return sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    def get_tokenize_strategy(self, args):
        return strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sdxl.SdxlTokenizeStrategy):
        return [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            False, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_sdxl.SdxlTextEncodingStrategy()

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        return text_encoders + [accelerator.unwrap_model(text_encoders[-1])]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, None, args.skip_cache_check, is_weighted=args.weighted_captions
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        keep_on_accelerator = self.is_train_text_encoder(args) or getattr(args, "text_encoder_device", "cpu") == "gpu"
        encoder_device = accelerator.device if keep_on_accelerator else torch.device("cpu")
        encoder_dtype = torch.float32 if encoder_device.type == "cpu" else weight_dtype
        logger.info("SDXL text encoders will reside on %s during training", encoder_device)
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # メモリ消費を減らす
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)
            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders + [accelerator.unwrap_model(text_encoders[-1])], accelerator)
            accelerator.wait_for_everyone()

        for encoder in text_encoders:
            encoder.to(encoder_device, dtype=encoder_dtype)

        if args.cache_text_encoder_outputs:
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
            input_ids1 = batch["input_ids"]
            input_ids2 = batch["input_ids2"]
            with torch.enable_grad():
                # Get the text embedding for conditioning
                # TODO support weighted captions
                # if args.weighted_captions:
                #     encoder_hidden_states = get_weighted_text_embeddings(
                #         tokenizer,
                #         text_encoder,
                #         batch["captions"],
                #         accelerator.device,
                #         args.max_token_length // 75 if args.max_token_length else 1,
                #         clip_skip=args.clip_skip,
                #     )
                # else:
                input_ids1 = input_ids1.to(accelerator.device)
                input_ids2 = input_ids2.to(accelerator.device)
                encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                    args.max_token_length,
                    input_ids1,
                    input_ids2,
                    tokenizers[0],
                    tokenizers[1],
                    text_encoders[0],
                    text_encoders[1],
                    None if not args.full_fp16 else weight_dtype,
                    accelerator=accelerator,
                )
        else:
            encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
            encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
            pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

            # # verify that the text encoder outputs are correct
            # ehs1, ehs2, p2 = train_util.get_hidden_states_sdxl(
            #     args.max_token_length,
            #     batch["input_ids"].to(text_encoders[0].device),
            #     batch["input_ids2"].to(text_encoders[0].device),
            #     tokenizers[0],
            #     tokenizers[1],
            #     text_encoders[0],
            #     text_encoders[1],
            #     None if not args.full_fp16 else weight_dtype,
            # )
            # b_size = encoder_hidden_states1.shape[0]
            # assert ((encoder_hidden_states1.to("cpu") - ehs1.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((encoder_hidden_states2.to("cpu") - ehs2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((pool2.to("cpu") - p2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # logger.info("text encoder outputs verified")

        return encoder_hidden_states1, encoder_hidden_states2, pool2

    def call_unet(
        self,
        args,
        accelerator,
        unet,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
        indices: Optional[List[int]] = None,
    ):
        if getattr(args, "output_only_lora", False):
            noisy_latents = noisy_latents.detach()
        noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]

        key = (
            tuple(orig_size.reshape(-1).tolist()),
            tuple(crop_size.reshape(-1).tolist()),
            tuple(target_size.reshape(-1).tolist()),
            accelerator.device, weight_dtype,
        )
        if key not in self._size_emb_cache:
            # Random crop coordinates can produce a new GPU tensor every update.
            if len(self._size_emb_cache) >= 128:
                del self._size_emb_cache[next(iter(self._size_emb_cache))]
            self._size_emb_cache[key] = sdxl_train_util.get_size_embeddings(
                orig_size, crop_size, target_size, accelerator.device, weight_dtype=weight_dtype
            )
        embs = self._size_emb_cache[key]

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        if indices is not None and len(indices) > 0:
            noisy_latents = noisy_latents[indices]
            timesteps = timesteps[indices]
            text_embedding = text_embedding[indices]
            vector_embedding = vector_embedding[indices]

        pack, unpack = make_offload_hooks(unet)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)

        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet, validation_settings=None):
        image_tensors = sdxl_train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet, validation_settings)
        return image_tensors

    def prepare_unet_with_accelerator(self, args, accelerator, unet):
        if not getattr(args, "blocks_to_swap", None):
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        keep_resident = (len(unet.input_blocks),) if getattr(args, "keep_middle_resident", False) else ()
        unet.enable_block_swap(
            args.blocks_to_swap, accelerator.device, use_arena=getattr(args, "swap_arena", False), keep_resident=keep_resident
        )
        unet = accelerator.prepare(unet, device_placement=[False])
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.move_to_device_except_swap_blocks(accelerator.device)
        unwrapped_unet.prepare_block_swap_before_forward()
        return unet


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = SdxlNetworkTrainer()
    trainer.train(args)
