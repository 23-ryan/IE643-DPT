import os.path as osp
from pydoc import classname
from turtle import forward

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.zsclip import CUSTOM_TEMPLATES
import matplotlib.pyplot as plt
from map_generator import *

_tokenizer = _Tokenizer()
devices='cpu'
CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "Brain": "a photo of a {} brain.",
}

# ================================================================= #
# 这个是VPT DEEP+CoOp（Frozen）+CLIP（Frozen）的实现。                #
# ================================================================= #


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


# ++++++++++++++++++++++++++++++++++++++++++++ #
#                  VPT DEEP!                   #
# ++++++++++++++++++++++++++++++++++++++++++++ #
class VPTDeepPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # hyper param
        self.n_ctx = cfg.TRAINER.VPT.N_CTX
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels # 768
        self.clip_imsize = clip_model.visual.input_resolution
        self.cfg_imsize = cfg.INPUT.SIZE[0]
        self.layers = clip_model.visual.transformer.layers
        
        ctx_vectors = torch.empty(self.layers, self.n_ctx, self.ctx_dim, dtype=self.dtype)
        for i in range(self.layers):
            nn.init.normal_(ctx_vectors[i], std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
    def forward(self):
        return self.ctx

class ProjLearner(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.proj = clip_model.visual.proj
        
    def forward(self,x):
        if self.proj is not None:
            x = x @ self.proj
        return x
    
class Transformer_VPTD(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # hyper param
        self.n_ctx = cfg.TRAINER.VPT.N_CTX
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels # 768
        self.clip_imsize = clip_model.visual.input_resolution
        self.cfg_imsize = cfg.INPUT.SIZE[0]
        self.layers = clip_model.visual.transformer.layers

        # model
        transformer = clip_model.visual.transformer
        self.resblocks: nn.Sequential = transformer.resblocks
        self.layers = transformer.layers

        self.ctx_learner = VPTDeepPromptLearner(cfg, classnames, clip_model)


    def forward(self, x):
        ctx = self.ctx_learner()
        ctx = ctx.unsqueeze(2).expand(-1, -1, x.shape[1], -1)
        
        weights = []
        for i in range(self.layers): 
            if i != 0:
                x = x[:-self.n_ctx, :, :]
            
            # print(ctx[i].shape, x.shape)
            x = torch.cat([x, ctx[i]], dim=0)
            x, attentions = self.resblocks[i](x)
            weights.append(attentions)

        weights = torch.stack(weights)
        return x, weights


class ImageEncoder_VPTD(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = Transformer_VPTD(cfg, classnames, clip_model)
        self.ln_post = clip_model.visual.ln_post
        # self.proj = clip_model.visual.proj
        self.proj = ProjLearner(clip_model)
        
    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # class_embedding is class token.
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, weights = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :]) # only take class token which is awsome.

        x = self.proj(x)

        return x, weights


class CustomCLIP_VPTD(nn.Module):
    def __init__(self, cfg, classnames, clip_model, devices):
        super().__init__()
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        clip_model.to(devices)
        prompts = prompts.to(devices)
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        clip_model.to('cpu')
        self.text_features = nn.Parameter(text_features)
        # visual
        self.image_encoder = ImageEncoder_VPTD(cfg, classnames, clip_model)
        # visual end
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image = image.to(next(self.image_encoder.parameters()).device)
        image_features, weights = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()

        return logits, weights

# end


@TRAINER_REGISTRY.register()
class VPT(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def model_inference(self, input, label=None):
        curr_out = self.model(input)
        img_attn_weights = curr_out[1]

        image = input
        temp = curr_out[0] - curr_out[0].max()
        mo = (torch.exp(temp)/(torch.exp(temp).sum(dim=1).reshape((-1,1))))
        pred_score = mo.max(1)[0]
        pred_label = mo.max(1)[1]

        actual_label = label

        ########################## HEAT MAP PER IMAGE ###########################
        # print(img_attn_weights.shape)
        # exit(0)

        image_index = 0
        img_pp = image[0].cpu().numpy()
        curr_img_attn = img_attn_weights[:,image_index,...]

        # fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(13, 13))
        # img_count = 0
        # for i in range(3):
        #     for j in range(4):
        #         if img_count < batch_size:

        #             axes[i, j].imshow(img_pp.transpose(1,2,0))
        #             heatmap = get_image_heat_map(img_pp, curr_img_attn, head_num=i*3+j, token=0, model="VPT")
        #             axes[i, j].imshow(heatmap, cmap="inferno", alpha=0.6)
        #             axes[i, j].title.set_text(f"Attn head: {img_count}")
        #             axes[i, j].axis("off")
        #             img_count += 1


        heatmap = get_image_heat_map(img=img_pp, att_mat=curr_img_attn, token=0, model=self.cfg.TRAINER.NAME)
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        img_to_show = img_pp.transpose(1,2,0)

        # if(self.cfg.DATASET.NAME == 'brain'):
        #     ax[0].imshow(img_pp[0], cmap='grey')
        # else:
        #     ax[0].imshow(img_to_show)

        # ax[0].axis("off")
        # ax[0].set_title('Original Image')
        
        # if(self.cfg.DATASET.NAME == 'brain'):
        #     ax[1].imshow(img_pp[0], cmap='grey')
        # else:
        #     ax[1].imshow(img_to_show)

        # ax[1].imshow(heatmap, cmap="inferno", alpha=0.5)
        # ax[1].axis("off")
        # ax[1].set_title('Masked Image')


        # plt.subplots_adjust(bottom=0.2)
        # fig.text(0.5, 0.05, f'Actual: {actual_label[image_index]},  Precition: {pred_label[image_index]}, Score: {pred_score[image_index]}', ha='center', fontsize=12)

        # plt.tight_layout()

        # plt.show()
        return curr_out[0]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        # ================================== #
        #              VPT DEEP              #
        # ================================== #
        print("Building custom CLIP VPT Deep")
        self.model = CustomCLIP_VPTD(cfg, classnames, clip_model, self.device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "image_encoder.transformer.ctx_learner" not in name:
                param.requires_grad_(False)
            else:
                print(name)
 

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.image_encoder.transformer.ctx_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("image_encoder.transformer.ctx_learner", self.model.image_encoder.transformer.ctx_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, weights = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))


            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=True)
