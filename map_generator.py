from PIL import Image
import numpy as np
import torch


def get_image_heat_map(img, att_mat, head_num=-1, token=0, model="ZeroshotCLIP"):

    patch_size = 32 # default
    num_tokens = 1
    patch_num = 49
    
    att_mat = att_mat.cpu().float()  # [n_layers, n_heads, token_seq_length, token_seq_length]
    n_heads = att_mat.shape[1]

    # Attention output
    attentions = att_mat[0, :, 0, num_tokens:patch_num + 1].reshape(n_heads, -1)

    w_featmap = img.shape[2] // patch_size
    h_featmap = img.shape[1] // patch_size

    # Attention output per head of average
    if(head_num < 0):
        attentions = attentions.reshape(n_heads, w_featmap, h_featmap).mean(dim=0)
    else:
        attentions = attentions.reshape(n_heads, w_featmap, h_featmap)[head_num]

    # reshaping it to same as that of the image size
    attention = np.asarray(Image.fromarray(attentions.numpy()).resize((h_featmap * patch_size, w_featmap * patch_size))).copy()
    return attention



def get_image_attn_mask(img_size, att_mat, avg=False, layer=-1, token=0):
    """
    image.shape => [img_dim, img_dim]
    grid => img_dim//patch_size
    token_seq_length => 1+grid**2, where the "1+" comes from the addition of CLS token
    att_mat.shape => [n_layers, n_heads, token_seq_length, token_seq_length]
    """

    # Heavily based on: 
    # from https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    att_mat = att_mat.cpu()  # [n_layers, n_heads, token_seq_length, token_seq_length]

    # Average the attention weights across all heads.
    att_mat = att_mat.mean(dim=1)  # [n_layers, token_seq_length, token_seq_length]
    aug_att_mat = att_mat.float()  # torch.matmul won't work with fp16

    #
    # CLIP uses a series of ResidualAttentionBlocks that look like this:
    # input x => LN => ATTN => ADD x => LN => MLP => ADD x => output y
    #
    # But the input x is NOT a simple image at all. It is the result of
    # a conv2d with a kernel and stride the same size so it "chops" the image into
    # something like patches with a lot of layers (the token embedding dimension).
    # A class (CLS) token (learned and frozen for inference) is concatenated to the start and to the
    # whole "sequence" we add the positional encoder (again, learned and frozen for inference) 
    #
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        tmp_mat = aug_att_mat[n] + joint_attentions[n-1]
        tmp_mat /= tmp_mat.norm()
        # ignoring the effects of the MLP...
        joint_attentions[n] = torch.matmul(tmp_mat, joint_attentions[n-1])

    # Attention from the output token to the input space.
    if avg:
        v = joint_attentions.mean(dim=0)
    else:
        v = joint_attentions[layer]

    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    att = torch.arange(att_mat.size(1))
    att = att[att != att[token]]
    mask = v[token, att].reshape(grid_size, grid_size)  # token=0 is CLS (the only one used by CLIP at the end)
    mask = np.asarray(Image.fromarray(mask.numpy()).resize((img_size))).copy()
    mask -= mask.min()
    mask /= mask.max()
    return mask
