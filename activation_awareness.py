import copy
import torch
import gc
def pseudo_quantize_tensor(w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False):
    org_w_shape = w.shape

    if len(org_w_shape) == 2:  # Handling 1D weights
        if q_group_size > 0:
            assert org_w_shape[-1] % q_group_size == 0
            w = w.reshape(-1, q_group_size)
            assert w.dim() == 2

        if zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2 ** n_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (n_bit - 1) - 1
            min_int = -(2 ** (n_bit - 1))
            scales = max_val / max_int
            zeros = 0

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        if inplace:
            (((w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)).mul_(scales))
        else:
            w = ((torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros) * scales)

        assert torch.isnan(w).sum() == 0

    elif len(org_w_shape) == 4:  # Handling 2D convolutional weights
        if zero_point:
            max_val = w.amax(dim=(1, 2, 3), keepdim=True)
            min_val = w.amin(dim=(1, 2, 3), keepdim=True)
            max_int = 2 ** n_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        else:
            max_val = w.abs().amax(dim=(1, 2, 3), keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (n_bit - 1) - 1
            min_int = -(2 ** (n_bit - 1))
            scales = max_val / max_int
            zeros = 0

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        if inplace:
            (((w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)).mul_(scales))
        else:
            w = ((torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros) * scales)

        assert torch.isnan(w).sum() == 0

    else:
        raise ValueError("Unsupported input shape")

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales, zeros
        # if len(org_w_shape) == 2:
        #     return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
        # elif len(org_w_shape) == 4:
        #     return w, scales.view(w.shape[0], -1, 1, 1), zeros.view(w.shape[0], -1, 1, 1)
    else:
        return w


@torch.no_grad()
def get_act_scale(x):
    if (len(x.shape) == 2):
        return x.abs().view(-1, x.shape[-1]).mean(0)
    elif (len(x.shape) == 4):
        return x.abs().view(-1, x.shape[1]).mean(0)


@torch.no_grad()
def _search_module_scale(x, w, module, y):
    # w: co, ci
    # x: n, ci
    x_max = get_act_scale(x)
    best_error = float("inf")
    best_ratio = -1
    best_scales = None
    n_grid = 20
    history = []
    best_ratio = -1
    sls = []
    org_sd = copy.deepcopy(module.state_dict())
    for ratio in range(n_grid):
        ratio = ratio * 1 / n_grid
        scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        if (len(module.weight.shape) > 2):
            module.weight.mul_(scales.view(1, -1, 1, 1).to(module.weight.device))
            module.weight.data = pseudo_quantize_tensor(w) / (scales.view(1, -1, 1, 1))
        else:
            module.weight.mul_(scales.view(1, -1).to(module.weight.device))
            module.weight.data = pseudo_quantize_tensor(w) / (scales.view(1, -1))
        out = module(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = (
            (y - out).float().pow(2).mean().item()
        )  # float prevents overflow
        history.append(loss)
        sls.append(ratio)
        is_best = loss < best_error
        if is_best:
            best_error = loss
            best_ratio = ratio
            best_scales = scales
            # print(loss, ratio, scales)
        module.load_state_dict(org_sd)

    assert torch.isnan(best_scales).sum() == 0, best_scales
    return best_scales.detach()



@torch.no_grad()
def auto_clip_layer(
    w, input_feat, n_bit, module=None, q_config=None, n_grid=20, max_shrink=0.5, n_sample_token=512
):

    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    if(q_config is not None):
        group_size = (
            q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
        )
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
        input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
        w = w.reshape(w.shape[0], 1, -1, group_size)

        oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
        assert w.shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(w.shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
    else:
        w_all = copy.deepcopy(w)
        best_max_val_all = []
        org_max_val = w.abs().view(w.shape[0],-1).amax(dim=1, keepdim=True) # C_o-wise
        # Unsqueeze Clipping Max Values
        for ip in range(w_all.dim()-2):
            org_max_val = org_max_val.unsqueeze(-1)

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = module(input_feat) # co, n_token, n_group

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit)
            module.weight.data = q_w
            cur_out = module(input_feat)
            if isinstance(cur_out, tuple):
                cur_out = cur_out[0]

            err = (cur_out - org_out).pow(2).view(w.shape[0],-1).mean(dim=1)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs.squeeze()
            min_errs.squeeze()[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]

    best_max_val_all.append(best_max_val)
    best_max_val = torch.cat(best_max_val_all, dim=0)

    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze()