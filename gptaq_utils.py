import math
import time
import tqdm
import torch
import torch.nn as nn
import utils
import quant_utils
import model_utils
import logging
import functools

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class GPTAQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)
        self.act_magnitude = torch.zeros((self.columns,), device=self.dev)  # 每个输入通道的平均激活幅度
        self.nsamples = 0

    def add_batch(self, inp, out, fp_inp=None):

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        # 使用原模型的FP输入计算通道重要性（而非量化后的激活值）
        # fp_inp 形状: [hidden_size, seq_len] (已经被转置过，来自 FPInputsCache)
        if fp_inp is not None:
            # fp_inp 已经是转置后的形状 [hidden_size, seq_len]
            # 需要转回 [seq_len, hidden_size] 来计算 act_abs_mean
            if fp_inp.shape[0] == self.columns:
                # fp_inp: [hidden_size, seq_len] -> 转置后: [seq_len, hidden_size]
                act_abs_mean = fp_inp.t().abs().mean(dim=0)
            else:
                # 维度不匹配时，使用量化后的输入（回退方案）
                act_abs_mean = inp.abs().mean(dim=0)
        else:
            act_abs_mean = inp.abs().mean(dim=0)
        
        # 更新平均激活幅度的累积平均
        self.act_magnitude = (self.act_magnitude * self.nsamples + act_abs_mean * tmp) / (self.nsamples + tmp)

        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.dXXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        # 计算 dXXT 需要使用原模型的 FP 输入
        # fp_inp 形状: [hidden_size, seq_len] (已转置), inp 形状: [hidden_size, seq_len] (已转置)
        if fp_inp is not None and fp_inp.shape[0] == self.columns:
            fp_inp_scaled = fp_inp.float() * math.sqrt(2 / self.nsamples)
            dX = fp_inp_scaled - inp
            self.dXXT += dX.matmul(inp.t())

    def fasterquant(
            self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, alpha=0.25
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        # 准备激活值幅度权重（用于MSE加权）
        act_weights = self.act_magnitude.clone()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, act_weights=act_weights)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        self.dXXT[:, dead] = 0
        act_weights[dead] = 0  # dead通道的激活值幅度也置零

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], act_weights=act_weights[i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            self.dXXT = self.dXXT[perm][:, perm]
            act_weights = act_weights[perm]  # 激活值幅度也按相同顺序重排
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        # scale it by alpha due to collection of dXXT axnd H
        P = alpha * ((self.dXXT @ Hinv.T).triu_(diagonal=1)) @ Hinv
        del self.dXXT

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i):(i1 + i + groupsize)], 
                                act_weights=act_weights[(i1 + i):(i1 + i + groupsize)]
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) - w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) - W1.matmul(P[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        self.dXXT = None
        self.act_magnitude = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


@torch.no_grad()
def gptaq_fwrd(model, dataloader, dev, args):
    '''
    From GPTQ repo
    Support both OPT and LLaMA models
    '''
    logging.info('-----GPTAQ Quantization-----')

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Detect model type
    if 'opt' in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        is_opt = True
    else:  # LLaMA
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        # Move rotary_emb to GPU (needed for newer transformers versions)
        if hasattr(model.model, 'rotary_emb'):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
        is_opt = False

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if not is_opt:  # LLaMA has position_ids
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if is_opt:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    attention_mask = cache['attention_mask']
    position_ids = cache.get('position_ids', None)

    quantizers = {}

    # Define sequential layers based on model type
    if is_opt:
        sequential = [
            ['self_attn.q_proj.module', 'self_attn.k_proj.module', 'self_attn.v_proj.module'],
            ['self_attn.out_proj.module'],
            ['fc1.module'],
            ['fc2.module']
        ]
    else:  # LLaMA
        sequential = [
            ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
            ['self_attn.o_proj.module'],
            ['mlp.up_proj.module', 'mlp.gate_proj.module'],
            ['mlp.down_proj.module']
        ]

    fp_inputs_cache = model_utils.FPInputsCache(sequential)
    fp_inps = inps.clone()

    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        bits_config = quant_utils.disable_act_quant(layer)
        fp_inputs_cache.add_hook(full)

        for j in range(args.nsamples):
            if is_opt:
                fp_inps[j] = layer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                fp_inps[j] = layer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        fp_inputs_cache.clear_hook()
        quant_utils.enable_act_quant(layer, bits_config)

        for names in sequential:
            subset = {n: full[n] for n in names if n in full}

            if not subset:
                continue

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not (args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and ('down_proj' in name or 'fc2' in name):
                    layer_weight_bits = 8
                gptq[name] = GPTAQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    fp_cache = fp_inputs_cache.fp_cache.get(name, None)
                    fp_inp = fp_cache.pop(0) if fp_cache else None
                    gptq[name].add_batch(inp[0].data, out.data, fp_inp=fp_inp)

                return tmp

            first_module_name = list(subset.keys())[0]
            handle = subset[first_module_name].register_forward_hook(add_batch(first_module_name))

            for j in range(args.nsamples):
                if is_opt:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            handle.remove()

            # copy H and dXXT
            for name in subset:
                if name != first_module_name:
                    gptq[name].H = gptq[first_module_name].H
                    gptq[name].dXXT = gptq[first_module_name].dXXT

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order,
                    static_groups=args.static_groups
                )
                if is_opt:
                    quantizers['model.decoder.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                else:
                    quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            if is_opt:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        fp_inputs_cache.clear_cache()
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTAQ Quantization Done-----\n')

    return quantizers
