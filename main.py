import utils
import torch
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import gptaq_utils
import eval_utils
import hadamard_utils
import csv
import numpy as np
import os


def save_layer_scales_to_csv(quantizers, layer_key, output_path):
    """提取指定层的 scale 并保存为 CSV，供 MATLAB 三维绘图"""
    if layer_key not in quantizers:
        print(f"Warning: {layer_key} not found in quantizers!")
        print(f"Available keys: {list(quantizers.keys())[:10]}...")
        return
    
    quantizer = quantizers[layer_key]
    scale = quantizer.scale.cpu().numpy()
    
    # 确保2D形状 [out_channels, num_groups]
    if scale.ndim == 1:
        scale = scale.reshape(-1, 1)
    elif scale.ndim > 2:
        scale = scale.squeeze()
        if scale.ndim == 1:
            scale = scale.reshape(-1, 1)
    
    print(f"Layer: {layer_key}, Scale shape: {scale.shape}")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入头部注释供MATLAB读取
        writer.writerow([f"# Layer: {layer_key}"])
        writer.writerow([f"# Scale shape: {scale.shape}"])
        writer.writerow([f"# Rows: output channels ({scale.shape[0]})"])
        writer.writerow([f"# Cols: groups ({scale.shape[1] if scale.ndim > 1 else 1})"])
        writer.writerow([])
        # 写入数据
        for row in scale:
            writer.writerow(row.tolist() if isinstance(row, np.ndarray) else [row])
    
    print(f"Scale data saved to: {output_path}")


def add_aq(model, args):
    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not(args.a_asym)
            layer_a_clip = args.a_clip_ratio

            if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                              groupsize=args.v_groupsize,
                                              sym=not(args.v_asym),
                                              clip_ratio=args.v_clip_ratio)

            if 'lm_head' in name: #Skip lm_head quantization
                layer_input_bits = 16

            if 'down_proj' in name:  #Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip)

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)
            k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                          "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                            layer.self_attn,
                            rope_function_name,
                            config=model.config,
                            **k_quant_config)


def main():
    args = utils.parser_gen()

    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()

    # Rotate the weights
    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)
            
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if 'down_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'o_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present

    if args.enable_aq_calibration:
        add_aq(model, args)

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path: # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"], strict=False)
            
        elif not args.w_rtn: # GPTQ Weight Quantization
            assert "llama" in args.model or "opt" in args.model, "Only llama and opt are supported for GPTQ/GPTAQ!"
            
            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            if args.asym_calibrate:
                quantizers = gptaq_utils.gptaq_fwrd(model, trainloader, utils.DEV, args)
                save_dict["w_quantizers"] = quantizers
            else:
                quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
                save_dict["w_quantizers"] = quantizers
            
            # 保存指定层的量化步长到CSV
            rotation_tag = "with_rotate" if args.rotate else "no_rotate"
            layer_name = "model.layers.31.mlp.down_proj.module"  # OPT-125m 第一层 fc2
            
            csv_filename = f"opt125m_fc2_scales_{rotation_tag}.csv"
            csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
            
            save_layer_scales_to_csv(quantizers, layer_name, csv_path)
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
            
        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)

    if not args.enable_aq_calibration:
        add_aq(model, args)

    # Evaluating on dataset
    testloader = data_utils.get_loaders(
            "wikitext2",
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True
        )

    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)

    testloader = data_utils.get_loaders(
            "c4",
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True
        )

    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)

    # if args.wandb:
    #         wandb.log({'ppl/{}'.format(args.eval_dataset.upper()): dataset_ppl})

    if not args.lm_eval:
        return
    else:
        # Import lm_eval utils
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM

    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

    # commenting out this line as it will include two lambda sub-tasks
    # task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
    task_names = args.tasks
    results = lm_eval.simple_evaluate(hflm, tasks=task_names, batch_size=args.lm_eval_batch_size)['results']

    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    print(metric_vals)



if __name__ == '__main__':
    main()
