"""
PPL对比实验：验证LayerNorm融合并替换为RMSNorm的效果
运行: python compare_fusion_ppl.py --model facebook/opt-125m --rotate
"""
import torch
import transformers
import model_utils
import data_utils
import rotation_utils
import eval_utils
import utils
import quant_utils
import hadamard_utils


def evaluate_ppl(model, args):
    """评估wikitext2数据集上的PPL"""
    testloader = data_utils.get_loaders(
        "wikitext2",
        seed=0,
        model=args.model,
        seqlen=model.seqlen,
        hf_token=args.hf_token,
        eval_mode=True
    )
    ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
    return ppl


def configure_hadamard(model, qlayers, args):
    """配置Hadamard参数（支持Llama和OPT模型）"""
    # 检测模型类型
    model_name_lower = args.model.lower()
    is_llama = "llama" in model_name_lower
    is_opt = "opt" in model_name_lower

    # 获取配置参数
    if is_llama:
        intermediate_size = model.config.intermediate_size
        mlp_output_name = 'down_proj'
        attn_output_name = 'o_proj'
    elif is_opt:
        intermediate_size = model.config.ffn_dim
        mlp_output_name = 'fc2'
        attn_output_name = 'out_proj'
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    num_attention_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    for name in qlayers:
        # 配置MLP输出层（完全Hadamard）
        if mlp_output_name in name:
            if is_llama:
                # Llama: 使用在线Hadamard
                had_K, K = hadamard_utils.get_hadK(intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            else:
                # OPT: 禁用在线Hadamard，避免数值问题（ffn_dim通常不是标准2的幂）
                print(f"  [INFO] 禁用 {name} 的在线Hadamard（OPT模型ffn_dim={intermediate_size}）")

        # 配置Attention输出层（部分Hadamard）- 仅Llama支持
        if attn_output_name in name:
            if is_llama:
                # Llama: num_heads通常是2的幂，支持部分Hadamard
                had_K, K = hadamard_utils.get_hadK(num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = head_dim
                qlayers[name].fp32_had = args.fp32_had
            else:
                # OPT: num_heads通常不是2的幂，禁用Hadamard避免PPL暴涨
                print(f"  [INFO] 禁用 {name} 的Hadamard变换（OPT模型num_heads={num_attention_heads}不是2的幂）")


def main():
    args = utils.parser_gen()
    transformers.set_seed(args.seed)
    
    print(f"\n模型: {args.model}")
    print(f"Rotate: {args.rotate} (mode: {args.rotate_mode})" if args.rotate else f"Rotate: {args.rotate}")
    
    results = {}
    
    # Step 1: fuse_modules
    print("\n[1/4] fuse_modules...")
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    rotation_utils.fuse_modules(model)
    
    model.to(utils.DEV)
    results['fuse_wiki'] = evaluate_ppl(model, args)
    print(f"  PPL: wiki={results['fuse_wiki']:.4f}")
    
    model.cpu()
    utils.cleanup_memory()
    
    # Step 2: +RMSNorm
    print("\n[2/4] +RMSNorm...")
    rotation_utils.replace_layernorm_with_rmsnorm(model)
    
    model.to(utils.DEV)
    results['rmsnorm_wiki'] = evaluate_ppl(model, args)
    print(f"  PPL: wiki={results['rmsnorm_wiki']:.4f}")
    
    model.cpu()
    utils.cleanup_memory()
    
    # Step 3: fuse + rotate + ActQuant（条件执行）
    print("\n[3/4] fuse+rotate+ActQuant...")
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    
    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory()
        
        # 添加ActQuantWrapper并配置Hadamard参数
        quant_utils.add_actquant(model)
        qlayers = quant_utils.find_qlayers(model)
        configure_hadamard(model, qlayers, args)
    else:
        print("  [INFO] --rotate 未启用，跳过此步骤")
        results['rotate_wiki'] = float('nan')
    
    model.to(utils.DEV)
    if args.rotate:
        results['rotate_wiki'] = evaluate_ppl(model, args)
        print(f"  PPL: wiki={results['rotate_wiki']:.4f}")
    
    model.cpu()
    utils.cleanup_memory()
    
    
    # 总结
    print("\n" + "=" * 50)
    print("结果总结 (WikiText2)")
    print("=" * 50)
    print(f"{'步骤':<20} {'PPL':>10} {'Δ%':>10}")
    print("-" * 50)
    
    # 使用fuse作为baseline
    baseline_ppl = results['fuse_wiki']
    
    steps = [
        ('fuse', 'fuse_wiki'),
        ('+rmsnorm', 'rmsnorm_wiki'),
        ('fuse+rotate', 'rotate_wiki'),
    ]
    
    for name, wk in steps:
        wiki_ppl = results[wk]
        if not isinstance(wiki_ppl, float) or wiki_ppl != wiki_ppl:  # 检查nan
            print(f"{name:<20} {'N/A':>10} {'N/A':>10}")
        else:
            wiki_diff = (wiki_ppl - baseline_ppl) / baseline_ppl * 100
            print(f"{name:<20} {wiki_ppl:>10.4f} {wiki_diff:>+9.2f}%")
    
    print("=" * 50)
    
    # 输出最终模型结构信息
    print_model_structure(model, args)


def print_model_structure(model, args):
    """打印模型结构信息（简化版，重复层只显示一层，参数和模块名对应）"""
    import re
    
    print("\n" + "=" * 50)
    model_type_str = "LLaMA" if "llama" in args.model.lower() else "OPT"
    print(f"{model_type_str} 模型结构信息")
    print("=" * 50)
    
    # 1. 所有模块和对应的参数（重复的层只显示一层）
    print("\n1. 所有模块及其参数:")
    shown_modules = set()
    module_params = {}  # 存储每个模块的参数
    
    # 先收集所有参数，按模块分组
    for name, param in model.named_parameters():
        # 提取模块名（参数名去掉最后一项）
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            module_name, param_name = parts
        else:
            module_name = ""
            param_name = name
        
        # 将层索引替换为{L}来表示通配
        generic_module = re.sub(r'\.layers\.\d+', '.layers.{L}', module_name)
        
        if generic_module not in module_params:
            module_params[generic_module] = {}
        
        # 参数名也做通配处理
        generic_param = re.sub(r'\.layers\.\d+', '.layers.{L}', name)
        module_params[generic_module][generic_param] = param.shape
    
    # 获取所有模块信息
    module_types = {}
    for name, module in model.named_modules():
        if not name:
            continue
        generic_name = re.sub(r'\.layers\.\d+', '.layers.{L}', name)
        if generic_name not in module_types:
            module_types[generic_name] = module.__class__.__name__
    
    # 按层级顺序输出模块和参数
    sorted_modules = sorted(module_params.keys(), key=lambda x: (len(x.split('.')), x))
    
    for generic_module in sorted_modules:
        if generic_module in module_types:
            module_type = module_types[generic_module]
        else:
            module_type = "Module"
        
        # 跳过重复的层模块，只保留通配形式
        if '.layers.{L}' in generic_module and re.search(r'\.layers\.\d+', generic_module):
            continue
            
        # 输出模块名和类型
        display_module = generic_module if generic_module else "(root)"
        print(f"\n{display_module}: {module_type}")
        
        # 输出该模块的所有参数
        params = module_params[generic_module]
        for param_name, shape in sorted(params.items()):
            # 只显示参数名（去掉模块前缀）
            short_name = param_name.rsplit('.', 1)[-1]
            print(f"  - {short_name}: {shape}")
    
    # 2. 模型类型和基本信息
    print("\n" + "=" * 60)
    print("2. 模型类型和基本信息")
    print("=" * 60)
    print(f"模型类型: {type(model)}")
    print(f"模型类名: {model.__class__.__name__}")
    print(f"配置类名: {model.config.__class__.__name__}")
    
    # 3. 模型关键属性
    print("\n" + "=" * 60)
    print("3. 模型的关键属性")
    print("=" * 60)
    key_attrs = ['config', 'device', 'dtype', 'training']
    for attr in key_attrs:
        if hasattr(model, attr):
            print(f"  - {attr}")
    
    # 4. 模型结构探索
    print("\n" + "=" * 60)
    print("4. 模型结构探索")
    print("=" * 60)
    
    # 检测模型结构类型
    if hasattr(model, 'model'):
        print("model.model: 存在")
        inner_model = model.model
    elif hasattr(model, 'decoder'):
        print("model.decoder: 存在")
        inner_model = model.decoder
    elif hasattr(model, 'transformer'):
        print("model.transformer: 存在")
        inner_model = model.transformer
    else:
        print("model.model: 不存在")
        print("model.decoder: 不存在")
        print("model.transformer: 不存在")
        inner_model = None
    
    # Decoder层详细结构
    if inner_model is not None and hasattr(inner_model, 'layers'):
        layers = inner_model.layers
        print(f"\n模型总层数: {len(layers)}")
        print(f"隐藏层维度: {model.config.hidden_size}")
        print(f"注意力头数: {model.config.num_attention_heads}")
        print(f"每个头的维度: {model.config.hidden_size // model.config.num_attention_heads}")
        
        if len(layers) > 0:
            first_layer = layers[0]
            print(f"\n每层包含的子模块:")
            for name, module in first_layer.named_children():
                print(f"  - {name}: {module.__class__.__name__}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
