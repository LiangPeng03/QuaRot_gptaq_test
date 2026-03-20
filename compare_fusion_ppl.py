"""
PPL对比实验：验证LayerNorm融合并替换为RMSNorm的效果
参考: TransformerCompression/experiments/eval_opt125m_layernorm.py
运行: python compare_fusion_ppl.py --model facebook/opt-125m
"""
import torch
import transformers
import model_utils
import data_utils
import rotation_utils
import eval_utils
import utils
import copy


def evaluate_ppl(model, args, dataset="wikitext2"):
    """评估指定数据集上的PPL"""
    testloader = data_utils.get_loaders(
        dataset,
        seed=0,
        model=args.model,
        seqlen=model.seqlen,
        hf_token=args.hf_token,
        eval_mode=True
    )
    ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
    return ppl


def main():
    # 使用与main.py相同的参数解析
    args = utils.parser_gen()
    transformers.set_seed(args.seed)
    
    print("=" * 60)
    print("LayerNorm融合PPL对比实验 (分步评估)")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"评估数据集: wikitext2, c4")
    print()
    
    # =========================================================================
    # Step 1: 加载原始模型并评估PPL
    # =========================================================================
    print("=" * 60)
    print("[Step 1] 加载原始模型并评估PPL...")
    print("=" * 60)
    
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    model.to(utils.DEV)
    
    original_ppl_wiki = evaluate_ppl(model, args, "wikitext2")
    original_ppl_c4 = evaluate_ppl(model, args, "c4")
    
    print(f"  Original WikiText2 PPL: {original_ppl_wiki:.4f}")
    print(f"  Original C4 PPL: {original_ppl_c4:.4f}")
    print()
    
    # 将模型移回CPU释放内存
    model.cpu()
    utils.cleanup_memory()
    
    # =========================================================================
    # Step 2: 应用 fuse_modules (只融合权重，不替换RMSNorm)
    # =========================================================================
    print("=" * 60)
    print("[Step 2] 应用 fuse_modules (融合LayerNorm权重)...")
    print("=" * 60)
    
    # 重新加载模型以确保干净状态
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    
    # 执行融合（只融合，不替换RMSNorm）
    rotation_utils.fuse_modules(model)
    print("  融合完成！")
    print()
    
    # =========================================================================
    # Step 3: 评估 fuse_modules 后的PPL
    # =========================================================================
    print("=" * 60)
    print("[Step 3] 评估 fuse_modules 后的PPL...")
    print("=" * 60)
    
    model.to(utils.DEV)
    after_fuse_ppl_wiki = evaluate_ppl(model, args, "wikitext2")
    after_fuse_ppl_c4 = evaluate_ppl(model, args, "c4")
    
    print(f"  After fuse_modules WikiText2 PPL: {after_fuse_ppl_wiki:.4f}")
    print(f"  After fuse_modules C4 PPL: {after_fuse_ppl_c4:.4f}")
    print()
    
    # 将模型移回CPU释放内存
    model.cpu()
    utils.cleanup_memory()
    
    # =========================================================================
    # Step 4: 应用 replace_layernorm_with_rmsnorm (替换为RMSNorm)
    # =========================================================================
    print("=" * 60)
    print("[Step 4] 将 LayerNorm 替换为 RMSNorm...")
    print("=" * 60)
    
    # 继续在已融合的模型上操作
    rotation_utils.replace_layernorm_with_rmsnorm(model)
    print("  RMSNorm替换完成！")
    print()
    
    # =========================================================================
    # Step 5: 评估 RMSNorm 替换后的PPL
    # =========================================================================
    print("=" * 60)
    print("[Step 5] 评估 RMSNorm 替换后的PPL...")
    print("=" * 60)
    
    model.to(utils.DEV)
    after_rmsnorm_ppl_wiki = evaluate_ppl(model, args, "wikitext2")
    after_rmsnorm_ppl_c4 = evaluate_ppl(model, args, "c4")
    
    print(f"  After RMSNorm WikiText2 PPL: {after_rmsnorm_ppl_wiki:.4f}")
    print(f"  After RMSNorm C4 PPL: {after_rmsnorm_ppl_c4:.4f}")
    print()
    
    # 清理
    model.cpu()
    utils.cleanup_memory()
    
    # =========================================================================
    # 总结对比结果
    # =========================================================================
    print("=" * 60)
    print("对比结果总结")
    print("=" * 60)
    
    # WikiText2
    wiki_diff_fuse = abs(after_fuse_ppl_wiki - original_ppl_wiki) / original_ppl_wiki * 100
    wiki_diff_rmsnorm = abs(after_rmsnorm_ppl_wiki - original_ppl_wiki) / original_ppl_wiki * 100
    
    print(f"\nWikiText2:")
    print(f"  原始模型 PPL:           {original_ppl_wiki:.6f}")
    print(f"  fuse_modules 后 PPL:    {after_fuse_ppl_wiki:.6f} (变化: {after_fuse_ppl_wiki - original_ppl_wiki:+.6f}, {wiki_diff_fuse:.4f}%)")
    print(f"  RMSNorm替换后 PPL:      {after_rmsnorm_ppl_wiki:.6f} (变化: {after_rmsnorm_ppl_wiki - original_ppl_wiki:+.6f}, {wiki_diff_rmsnorm:.4f}%)")
    
    # C4
    c4_diff_fuse = abs(after_fuse_ppl_c4 - original_ppl_c4) / original_ppl_c4 * 100
    c4_diff_rmsnorm = abs(after_rmsnorm_ppl_c4 - original_ppl_c4) / original_ppl_c4 * 100
    
    print(f"\nC4:")
    print(f"  原始模型 PPL:           {original_ppl_c4:.6f}")
    print(f"  fuse_modules 后 PPL:    {after_fuse_ppl_c4:.6f} (变化: {after_fuse_ppl_c4 - original_ppl_c4:+.6f}, {c4_diff_fuse:.4f}%)")
    print(f"  RMSNorm替换后 PPL:      {after_rmsnorm_ppl_c4:.6f} (变化: {after_rmsnorm_ppl_c4 - original_ppl_c4:+.6f}, {c4_diff_rmsnorm:.4f}%)")
    
    # 判断结果
    tolerance = 1.0  # 1% 容差
    print(f"\n" + "=" * 60)
    print("结果判断")
    print("=" * 60)
    
    if wiki_diff_fuse < tolerance:
        print(f"✓ WikiText2 fuse_modules: PPL变化 {wiki_diff_fuse:.4f}% < {tolerance}% (通过)")
    else:
        print(f"✗ WikiText2 fuse_modules: PPL变化 {wiki_diff_fuse:.4f}% >= {tolerance}% (未通过)")
        
    if c4_diff_fuse < tolerance:
        print(f"✓ C4 fuse_modules: PPL变化 {c4_diff_fuse:.4f}% < {tolerance}% (通过)")
    else:
        print(f"✗ C4 fuse_modules: PPL变化 {c4_diff_fuse:.4f}% >= {tolerance}% (未通过)")
    
    print(f"\n  LayerNorm -> RMSNorm WikiText2变化: {wiki_diff_rmsnorm:.4f}%")
    print(f"  LayerNorm -> RMSNorm C4变化: {c4_diff_rmsnorm:.4f}%")
    
    # =========================================================================
    # 分析说明
    # =========================================================================
    print("\n" + "=" * 60)
    print("分析说明")
    print("=" * 60)
    print("fuse_modules 原理：")
    print("1. 将LayerNorm的缩放参数(γ)融合到相邻线性层的权重")
    print("2. 将去均值操作(x-μ)提前到Embedding层或融入输出层权重")
    print("3. 此步骤数学上等价，PPL应几乎不变")
    print()
    print("replace_layernorm_with_rmsnorm 原理：")
    print("1. 将LayerNorm模块替换为RMSNorm模块")
    print("2. RMSNorm不做去均值操作，只进行缩放")
    print("3. 由于fuse_modules已将去均值融入权重，此替换应保持等价性")
    print()
    print("预期效果：")
    print("- fuse_modules 后 PPL 变化应 < 1%（浮点运算舍入误差）")
    print("- RMSNorm 替换后 PPL 变化应 < 1%（因去均值已融入权重）")
    print()
    
    # 保存结果
    results = {
        "model": args.model,
        "original_ppl_wiki": original_ppl_wiki,
        "after_fuse_ppl_wiki": after_fuse_ppl_wiki,
        "after_rmsnorm_ppl_wiki": after_rmsnorm_ppl_wiki,
        "wiki_diff_fuse_percent": wiki_diff_fuse,
        "wiki_diff_rmsnorm_percent": wiki_diff_rmsnorm,
        "original_ppl_c4": original_ppl_c4,
        "after_fuse_ppl_c4": after_fuse_ppl_c4,
        "after_rmsnorm_ppl_c4": after_rmsnorm_ppl_c4,
        "c4_diff_fuse_percent": c4_diff_fuse,
        "c4_diff_rmsnorm_percent": c4_diff_rmsnorm,
        "wiki_fuse_pass": wiki_diff_fuse < tolerance,
        "wiki_rmsnorm_pass": wiki_diff_rmsnorm < tolerance,
        "c4_fuse_pass": c4_diff_fuse < tolerance,
        "c4_rmsnorm_pass": c4_diff_rmsnorm < tolerance,
    }
    
    import json
    output_file = f"fusion_ppl_comparison_{args.model.split('/')[-1]}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
