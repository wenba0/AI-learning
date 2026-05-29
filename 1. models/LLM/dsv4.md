### mHC
**问题背景及方法：**
![](file-20260529155556587.png)
针对残差连接变体中常见的缺点，例如梯度消失与表示崩溃之间的跷跷板效应。本论文提出超链接方法，从理论上讲，超连接允许网络调整不同深度特征之间的连接强度，并动态地重新排列网络层
	1 无约束的 HC 架构破坏了残差连接的“恒等映射”（Identity Mapping）属性，导致深层网络中的信号传播仍然发散（爆炸或消失），且宽残差流带来了严重的显存访问（I/O）瓶颈；同时超链接只在小LLM模型进行验证，无法说明在多模态理解和生成等场景的泛化性。
	2 HC会带来的不稳定问题: 由于原始HC无约束，这会导致正反向计算过程中，该映射实际上会偏离恒等映射，一旦层数过深，就可能导致信号在前反向传播过程中发生爆炸或消失。可以观察到，单层和多层的HC，其正反向的Amax Gain Magnitude会被放大很多。
![888](file-20260529155540869.png)
![[Pasted image 20260515104541.png|898]]
类比为一个城市的交通系统
+ 残差链接(图a)：一条稳定主干道。不管你的车（信息）有多少，都只能排队走这一条道。模型越大、任务越复杂，这条单车道就越拥挤。  
+ HC(图b)：既然一条车道不够，那我们修多条车道.但没有任何交通规则。有的车道堵死了，有的车道上无车。最后整个高速公路系统瘫痪。  
+ mHC(图c)：给这些新修的路加上交通法规和要求限制，例如：车可以走不同路线，但一条道路上的总车流量不能突然翻倍，也不允许凭空多出车辆，更不允许某个区域突然被抽空。这个规则就是双随机矩阵。  
mHC学术做法是，通过Sinkhorn-Knopp算法让任意矩阵变成双随机矩阵。  
	什么是双随机矩阵？每一行的和均为1、每一列的和均为1，所有元素均为非负  
	什么是Sinkhorn-Knopp算法？先对矩阵取指数保证非负，交替对行和列做归一化，重复20次，该矩阵就近似收敛为双随机矩阵

原理介绍
```
对于hc, pre输入映射，将多流加权变成一个送给Layer（attn或mlp）,post将Layer的输出变成多流，然后与残差进行求和
Pre:  layer_input = Σ_i （α_i · residual_i）     ← α_i 是 per-stream scalar gate
Post: residual_new_j = β_j · x + residual_j    ← β_j 也是 scalar gate
问题：流之间不互通！

mhc: comb_mix是个双随机矩阵，
Pre:  仍产出 post_mix, comb_mix, layer_input
Post:  residual_new_j = post_mix_j · x  +  Σ_i comb_mix_ij · residual_i
                       ↑ Post Mapping       ↑ Res Mapping (新！)
```

**vllm的实现**
首层输入的hidden_states会复制config.hc_mult份，然后分别在attn和mlp前面执行pre，后面执行post,其中res_mapping的操作已经融合到了post中，对于cuda场景将post与pre通过mhc_fused_post_pre融合到了一起，amd的rocm场景pre和post是分开的，vllm中这几个算子都是tilelang的
```python
	def _forward_cuda(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual is None:
            # Run standalone hc_pre on first layer
            residual = x
            x, post_mix, res_mix = self.hc_pre(
                x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
        else:
            residual, post_mix, res_mix, x = self.mhc_fused_post_pre(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
            )
        x = self.attn_norm(x)
        x = self.attn(positions, x, None)
        residual, post_mix, res_mix, x = self.mhc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix

    def _forward_rocm(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(positions, x, None)
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x, None, None, None
```

CUDA vs ROCm 路径差异

| 特性           | CUDA (forward_cuda)                                                                                                                                                                                                                                                                                                                         | ROCm (forward_rocm)      |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| Pre 实现       | TileLang JIT kernel                                                                                                                                                                                                                                                                                                                         | AITER / PyTorch fallback |
| Post 实现      | TileLang                                                                                                                                                                                                                                                                                                                                    | AITER / PyTorch          |
| FusedPostPre | ✅ 有（fused kernel）                                                                                                                                                                                                                                                                                                                           | ❌ 无（分解为 Post + Pre 两步）   |
| 流管理          | 3 个 aux CUDA stream 并行 GEMM                                                                                                                                                                                                                                                                                                                 | 无 aux stream（避免 hang）    |
| 首层特殊处理       | [residual=None](vscode-file://vscode-app/d:/software/vscode/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 时先跑 standalone [hc_pre](vscode-file://vscode-app/d:/software/vscode/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) | 同左                       |
