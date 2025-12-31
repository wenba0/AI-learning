以Qwen3-30B-A3B为例

在MOE之前：
```python
# 输入模型input_ids #1*18   通过embedding将其转化为input_embeds 1*18*2048
inputs_embeds = self.embed_tokens(input_ids)
# 通过rotary_embedding将其转化为(1*18*128, 1*18*128)
position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

```