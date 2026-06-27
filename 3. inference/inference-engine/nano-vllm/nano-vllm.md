调度
优先调度prefill，

即waiting队列，如果budgets不够导致无法调度，则调度
对于每个新的请求，都需要经过1 `can_allocate`计算出是否能被调度，以及prefix cache命中情况 2 `allocate`来实际分配block并放到block_table中


![1314](../../../assets/file-20260627170627918.png)