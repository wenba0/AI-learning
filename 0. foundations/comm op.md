reduce allreduce reduce-scatter的关系？以求和为例
+ reduce将所有卡上的结果求和后只保存在其中一张卡上
+ allreduce之后每张卡上都有相同的求和结果；相当于reduce