# 公开数据与私有数据结构

`data/synthetic/` 为独立创作的完全虚构示例，没有从真实教师、通知或学生记录替换姓名后拼接而成。日期、金额、学分、教师甲乙及规则都是合成内容，不能作为真实办事依据。训练16条、验证4题、测试4题，仅用于文件格式、划分和程序通路示例，不足以支持有效模型效果结论。

训练文件 `campus_train.json`、`campus_val.json`、`campus_test.json` 为JSON数组，每条只含 `instruction`、`input`、`output`。`dataset_info.json` 定义这些字段与LLaMA-Factory的对应。参考回答只用于监督训练或生成后的评价，不直接拼进推理提示。

同名集合的 `train_metadata.json`、`val_metadata.json`、`test_metadata.json` 在三字段外保留 `id`、`split_group_id`、`topic`、`question`、`context`、`source_file`、`source_locator`。同一事实的不同问法应使用同一 `split_group_id`；跨集合不可共享事实家族或相同问题。示例另有 `synthetic=true` 标志，私有真实数据不应伪称合成。

使用自己的资料时，须在本地替换整套一致的数据和元数据，保留 `dataset_info.json` 与 `split_manifest.json`；先检查依据、适用条件及划分，再运行 `prepare_local.py --data-dir`。不要将姓名替换成编号后就默认匿名，正文、参考回答、来源文件名、教师履历、联系方式与交叉关联仍可能识别人。

`demo/knowledge.json` 只保存合成训练材料的原文片段与问题别名，不保存参考答案或缓存预测；自动匹配是字符二元组重合度，取别名分与0.75倍片段分的较大者，阈值0.20、最多3条。检索能力需用另行留出的真实独立查询评价。
