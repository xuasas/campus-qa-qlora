# 运行与复现指南

基于 Qwen2.5-0.5B-Instruct、QLoRA 与 Rust/egui 的校园资料问答研究项目。输入为“资料片段＋问题”，模型依据给定材料生成回答。比较 LoRA rank 4、8、16，alpha/r 固定为 2；Q0 为未做本项目领域微调的同量化指令基座。所有候选基座均采用 4 位 NF4 和双重量化，Q4 的数字指 rank，不是只有 Q4 使用 4 位量化。

这是经过筛选的公开副本。仅包含代码、参数模板、完全虚构的示例数据与去掉来源明细的历史汇总。没有真实校园原文、教师履历、学生记录、原始问答、论文、模型权重或个人运行日志。原始实验为 248/40/60 划分；公开示例为 16/4/4，不能用示例复现历史分数。

## 文件说明

- `scripts/`：配置生成、独立训练预算预实验、三种 rank 训练、统一给定资料生成、文本指标计算、验证集选择和推理桥接。
- `configs/templates/`：可公开参数模板；`configs/runtime/` 是本机生成目录，不能上传。
- `data/synthetic/`：24 条合成样本与来源字段示例，事实家族跨集合不重叠。
- `demo/`：合成演示题和仅由合成训练材料构建的 8 条片段库，不含缓存回答。
- `demo_rust/`：Windows Rust/egui 界面源码与锁定文件；不包含编译产物或系统字体。
- `results/historical_summary.json`：原实验的聚合结果，已移除本机路径、来源和文件摘要。
- `docs/`：数据、复现范围、脱敏及上传说明。

## 环境与准备

原实验使用 Windows、Python 3.11、RTX 3050 Ti Laptop 4GB、PyTorch 2.8.0+cu126。版本记录见 `requirements.txt`。先在自己的虚拟环境安装适合显卡驱动的 CUDA 版 PyTorch，再执行 `python -m pip install -r requirements.txt`。公开副本没有自动安装环境、下载模型或修补第三方包的脚本。

自行取得 Qwen2.5-0.5B-Instruct 本地目录（包括配置、分词器及 `model.safetensors`），遵守模型自身许可。该模型的本地原始模型卡标记为 Apache-2.0；本仓库不重新分发模型，也不替模型声明额外许可。项目代码的许可由仓库所有者确定，本次未代为添加开源授权条款。

从仓库根目录执行，替换尖括号内的参数：

```text
python scripts/prepare_local.py --model-path "<本地基座目录>"
python scripts/train_formal.py --rank 4 --check-only
```

`prepare_local.py` 只生成本机配置和摘要，不训练、不生成回答。默认使用合成数据。要使用自己的授权数据，按 `docs/DATA.md` 准备同样结构并传入 `--data-dir "<私有数据目录>"`。私有数据可放仓库外。生成目录禁止覆盖已有训练输出；重新实验应使用单独的工作副本。

## 训练与评测顺序

```text
# 可选预算预实验：在独立的 rank=8 运行上比较一轮和两轮
python scripts/select_training_budget.py

# 正式模板固定两轮；如要采用预实验选出的其他预算，应另建实验配置
python scripts/train_formal.py --rank 4
python scripts/train_formal.py --rank 8
python scripts/train_formal.py --rank 16
python scripts/evaluate_formal.py --split val
python scripts/score_formal_metrics.py --split val
python scripts/make_rating_template.py --split val
```

根据给定资料核对 `evaluations/runtime/val/ratings.json`，填写事实0–2、条件0–2、无编造0–1三项分数与总分，不可用参考回答冒充模型输出。必须披露实际评分方式。评分完成后：

```text
python scripts/select_validated_model.py --review-method human
python scripts/evaluate_formal.py --split test
python scripts/score_formal_metrics.py --split test
```

`--review-method` 取 `human`、`assistant` 或 `mixed`，应与实际一致。缺失评分会被拒绝，选择记录在测试预测前冻结，不因测试结果更好而改选。ROUGE-L 使用 jieba 分词后的逐题 LCS F1 均值乘100；BLEU-4 为整套回答的 corpus BLEU，不能平均逐句 BLEU。它们与0–5分分级评分均不等于业务准确率。

## Rust 本地界面

界面沿用 rank=4 的实验演示入口，不会自动把任意已选模型当作 Q4。先完成自己的 Q4 训练，再启动。它不是模型推荐器；验证选出其他配置时，应另行调整演示入口并保留记录。

```text
$env:QLORA_ROOT = (Get-Location).Path
$env:QLORA_PYTHON = (Get-Command python).Source
cargo run --locked --manifest-path demo_rust/Cargo.toml
```

当前源码使用 Windows 进程接口，按 Windows 演示版提供。需要 Rust 1.95 或以上及对应构建工具；中文字体从系统读取，不随项目分发。`QLORA_PYTHON` 必须指向装好推理依赖的环境。源代码直接编译运行时可从编译位置推断仓库根目录；移动程序后应设置 `QLORA_ROOT`。界面可在不加载模型时浏览合成资料，模型回答仍须实际加载并生成。

## 原实验结果与边界

历史60题给定资料测试中，Q0/Q4/Q8/Q16 的 ROUGE-L 为25.63/50.80/51.21/52.63，BLEU-4 为4.84/23.98/25.46/26.72，助手辅助均分为2.983/4.050/3.983/4.017。验证集先选定Q4；未实施独立真人盲评，只有一个训练种子，Q0有4条输出达到256 token上限。原始逐题数据不公开，因此公开汇总无法独立验证全部原始评分。

这些数值属于原私有数据上的既有实验，不能当成公开合成数据的运行结果。系统仍可能出现计算错误、条件遗漏和资料误配；关键词匹配演示不构成独立端到端准确率评价。训练时间包含入口内部准备、训练、验证和保存；显存峰值从训练开始回调重置后记录，不含此前加载峰值。

## 文件安全检查

运行 `python scripts/check_public_files.py`，并阅读 `docs/UPLOAD.md`。整个公开目录首次上传不含 `.git` 历史、凭据、私有数据或权重；使用项目后生成的日志和本机配置也不能上传。`.gitignore` 对网页手工选文件不生效，请只选本次公开副本原有的文件。
