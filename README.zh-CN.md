# CS336：从零构建语言模型

[English](README.md) | 简体中文

本仓库记录我在 [Stanford CS336](https://cs336.stanford.edu/) 中的课程作业、
代码实现和学习笔记。

## 仓库结构

```text
cs336/
├── assignment1-basics/     # 分词器、Transformer LM、优化器和训练工具
├── assignment2-systems/    # 后续作业预留目录
├── assignment3-scaling/    # 后续作业预留目录
├── assignment4-data/       # 后续作业预留目录
└── assignment5-alignment/  # 后续作业预留目录
```

## Assignment 1：Basics

`assignment1-basics/` 包含 Assignment 1 的实现，主要包括：

- 字节级 BPE 训练与预分词
- 支持特殊 token 和流式编码的 Tokenizer
- Linear、Embedding、RMSNorm、SwiGLU、RoPE 与注意力模块
- Pre-norm Transformer Block 与 Transformer 语言模型
- Cross-entropy、AdamW、学习率调度与梯度裁剪
- 批次采样以及模型、优化器 checkpoint 的保存与加载

这些实现会通过课程提供的单元测试进行验证。

### 环境与测试

项目使用 [`uv`](https://docs.astral.sh/uv/) 管理环境：

```bash
cd assignment1-basics
uv run pytest
```

### BPE 训练命令行

`assignment1-basics/scripts/train_bpe_cli.py` 是 BPE 训练代码的命令行入口。

使用仓库自带调试语料的示例：

```bash
cd assignment1-basics
uv run python scripts/train_bpe_cli.py \
  tests/fixtures/tinystories_sample_5M.txt \
  --vocab-size 500 \
  --special-token '<|endoftext|>' \
  --output-dir runs/debug-ts
```

运行时将生成：

- `vocab.json`：token ID 到 Base64 编码字节串的映射
- `merges.txt`：按创建顺序保存的 BPE 合并记录，每项的字节串均使用 Base64 编码

这两个文件保留了训练得到的词表和合并顺序，便于之后检查或重新加载。

## 学习笔记

BPE 训练流程和 pair 索引优化的实现笔记位于
[`assignment1-basics/notes/bpe.md`](assignment1-basics/notes/bpe.md)。

## 数据与生成产物

数据集、虚拟环境、checkpoint、token ID 数组、日志和实验输出不会纳入版本
控制，以避免仓库中出现体积过大的文件。

## 致谢

作业说明、初始代码和测试夹具来自 Stanford CS336 课程团队。
