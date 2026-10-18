# 来源与署名

项目作者：王嘉乐、毛俊琪、吴松灿。

- 推理基础：[HyperQwen](https://github.com/syv-ai/HyperQwen)，固定版本见 `config/versions.env`。安装脚本下载完整上游源码并保留其 LICENSE。
- 推理运行时：[vLLM](https://github.com/vllm-project/vllm)，0.31.0，Apache-2.0。
- W4A4 kernel 与融合激活算子：[Humming](https://github.com/vllm-project/humming)，包版本 `humming-kernels==0.1.16`，Apache-2.0。本项目提供兼容性修改和模型前向路径接入，不将上游算子作为原创算法。
- 模型与 fast overlay：来源和精确 revision 见 README；使用时遵守对应模型仓库许可证。
- `third_party/eval.zip` 为比赛方提供的原始评测包，SHA-256 为 `ecc1296a5773c12b1acf5ad137b7bc9ea88f55e7ec82a34e93c7fd00ffad36ad`。`third_party/official-eval/` 中 README、run.sh、runner.py 与包内文件逐字节一致，不属于我方推理实现；工具中的参考说明不代表我方成绩。
- `patches/` 为模型 revision 固定、Humming 编译兼容、SM89 attention 参数及融合接口补丁。补丁保留原文件已有版权标记。

本仓库不包含模型权重、完整实验历史或虚拟环境。答辩展示对应的精简逐请求记录和评测配置见 `evidence/`。
