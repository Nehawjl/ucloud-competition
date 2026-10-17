# 来源与署名

项目作者：wangjl（GitHub：Nehawjl，645765686@qq.com）。

- 推理基础：[HyperQwen](https://github.com/syv-ai/HyperQwen)，固定版本见 `config/versions.env`。安装脚本下载完整上游源码并保留其 LICENSE。
- 推理运行时：[vLLM](https://github.com/vllm-project/vllm)，0.31.0，Apache-2.0。
- W4A4 kernel 与融合激活算子：[Humming](https://github.com/vllm-project/humming)，包版本 `humming-kernels==0.1.16`，Apache-2.0。本项目提供兼容性修改和模型前向路径接入，不将上游算子作为原创算法。
- 模型与 fast overlay：来源和精确 revision 见 README；使用时遵守对应模型仓库许可证。
- `third_party/official-eval/run.sh` 与 `runner.py` 为比赛方提供的评测工具原样副本，不属于我方推理实现；工具中的参考说明不代表我方成绩。
- `patches/` 为模型 revision 固定、Humming 编译兼容、SM89 attention 参数及融合接口补丁。补丁保留原文件已有版权标记。

本仓库不包含模型权重、原始实验文档、逐请求输出或实验数据归档。
