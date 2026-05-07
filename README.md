# Role-Augmented Intent-Driven Generative Search Engine Optimization (RAID G-SEO)
## 项目简介

我们扩展了 GEO 的测试数据集，使用 GPT-4 为每个原始查询生成了 4 个相关变体，以模拟更加多样化和具有挑战性的查询场景。

我们采用了与 GEO 相同的实验环境设置，并进一步更新了用于 *subjective impression* 评估的 G-EVAL Prompt。相比 GEO 原始工作，我们的 Prompt 提供了更加清晰的评分标准，并特别考虑了 source 未被 LLM 响应引用时记为 0 分的情况。

此外，我们提出了基于检索意图的 RAID G-SEO 方法。该方法采用“内容摘要 → 意图提取与精炼 → 步骤生成 → 实施优化”的四阶段框架，并在意图精炼阶段引入了基于多角色视角的 4W 深度反思机制，以增强意图表示的泛化能力，从而覆盖更多潜在的检索需求。

实验主要基于 GLM-4-9B-0414 模型完成。

## 参考工作

```bibtex
@misc{aggarwal2023geo,
      title={GEO: Generative Engine Optimization}, 
      author={Pranjal Aggarwal and Vishvak Murahari and Tanmay Rajpurohit and Ashwin Kalyan and Karthik R Narasimhan and Ameet Deshpande},
      year={2023},
      eprint={2311.09735},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## 引用

如果你觉得我们的工作对你的研究有帮助，欢迎引用：

```bibtex
@misc{chen2026roleaugmentedintentdrivengenerativesearch,
      title={Role-Augmented Intent-Driven Generative Search Engine Optimization}, 
      author={Xiaolu Chen and Haojie Wu and Jie Bao and Zhen Chen and Yong Liao and Hu Huang},
      year={2026},
      eprint={2508.11158},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2508.11158}, 
}
```
