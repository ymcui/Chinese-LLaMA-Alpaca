# 效果对比：Alpaca-Plus-7B、Plus-13B、33B

为了快速评测相关模型的实际表现，本项目在给定相同的prompt的情况下，在一些常见任务上对比测试了本项目的中文Alpaca-Plus-7B、中文Alpaca-Plus-13B、中文Alpaca-33B的效果。生成回复具有随机性，受解码超参、随机种子等因素影响。以下相关评测并非绝对严谨，测试结果仅供晾晒参考，欢迎自行体验。

⚠️ *Alpaca-Plus-7B和Alpaca-Plus-13B结果沿用之前的生成结果。33B除多轮对话之外的任务由huggingface原生接口进行解码（F16），多轮对话由llama.cpp解码（q4_0量化版）。*

| 测试任务         |                详细样例                | 样例数 | Alpaca-Plus-7B | Alpaca-Plus-13B | Alpaca-33B |
| ---------------- | :------------------------------------: | :----: | :------------: | :-------------: | :--------: |
| **💯总平均分**    |                   -                    |  200   |      75.3      |      79.4       | 👍🏻**82.0** |
| 知识问答         |            [QA.md](./QA.md)            |   20   |      70.5      |      79.5       | 👍🏻**82.3** |
| 开放式问答       |           [OQA.md](./OQA.md)           |   20   |   👍🏻**80.5**   |    👍🏻**80**     |    78.5    |
| 数值计算、推理   |     [REASONING.md](./REASONING.md)     |   20   |       51       |      61.5       | 👍🏻**84.5** |
| 诗词、文学、哲学 |    [LITERATURE.md](./LITERATURE.md)    |   20   |      78.5      |   **👍🏻81.3**    |     76     |
| 音乐、体育、娱乐 | [ENTERTAINMENT.md](./ENTERTAINMENT.md) |   20   |      72.3      |   👍🏻**76.8**    |    72.5    |
| 写信、写文章     |    [GENERATION.md](./GENERATION.md)    |   20   |       81       |   👍🏻**86.5**    |     79     |
| 文本翻译         |   [TRANSLATION.md](./TRANSLATION.md)   |   20   |      86.8      |      89.3       | 👍🏻**92.3** |
| 多轮交互         |      [DIALOGUE.md](./DIALOGUE.md)      |   20   |      80.3      |   👍🏻**81.3**    |     78     |
| 代码编程         |          [CODE.md](./CODE.md)          |   20   |      62.5      |      67.5       | 👍🏻**84.0** |
| 伦理、拒答       |        [ETHICS.md](./ETHICS.md)        |   20   |      89.8      |      90.5       | 👍🏻**92.5** |

说明：

- 以上分数应视为paired score，也就是说分数是一个相对值，而不是绝对值，是多个系统相比较得到的结果
- 基于以上说明，分数之间的大小关系有一些参考价值，而分数的绝对值没有太大参考价值
- 除多轮任务之外，所有任务均基于单轮回复进行打分（不包含任何对话历史）
- 每个样例运行2-3次，人工选取最好的一组交给[机器评分](#打分方式)以降低随机性带来的偏差

#### 运行参数

测试中使用了统一的解码参数，以下是llama.cpp的示例：
```bash
./main -m zh-alpaca-models/{Plus-7B,Plus-13B}/ggml-model-q8_0.bin --color -f ./prompts/alpaca.txt -ins \
  -b 16 -c 2048 -n 512 -t 6 \
  --temp 0.2 --top_k 40 --top_p 0.9 \
  --repeat_penalty 1.1
```

*注：可能并不适合所有任务。实际使用时，对话、写作类等自由生成类任务可适当调高temp。*

#### 打分方式

- 一共10组任务，每组任务满分100分；每组任务20个样例，每个样例满分10分
- 样例的得分之和规整到100分区间作为该模型在该任务上的得分
- 使用GPT-4和ChatGPT（GPT-3.5）对两个系统的输出进行打分（10分制），模板如下：

```
The followings are ChatGPT-like systems' outputs based on a single prompt. Please rate an overall score on a ten point scale for each system and give a short explanation to justify your scores. Please try not to give the same scores for different system unless they are indistinguishable.

Prompt:
<prompt-input>

System1:
<system1-output>

System2:
<system2-output>
```

*注：优先使用GPT-4打分。由于GPT-4的交互次数限制，一部分打分由ChatGPT（gpt-3.5-turbo）进行。*
