### ⚠️ Plus版本起不再单独评测Q4版本的模型

# 中文Alpaca-7B/13B量化版本输出示例

为了快速评测相关模型的实际表现，本项目在给定相同的prompt的情况下，在一些常见任务上对比测试了本项目的中文Alpaca-7B和中文Alpaca-13B的效果。生成回复具有随机性，受解码超参、随机种子等因素影响。以下相关评测并非绝对严谨，测试结果仅供晾晒参考，欢迎自行体验。

说明：

- 以下分数应视为paired score，也就是说分数是一个相对值，而不是绝对值，是两个系统相比较得到的结果
- 基于以上说明，分数之间的大小关系有一些参考价值，而分数的绝对值没有太大参考价值
- 除多轮任务之外，所有任务均基于单轮回复进行打分（不包含任何对话历史）
- 每个样例运行2-3次，人工选取最好的一组交给[机器评分](#打分方式)以降低随机性带来的偏差

⚠️ *以下测试结果均基于**4-bit量化模型**，理论效果比非量化版本差一些。*

| 测试任务         |                详细样例                | 样例数 | 中文Alpaca-7B | 中文Alpaca-13B |
| ---------------- | :------------------------------------: | :----: | :-----------: | :------------: |
| **💯总平均分**    |                   -                    |  160   |    **49**     |    **👍🏻71**    |
| 知识问答         |            [QA.md](./QA.md)            |   20   |      53       |    **👍🏻77**    |
| 开放式问答       |           [OQA.md](./OQA.md)           |   20   |      64       |    **👍🏻73**    |
| 数值计算、推理   |     [REASONING.md](./REASONING.md)     |   20   |      23       |    **👍🏻50**    |
| 诗词、文学、哲学 |    [LITERATURE.md](./LITERATURE.md)    |   20   |      31       |    **👍🏻54**    |
| 音乐、体育、娱乐 | [ENTERTAINMENT.md](./ENTERTAINMENT.md) |   20   |      36       |    **👍🏻65**    |
| 写信、写文章     |    [GENERATION.md](./GENERATION.md)    |   15   |      65       |    **👍🏻78**    |
| 文本翻译         |   [TRANSLATION.md](./TRANSLATION.md)   |   15   |      63       |    **👍🏻79**    |
| 多轮交互         |      [DIALOGUE.md](./DIALOGUE.md)      |   10   |      80       |    **👍🏻83**    |
| 代码编程         |          [CODE.md](./CODE.md)          |   10   |      27       |    **👍🏻49**    |
| 伦理、拒答       |        [ETHICS.md](./ETHICS.md)        |   10   |      50       |   **👍🏻100**    |

#### 运行参数

测试中使用了统一的解码参数（可能并不适合所有任务）：
```bash
./main -m zh-alpaca-models/7B/ggml-model-q4_0.bin --color -f ./prompts/alpaca.txt -ins \
  -b 24 -c 2048 -n 512 -t 6 \
  --temp 0.2 --top_k 40 --top_p 0.9 \
  --repeat_penalty 1.3
```


#### 打分方式

- 一共10组任务，每组任务满分100分；每组任务10-20个样例，每个样例满分10分
- 样例的得分之和规整到100分区间作为该模型在该任务上的得分
- 使用GPT-4和ChatGPT（GPT-3.5）对两个系统的输出进行打分（10分制），模板如下：

```
The followings are two ChatGPT-like systems' outputs. Please rate an overall score on a ten point scale for each and give explanations to justify your scores.

Prompt:
<prompt-input>

System1:
<system1-output>

System2:
<system2-output>
```

*注：优先使用GPT-4打分。由于GPT-4的交互次数限制，一部分打分由ChatGPT进行。*
