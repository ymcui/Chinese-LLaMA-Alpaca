[**🇨🇳中文**](./README.md) | [**🌐English**](./README_EN.md) | [**📖文档/Docs**](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki) | [**❓提问/Issues**](https://github.com/ymcui/Chinese-LLaMA-Alpaca/issues) | [**💬讨论/Discussions**](https://github.com/ymcui/Chinese-LLaMA-Alpaca/discussions)

<p align="center">
    <br>
    <img src="./pics/banner.png" width="700"/>
    <br>
</p>
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-LLaMA-Alpaca.svg?color=blue&style=flat-square">
    <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/ymcui/Chinese-LLaMA-Alpaca">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ymcui/Chinese-LLaMA-Alpaca">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/ymcui/Chinese-LLaMA-Alpaca">
    <a href="https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki"><img alt="GitHub wiki" src="https://img.shields.io/badge/Github%20Wiki-v3.2-green"></a>
</p>



Large Language Models (LLM), represented by ChatGPT and GPT-4, have sparked a new wave of research in the field of natural language processing, demonstrating capabilities of Artificial General Intelligence (AGI) and attracting widespread attention from the industry. However, the expensive training and deployment of large language models have posed certain obstacles to building transparent and open academic research.

To promote open research of large models in the Chinese NLP community, this project has open-sourced the **Chinese LLaMA model and the Alpaca large model with instruction fine-tuning**. These models expand the Chinese vocabulary based on the original LLaMA and use Chinese data for secondary pre-training, further enhancing Chinese basic semantic understanding. Additionally, the project uses Chinese instruction data for fine-tuning on the basis of the Chinese LLaMA, significantly improving the model's understanding and execution of instructions. Please refer to our technical report for further details [(Cui, Yang, and Yao, 2023)](https://arxiv.org/abs/2304.08177).

**Main contents of this project:**

- 🚀 Extended Chinese vocabulary on top of original LLaMA with significant encode/decode efficiency
- 🚀 Open-sourced the Chinese LLaMA (general purpose) and Alpaca (instruction-tuned)   (7B, 13B)
- 🚀 Open-sourced the pre-training and instruction finetuning (SFT) scripts for further tuning on user's data
- 🚀 Quickly deploy and experience the quantized version of the large model on CPU/GPU of your laptop (personal PC) 
- 🚀 Support [🤗transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp), [text-generation-webui](https://github.com/oobabooga/text-generation-webui), [LlamaChat](https://github.com/alexrozanski/LlamaChat), [LangChain](https://github.com/hwchase17/langchain), , [privateGPT](https://github.com/imartinez/privateGPT), etc.
- Released versions: 7B (basic, **Plus**), 13B (basic, **Plus**)

💡 The following image shows the actual experience effect of the 7B version model after local deployment (animation unaccelerated, tested on Apple M1 Max).

![](./pics/screencast.gif)

----

[Multi-modal VLE](https://github.com/iflytek/VLE) | [Chinese MiniRBT](https://github.com/iflytek/MiniRBT) | [Chinese LERT](https://github.com/ymcui/LERT) | [Chinese-English PERT](https://github.com/ymcui/PERT) | [Chinese MacBERT](https://github.com/ymcui/MacBERT) | [Chinese ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [Chinese XLNet](https://github.com/ymcui/Chinese-XLNet) | [Chinese BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [Knowledge distillation tool TextBrewer](https://github.com/airaria/TextBrewer) | [Model pruning tool TextPruner](https://github.com/airaria/TextPruner)

## News

**[May 16, 2023] [Release v3.2](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v3.2): Add SFT scripts, LangChain supports, Gradio-based web demo, etc.**

[May 10, 2023] [Release v3.1](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v3.1): LLaMA/Alpaca Plus 13B versions are available, more training data used than basic ones.

[Apr 28, 2023] [Release v3.0](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v3.0): LLaMA/Alpaca Plus versions are available, more training data used than basic ones. 

<details>
<summary><b>Previous News</b></summary>

[Apr 18, 2023] Release v2.2: Add LlamaChat support (macOS UI), tokenizer merging scripts, documentations are migrated to GitHub Wiki. Refer to [Release Note](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v2.2)

[Apr 13, 2023] Release v2.1: Add HuggingFace-transformers and text-generation-webui interfances. Refer to [Release Note](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v2.1)

[Apr 07, 2023] Release v2.0: Release 13B versions of Chinese LLaMA and Alpaca model. Main upgrades: stronger factuality, better performance on QA, translation and more. Refer to [Release Note](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v2.0)

[Mar 31, 2023] Release v1.1, major updates: simplification of model merging steps, addition of instruction data crawling script, and important notes about the new version of llama.cpp. See [Release Note](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v1.1).

[Mar 28, 2023]  Open-sourcing Chinese LLaMA and Alpaca, currently offering the 7B version for download and experience 

</details>

## Content Navigation

| Chapter                                       | Description                                                  |
| --------------------------------------------- | ------------------------------------------------------------ |
| [Download](#Download)                         | Download links for Chinese LLaMA and Alpaca                  |
| [Model Reconstruction](#Model-Reconstruction) | (Important) Explains how to merge downloaded LoRA models with the original LLaMA |
| [Quick Deployment](#Quick=Deployment)         | Steps for quantize and deploy LLMs on personal computers     |
| [Example Results](#Example-Results)           | Examples of the system output                                |
| [Training Details](#Training-Details)         | Introduces the training details of Chinese LLaMA and Alpaca  |
| [FAQ](#FAQ)                                   | Replies to some common questions                             |
| [Limitations](Limitations)                    | Limitations of the models involved in this project           |

## Model Download

### ⚠️ User Notice (Must Read)

The official [LLaMA models released by Facebook prohibit commercial use](https://github.com/facebookresearch/llama), and the official model weights have not been open-sourced (although there are many third-party download links available online). In order to comply with the relevant licenses, it is currently not possible to release the complete model weights. We appreciate your understanding. After Facebook fully opens up the model weights, this project will update its policies accordingly. **What is released here are the LoRA weights**, which can be seen as a "patch" for the original LLaMA model, and the complete weights can be obtained by merging the two.

### Which model should I use?

The following table provides a basic comparison of the Chinese LLaMA and Alpaca models, as well as recommended usage scenarios (including, but not limited to). 

💡 **Plus versions** are trained on more data, which is highly recommended for use.

| Comparison Item                                         | Chinese LLaMA                                                | Chinese Alpaca                                               |
| ------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Training Method                                         | Traditional CLM (trained on general corpus)                  | Instruction Fine-tuning (trained on instruction data)        |
| Training Data                                           | unsupervised free text                                       | supervised instruction data                                  |
| Vocab size<sup>[3]</sup>                                | 4995**3**                                                    | 4995**4**=49953+1 (pad token)                                |
| Input Template                                          | Not required                                                 | Must meet template requirements<sup>[1]</sup>                |
| Suitable Scenarios ✔️                                    | Text continuation: Given a context, let the model continue writing | 1. Instruction understanding (Q&A, writing, advice, etc.)<br/>2. Multi-turn context understanding (chat, etc.) |
| Unsuitable Scenarios ❌                                  | Instruction understanding, multi-turn chat, etc.             | Unrestricted free text generation                            |
| llama.cpp                                               | Use `-p` parameter to specify context                        | Use `-ins` parameter to enable instruction understanding + chat mode |
| text-generation-webui                                   | Not suitable for chat mode                                   | Use `--cpu` to run without a GPU; if not satisfied with generated content, consider modifying prompt |
| LlamaChat                                               | Choose "LLaMA" when loading the model                        | Choose "Alpaca" when loading the model                       |
| [inference_hf.py](./scripts/inference_hf.py)            | No additional startup parameters required                    | Add `--with_prompt` parameter when launching                 |
| [web-demo](./scripts/gradio_demo.py)                    | Not applicable                                               | Simply provide the Alpaca model location; support multi-turn conversations |
| [LangChain-demo](./scripts/langchain_demo) / privateGPT | Not applicable                                               | Simply provide the Alpaca model location                     |
| Known Issues                                            | If not controlled for termination, it will continue writing until reaching the output length limit.<sup>[2]</sup> | Current version of the model generates relatively shorter texts, being more concise.<sup>[2]</sup> |

*[1] Templates are built-in for (llama.cpp/LlamaChat/[inference_hf.py](./scripts/inference_hf.py)/[web-demo](./scripts/gradio_demo.py)/[LangChain-demo](./scripts/langchain_demo).*

*[2] If you encounter issues such as low-quality model responses, nonsensical answers, or failure to understand questions, please check whether you are using the correct model and startup parameters for the scenario.*

*[3] Alpaca model has an additional pad token in vocabulary than LLaMA. **Please do not mix LLaMA/Alpaca tokenizers**.*


### Chinese LLaMA

The Chinese LLaMA model has expanded the Chinese vocabulary on the basis of the original version, and used Chinese plain text data for secondary pre-training. For details, see the [Training Details](#Training-Details) section.

| Model             |  Type   | Required Original Model<sup>[1]</sup> | Size<sup>[2]</sup> |                        Download Links<sup>[3]</sup>                        |
| :---------------- | :-----: | :---------------------: | :----------------: | :----------------------------------------------------------: |
| Chinese-LLaMA-7B  | general 20G | LLaMA-7B  |        770M        | [[BaiduDisk]](https://pan.baidu.com/s/1oORTdpr2TvlkxjpyWtb5Sw?pwd=33hb)</br>[[Google Drive]](https://drive.google.com/file/d/1iQp9T-BHjBjIrFWXq_kIm_cyNmpvv5WN/view?usp=sharing) |
| Chinese-LLaMA-Plus-7B ⭐️ | general 120G |      LLaMA-7B      |        790M        | [[BaiduDisk]](https://pan.baidu.com/s/1zvyX9FN-WSRDdrtMARxxfw?pwd=2gtr)</br>[[Google Drive]](https://drive.google.com/file/d/1N97m3rBj-rp-J1X8rgRfluyomEscfAq0/view?usp=sharing) |
| Chinese-LLaMA-13B | general 20G | LLaMA-13B |         1G         | [[BaiduDisk]](https://pan.baidu.com/s/1BxFhYhDMipW7LwI58cGmQQ?pwd=ef3t)<br/>[[Google Drive]](https://drive.google.com/file/d/12q9EH4mfKRnoKlbkkhzv1xDwWnroo9VS/view?usp=sharing) |
| Chinese-LLaMA-Plus-13B ⭐️ | general 120G | LLaMA-13B | 1G | [[BaiduDisk]](https://pan.baidu.com/s/1VGpNlrLx5zHuNzLOcTG-xw?pwd=8cvd)<br/>[[Google Drive]](https://drive.google.com/file/d/1q0L5Me_1j_9iiRRNfuEFUt3SOjQo3-g3/view?usp=share_link) |

### Chinese Alpaca

The Chinese Alpaca model further uses instruction data for fine-tuning on the basis of the above-mentioned Chinese LLaMA model. For details, see the [Training Details](#Training-Details) section.

**⚠️ Please use Alpaca model if you want to try ChatGPT-like model.**

| Model                     |       Type       |        Required Original Model<sup>[1]</sup>         | Size<sup>[2]</sup> |                 Download Links<sup>[3]</sup>                 |
| :------------------------ | :--------------: | :--------------------------------------------------: | :----------------: | :----------------------------------------------------------: |
| Chinese-Alpaca-7B         |  Instruction 2M  |                       LLaMA-7B                       |        790M        | [[BaiduDisk]](https://pan.baidu.com/s/1xV1UXjh1EPrPtXg6WyG7XQ?pwd=923e)</br>[[Google Drive]](https://drive.google.com/file/d/1JvFhBpekYiueWiUL3AF1TtaWDb3clY5D/view?usp=sharing) |
| Chinese-Alpaca-Plus-7B ⭐️  |  Instruction 4M  | *LLaMA-7B &<br/>Chinese-LLaMA-Plus-7B*<sup>[4]</sup> |        1.1G        | [[百度网盘]](https://pan.baidu.com/s/12tjjxmDWwLBM8Tj_7FAjHg?pwd=32hc)</br>[[Google Drive]](https://drive.google.com/file/d/1EDcTmq6tDmRxqarpapdyDGBE9opY0zrB/view?usp=share_link) |
| Chinese-Alpaca-13B        |  Instruction 3M  |                       LLaMA-7B                       |        1.1G        | [[BaiduDisk]](https://pan.baidu.com/s/1wYoSF58SnU9k0Lndd5VEYg?pwd=mm8i)<br/>[[Google Drive]](https://drive.google.com/file/d/1gzMc0xMCpXsXmU1uxFlgQ8VRnWNtDjD8/view?usp=share_link) |
| Chinese-Alpaca-Plus-13B ⭐️ | Instruction 4.3M | *LLaMA-7B &<br/>Chinese-LLaMA-Plus-7B<sup>[4]</sup>* |        1.3G        | [[百度网盘]](https://pan.baidu.com/s/1Mew4EjBlejWBBB6_WW6vig?pwd=mf5w)<br/> [[Google Drive]](https://drive.google.com/file/d/1CcLJvY7XsFAOjfSIqCpDI7jf3EEPDcEF/view?usp=share_link) |

### Model Hub

You can download all the above models in 🤗Model Hub, and use [🤗transformers](https://github.com/huggingface/transformers) and [🤗PEFT](https://github.com/huggingface/peft) to call Chinese LLaMA or the Alpaca LoRA model.

| Model              |             MODEL_NAME             |                             Link                             |
| ------------------ | :--------------------------------- | :----------------------------------------------------------: |
| Chinese-LLaMA-7B        | ziqingyang/chinese-llama-lora-7b        | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-llama-lora-7b) |
| Chinese-LLaMA-Plus-7B   | ziqingyang/chinese-llama-plus-lora-7b   | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-llama-plus-lora-7b) |
| Chinese-LLaMA-13B       | ziqingyang/chinese-llama-lora-13b       | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-llama-lora-13b) |
| Chinese-LLaMA-Plus-13B  | ziqingyang/chinese-llama-plus-lora-13b  | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-llama-plus-lora-13b) |
| Chinese-Alpaca-7B       | ziqingyang/chinese-alpaca-lora-7b       | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-alpaca-lora-7b) |
| Chinese-Alpaca-Plus-7B  | ziqingyang/chinese-alpaca-plus-lora-7b  | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-alpaca-plus-lora-7b) |
| Chinese-Alpaca-13B      | ziqingyang/chinese-alpaca-lora-13b      | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-alpaca-lora-13b) |
| Chinese-Alpaca-Plus-13B | ziqingyang/chinese-alpaca-plus-lora-13b | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-alpaca-plus-lora-13b) |


### Footnote and Others

**[1]** The original LLaMA model needs to be applied for use in [Facebook-LLaMA](https://github.com/facebookresearch/llama) or refer to this [PR](https://github.com/facebookresearch/llama/pull/73/files). Due to copyright issues, this project cannot provide downloads, and we ask for your understanding.

**[2]** The reconstructed model is slightly larger than the original LLaMA (due to the expanded vocabulary); the 7B model is about 13G+.

**[3]** After downloading, be sure to check whether the SHA256 of the ZIP file is consistent; for the full value, please see [SHA256.md](./SHA256.md).

**[4]** Merging steps for Alpaca-Plus are different from others, please refer to [wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Manual-Conversion#multiple-lora-weights-merging-applicable-to-chinese-alpaca-plus)。

The file directory inside the ZIP file is as follows (using Chinese-LLaMA as an example):

```
chinese_llama_lora_7b/
  - adapter_config.json       # LoRA weight configuration file
  - adapter_model.bin         # LoRA weight file
  - special_tokens_map.json   # special_tokens_map file
  - tokenizer_config.json     # tokenizer configuration file
  - tokenizer.model           # tokenizer file
```

The following is the size of each original model and 4-bit quantization. When converting the corresponding model, make sure that the machine has enough memory and disk space (minimum requirements):

|                    |   7B   |   13B   |   33B   |   65B   |
| :----------------- | :----: | :-----: | :-----: | :-----: |
| Original（FP16）   | 13 GB  |  24 GB  |  60 GB  | 120 GB  |
| Quantized (8-bit)  | 7.8 GB | 14.9 GB |    -    |    -    |
| Quantized（4-bit） | 3.9 GB | 7.8 GB  | 19.5 GB | 38.5 GB |

## Model Reconstruction

In order to merge the LoRA model with the original LLaMA for further tuning or inference, two methods are currently provided:

| Method                | Usage                                                        |                           Tutorial                           |
| :-------------------- | :----------------------------------------------------------- | :----------------------------------------------------------: |
| **Online conversion** | Suitable for Google Colab users, can use notebook for online conversion and model quantization. | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Online-conversion-with-Colab) |
| **Manual conversion** | Suitable for offline conversion, generates models in different formats for quantization or further fine-tuning. | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Manual-Conversion) |

Related documentation has been moved to the project's >>> [📚GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Model-Reconstruction).

## Quick Deployment

We mainly provide the following three ways for inference and local deployment.

| Method                                                       | Features                                                     | Platform | CPU  | GPU  | Quantization |  UI  |                           Tutorial                           |
| :----------------------------------------------------------- | ------------------------------------------------------------ | :------: | :--: | :--: | :----------: | :--: | :----------------------------------------------------------: |
| [**llama.cpp**](https://github.com/ggerganov/llama.cp)       | a tool for quantizing model and deploying on local CPU       | General  |  ✅   |  ✅   |      ✅       |  ❌   | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/llama.cpp-Deployment) |
| [**🤗Transformers**](https://github.com/huggingface/transformers) | original transformers inference method, support CPU/GPU      | General  |  ✅   |  ✅   |      ✅       |  ✅   | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Inference-with-Transformers) |
| [**text-generation-webui**](https://github.com/oobabooga/text-generation-webui) | a tool for deploying model as a web UI                       | General  |  ✅   |  ✅   |      ✅       |  ✅   | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/text-generation-webui) |
| [**LlamaChat**](https://github.com/alexrozanski/LlamaChat)   | a macOS app that allows you to chat with LLaMA, Alpaca, etc. |  MacOS   |  ✅   |  ❌   |      ✅       |  ✅   | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Using-LlamaChat-Interface) |
| [**LangChain**](https://github.com/hwchase17/langchain)      | LLM application development framework, suitable for secondary development | General | ✅<sup>†</sup> |  ✅   | ✅<sup>†</sup> |    ❌     | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Integrated-with-LangChain) |
| [**privateGPT**](https://github.com/imartinez/privateGPT) | LangChain-based multi-document QA framework | General | ✅ | ✅ | ✅ | ❌ | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Use-privateGPT-for-multi-document-QA) |

<sup>†</sup>: Supported by LangChain, but not implemented in the tutorial. Please refer to the official LangChain Documentation for details.

Related documentation has been moved to the project's >>> [📚GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Model-Inference-and-Deployment).


## System Performance

In order to quickly evaluate the actual performance of related models, this project compared the effects of Chinese Alpaca-7B, Alpaca-13B, Alpaca-Plus-7B and Alpaca-Plus-13B on some common tasks given the same prompt. Reply generation is random and is affected by factors such as decoding hyperparameters and random seeds. The following related evaluations are not absolutely rigorous, and the test results are for reference only. Welcome to experience it yourself. For detailed evaluation results, please see [examples](./examples).

| 测试任务         | Samples | Alpaca-13B | Alpaca-Plus-7B | Alpaca-Plus-13B |
| ---------------- | :----: | :--------: | :------------: | :-------------: |
| **💯Overall**    |  200   |    74.3    |      78.2      |   **👍🏻80.8**    |
|  Question Answering         |   20   |     70     |       74       |    **👍🏻79**     |
| Open QA       |   20   |     77     |       77       |       77        |
| Computation, Reasoning  |   20   |     61     |       61       |       60        |
| Poetry, Literature, Philosophy |   20   |     65     |    **👍🏻76**    |    **👍🏻76**     |
| Music, Sports, Entertainment |   20   |     68     |       73       |    **👍🏻80**     |
| Letters and Articles     |   20   |     83     |       82       |    **👍🏻87**     |
| Translation         |   20   |     84     |       87       |    **👍🏻90**     |
| Multi-turn Dialogue         |   20   |     88     |       89       |       89        |
| Coding         |   20   |     65     |       64       |    **👍🏻70**     |
| Ethics       |   20   |     82     |    **👍🏻99**    |    **👍🏻100**    |


## Training Details

The entire training process includes three parts: vocabulary expansion, pre-training, and instruction fine-tuning. Please refer to [merge_tokenizers.py](scripts/merge_tokenizers.py) for vocabulary expansion; refer to [run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) in 🤗transformers and the relevant parts of dataset processing in the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) project for pre-training and self-instruct fine-tuning.

We have open-sourced the scripts for pre-training and instruction finetuning (SFT): 

- Pre-training: [scripts/run_clm_pt_with_peft.py](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/run_clm_pt_with_peft.py), refer to [Pre-training Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Pretraining-Script)

- Instruction Finetuning: [scripts/run_clm_sft_with_peft.py](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/run_clm_sft_with_peft.py), refer to [SFT Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/SFT-Script)

Please refer to our  >>> [📚GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Training-Details).


## FAQ

FAQ provides answers to frequent questions. Please see our FAQ before submitting an issue.

```
Q1: Why can't you release the complete model weights?
Q2: Will there be versions of 33B, and 65B in the future?
Q3: The model doesn't perform well on some tasks!
Q4: Why expand the vocabulary? Can't you just pre-train the original LLaMA with Chinese data?
Q5: The reply is very short
Q6: Under Windows, the model cannot understand Chinese, the generation speed is very slow, etc.
Q7: Chinese-LLaMA 13B model cannot be launched with llama.cpp, reporting inconsistent dimensions.
Q8: Chinese-Alpaca-Plus does not show better performance than the others.
Q9: The model does not perform well on NLU tasks, such as text classification.
```

Please refer to our  >>> [📚GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/FAQ).

## Limitations

Although the models in this project have significantly improved Chinese understanding and generation capabilities compared to the original LLaMA and Alpaca, there are also the following limitations:

- It may produce unpredictable harmful content and content that does not conform to human preferences and values.
- Due to computing power and data issues, the training of the related models is not sufficient, and the Chinese understanding ability needs to be further improved.
- There is no online interactive demo available for now (Note: users can still deploy it locally themselves).

## Citation

If you find the model, data, code in our project useful, please consider citing our work as follows: https://arxiv.org/abs/2304.08177

```
@article{chinese-llama-alpaca,
      title={Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca}, 
      author={Cui, Yiming and Yang, Ziqing and Yao, Xin},
      journal={arXiv preprint arXiv:2304.08177},
      url={https://arxiv.org/abs/2304.08177},
      year={2023}
}
```

## Acknowledgements

This project is based on the following open-source projects for secondary development, and we would like to express our gratitude to the related projects and research and development personnel.

|                   Foundation Models, Codes                   |             Quantization, Inference, Deployment              |                             Data                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [LLaMA by Facebook](https://github.com/facebookresearch/llama)<br/>[Alpaca by Stanford](https://github.com/tatsu-lab/stanford_alpaca)<br/>[alpaca-lora by @tloen](https://github.com/tloen/alpaca-lora) | [llama.cpp by @ggerganov](https://github.com/ggerganov/llama.cpp)<br/>[LlamaChat by @alexrozanski]( https://github.com/alexrozanski/LlamaChat)<br/>[text-generation-webui by @oobabooga](https://github.com/oobabooga/text-generation-webui) | [pCLUE and translation data by @brightmart](https://github.com/brightmart/nlp_chinese_corpus) |

Episode: The current logo is automatically generated by GPT-4 with the DALL·E plugin (previously generated by midjourney).

## Disclaimer

**The resources related to this project are for academic research purposes only and are strictly prohibited for commercial use.** When using parts involving third-party code, please strictly follow the corresponding open-source agreements. The content generated by the model is affected by factors such as model calculation, randomness, and quantization accuracy loss. This project cannot guarantee its accuracy. For any content output by the model, this project does not assume any legal responsibility and does not assume responsibility for any losses that may result from the use of related resources and output results.

This project is initiated and maintained by individuals and collaborators in their spare time, so we cannot guarantee a timely response to resolving relevant issues.

## Feedback

If you have any questions, please submit them in GitHub Issues.

- Before submitting a question, please check if the FAQ can solve the problem and consult past issues to see if they can help.
- Please use our dedicated issue template for submitting.
- Duplicate and unrelated issues will be handled by [stable-bot](https://github.com/marketplace/stale); please understand.
- Raise questions politely and help build a harmonious discussion community.
