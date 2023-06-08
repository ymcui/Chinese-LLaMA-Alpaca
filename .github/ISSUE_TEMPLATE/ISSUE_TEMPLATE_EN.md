---
name: English Issue Template
about: Issues related to this project. We will first look into the issues that provide sufficient details.
---

### Describe the issue in detail

*Please describe the problem you encountered as specifically as possible. If possible, please copy-and-paste your running command. This will help us locate the issue more quickly.*

### Referential Information

#### Dependencies (code-related issues)

*Please provide transformers, peft, torch, etc. versions.*

#### Log or Screenshot

*Please provide a text log or screenshot to help us better understand the issue details.*

### Checklist

*Fill in the [ ] with an x to mark it as checked. Delete any option that is not related to this issue.*

- [ ] **Base model**: LLaMA  / Alpaca / LLaMA-Plus / Alpaca-Plus (and also model size: 7B/13B/33B)
- [ ] **Operating System**: Windows / MacOS / Linux
- [ ] **Issue type**: Download / Model conversion and merging / Pretraining and SFT / Inference (🤗 transformers) / Quantization and deployment (llama.cpp, text-generation-webui, LlamaChat) / Performance / Others
- [ ] **Validity Check**: You MUST check [SHA256.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/SHA256.md) for your model. We cannot guarantee the unexpected issues caused by using invalid models.
- [ ] Due to frequent dependency updates, please ensure you have followed the steps in our [Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki)
- [ ] I have read the [FAQ section](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/FAQ) AND searched for similar issues and did not find a similar problem or solution
- [ ] Third-party plugin issues: e.g., [llama.cpp](https://github.com/ggerganov/llama.cpp), [text-generation-webui](https://github.com/oobabooga/text-generation-webui), [LlamaChat](https://github.com/alexrozanski/LlamaChat), we recommend checking the corresponding project for solutions
