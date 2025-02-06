# 使用 ChatGPT 聊天到底需要多少费用？

OpenAI 在 6 月 13 日进行了大规模降价和升级，text-embedding-ada-002 降价 95%，GPT-3.5-turbo 降价 25%，GPT-4 最高可支持 32K 文本。本文将详细解析 ChatGPT 的定价机制，帮助用户更好地理解使用成本。

## OpenAI 定价说明

在 [OpenAI 的官网](https://openai.com/pricing) 上，对 ChatGPT 的价格计算方式有详细的说明。价格以 token 为单位计算，下面我们将具体分析实际费用。

### 官网聊天

官方网页上的聊天是免费的，收费主要针对 API 用户。API 用户默认赠送 5 美元额度。

### 大语言模型

#### 计算方法

**测试样本：**  
`Multiple models, each with different capabilities and price points. Prices are per 1,000 tokens. You can think of tokens as pieces of words, where 1,000 tokens is about 750 words. This paragraph is 35 tokens.`  
官方提供的样本文字说明，这段测试样本的长度为 35 个 tokens。

ChatGPT 使用的 tokenizer 是 Byte-Pair Encoding (BPE) 算法，以下是几种近似计算 token 值的方法：

1. **Transformers**  
2. **OpenAI tiktoken**  
3. **单词数 / 0.75**

**Transformers 方法示例代码：**

python
def get_transformers_tokens(text: str, encoding_name: str) -> int:
    tokenizer = transformers.AutoTokenizer.from_pretrained(encoding_name)
    tokens = tokenizer.tokenize(text)
    num_tokens = len(tokens)
    return num_tokens


使用此方法计算，得到的 tokens 值为 45。

**tiktoken 方法示例代码：**

python
def get_tiktoken_tokens(text: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


使用此方法计算，得到的 tokens 值也为 45。

有趣的是，虽然 OpenAI 官方网页中给出的 tokens 值为 35，但使用 [OpenAI 官方计算器](https://platform.openai.com/tokenizer) 计算的结果也是 45。而另一个工具 [Article Fiesta Token Calculator](https://articlefiesta.com/seo-tools/token-calculator) 则得出的 tokens 值为 35。

#### Tokens 消耗影响因素

Tokens 的消耗主要受以下因素影响：

1. **System Prompt**：聊天的背景信息，通常被软件包装，用户看不到。
2. **User Prompt**：用户的提问。
3. **AI's Reply**：ChatGPT 的回答。

#### 实际费用计算示例

以以下样本为例，计算具体费用：

plaintext
How to calculate tokens for chatGPT?
To calculate the number of tokens for a text sequence in ChatGPT, you can follow these steps:

    Tokenization: In ChatGPT, the text needs to be segmented into a sequence of words or subword units, called tokens. You can use a tokenizer to convert the text into tokens. The tokenizer used in ChatGPT is based on Byte-Pair Encoding (BPE) algorithm, which can convert the text into a decodable sequence of subword units.

    Counting tokens: Once the text has been segmented into tokens, you can count the number of tokens. In ChatGPT, the number of tokens is equal to the number of tokens contained in the text. You can use the len() function to count the number of tokens.


### ChatGPT 定价表

| **模型**       | **输入**            | **输出**            |
|----------------|---------------------|---------------------|
| 4K context     | $0.0015 / 1K tokens | $0.002 / 1K tokens  |
| 16K context    | $0.003 / 1K tokens  | $0.004 / 1K tokens  |

根据上述样本和定价，5 美元对应的对话次数约为 6887 次。也就是说，账号自带的 5 美元额度，大约可以进行 6000 多句英文对话（中文对话会有所不同）。

### GPT-4 定价表

| **模型**       | **输入**            | **输出**            |
|----------------|---------------------|---------------------|
| 8K context     | $0.03 / 1K tokens   | $0.06 / 1K tokens   |
| 32K context    | $0.06 / 1K tokens   | $0.12 / 1K tokens   |

### InstructGPT 定价表

| **模型**  | **价格**            |
|-----------|---------------------|
| Ada       | $0.0004 / 1K tokens |
| Babbage   | $0.0005 / 1K tokens |
| Curie     | $0.0020 / 1K tokens |
| Davinci   | $0.0200 / 1K tokens |

### 微调模型定价表

| **模型**  | **训练**            | **使用**            |
|-----------|---------------------|---------------------|
| Ada       | $0.0004 / 1K tokens | $0.0016 / 1K tokens |
| Babbage   | $0.0006 / 1K tokens | $0.0024 / 1K tokens |
| Curie     | $0.0030 / 1K tokens | $0.0120 / 1K tokens |
| Davinci   | $0.0300 / 1K tokens | $0.1200 / 1K tokens |

### 嵌入模型定价表

| **模型**       | **价格**            |
|----------------|---------------------|
| Ada v2         | $0.0001 / 1K tokens |
| Ada v1         | $0.0040 / 1K tokens |
| Babbage v1     | $0.0050 / 1K tokens |
| Curie v1       | $0.0200 / 1K tokens |
| Davinci v1     | $0.2000 / 1K tokens |

### 其他模型

#### 图像模型

| **分辨率**   | **价格**         |
|--------------|------------------|
| 1024×1024    | $0.020 / image   |
| 512×512      | $0.018 / image   |
| 256×256      | $0.016 / image   |

#### 音频模型

| **模型**  | **价格**             |
|-----------|----------------------|
| Whisper   | $0.006 / 分钟（四舍五入到秒） |

## 总结

虽然 OpenAI 提供了 token 计算工具，但工具的计算值与官方网页中的说明存在差异。  
使用 API 访问 ChatGPT 3.5 的话，账号自带的 5 美元额度，大约可以进行 6000 多句对话。  
想深入了解 token 和分词机制，可以进一步学习相关技术文档。

👉 [WildCard | 一分钟注册，轻松订阅海外线上服务](https://bbtdd.com/WildCard)