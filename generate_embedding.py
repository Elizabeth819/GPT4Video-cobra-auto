from openai import AzureOpenAI
import json

import openai

# 设置API密钥和终结点
api_key = "b6e3648236cb44848e71b2c6d49ef3a2"
api_endpoint = "https://embedding-3-large.openai.azure.com/"

client = AzureOpenAI(api_key=api_key,
                     api_version="2024-04-01-preview",
                     azure_endpoint=api_endpoint)

def generate_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"  # 使用embedding-3-large模型
    )
    return response.data[0].embedding

# 读取guitantou.json文件
with open('actionSummary-guitantou.json', 'r', encoding='utf-8') as file:
    documents = json.load(file)

# 生成向量并添加到文档
for doc in documents:
    summary_vector = generate_embedding(doc['summary'])
    action_vector = generate_embedding(doc['actions'])
    character_vector = generate_embedding(doc['characters'])

    doc['summaryVector'] = summary_vector
    doc['actionVector'] = action_vector
    doc['characterVector'] = character_vector

# 保存更新后的文档
with open('actionSummary-guitantou_with_vectors.json', 'w', encoding='utf-8') as file:
    json.dump(documents, file, ensure_ascii=False, indent=4)
