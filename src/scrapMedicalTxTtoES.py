import re
import json
import os
from datetime import datetime
from pathlib import Path
import requests
import time
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, helpers
import logging
from transformers import set_seed, AutoTokenizer, AutoModel

# ---------- 配置路径 ----------
input_path = Path("/home/eddie/script/medical_ask_answer/medical_o1_sft_Chinese.json")
output_path = Path("/home/eddie/script/medical_ask_answer/formatted_complex_cot_to_bot.jsonl")
medical_json = []  # 初始化为空列表

# 加载中文向量模型
sentence_model = SentenceTransformer("shibing624/text2vec-base-chinese")

# 连接到 Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# 创建索引（如果不存在）
index_name = 'medical_cases'
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)

# ---------- 读取原始数据 ----------
with input_path.open("r", encoding="utf-8") as f:
    data = json.load(f)  # 假设文件顶层是一个 list，内部每项形如 {"Question": ..., "Complex_CoT": ..., "Response": ...}
n = 0

# ---------- 转换并写入 JSONL ----------
with output_path.open("w", encoding="utf-8") as fout:
    actions = []  # 用于存储批量上传的动作
    for record in data:
        n += 1
        question = record.get("Question", "").strip()
        cot = record.get("Complex_CoT", "").strip()
        response = record.get("Response", "").strip()

        # 向量化字段
        question_vector = sentence_model.encode(question).tolist() if question else []
        cot_vector = sentence_model.encode(cot).tolist() if cot else []
        response_vector = sentence_model.encode(response).tolist() if response else []

        # 构造病例数据
        case_data = {
            "case_id": f"case{n}",  # 每个病例的唯一标识符
            "Question": question,  # 病例的提问
            "Question_vector": question_vector,  # 提问的向量
            "Complex_CoT": cot,  # 病例的复杂推理
            "Complex_CoT_vector": cot_vector,  # 推理的向量
            "Response": response,  # 病例的回复
            "Response_vector": response_vector,  # 回复的向量
            "metadata": {
                "created_by": "uknma4",
                "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "context": "医学诊断",
                "language": "zh"
            }
        }
        medical_json.append(case_data)  # 将该对象添加到列表中

        # 写入 JSON 对象到文件
        fout.write(json.dumps(case_data, ensure_ascii=False) + "\n")

        # 构造 Elasticsearch 动作
        actions.append({
            "_index": index_name,
            "_id": case_data["case_id"],  # 使用 case_id 作为文档 ID
            "_source": case_data
        })

# ---------- 执行上传 ----------
if actions:
    try:
        helpers.bulk(es, actions)  # 批量上传到 Elasticsearch
        print(f"✅ 已成功上传 {len(actions)} 条病例数据到 Elasticsearch 索引 '{index_name}'")
    except Exception as e:
        print(f"❌ 上传到 Elasticsearch 失败: {e}")
else:
    print("⚠️ 没有数据可上传")

print(f"✅ 已生成 {output_path}，共包含 {len(data)} 条样本。")