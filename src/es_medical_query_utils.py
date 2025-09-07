import requests
import time
import numpy as np  # 新增numpy用于处理向量
import requests
import json
import os
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, helpers
import logging
from typing import List, Dict, Any, Optional, Tuple
from huggingface_hub import snapshot_download
import logging
import threading
from text2vec import SentenceModel
content_question = "一个患有急性阑尾炎的病人已经发病快一周，腹痛稍有减轻但仍然发热，检查发现右下腹有压痛的包块，该如何处理？"
# 配置日志记录
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 初始化模型（缓存提升性能）
MODEL_CACHE = None


def load_model(model_name: str = "/home/eddie/models/textmodel"):
    """加载并缓存嵌入模型"""
    global MODEL_CACHE
    if MODEL_CACHE is None:
        logger.info(f"Loading sentence transformer model: {model_name}")
        MODEL_CACHE = SentenceModel(model_name)
    return MODEL_CACHE


# Elasticsearch 客户端配置
es = Elasticsearch(
    hosts=["http://localhost:9200"],
    retry_on_timeout=True,
    max_retries=3,
    request_timeout=30
)


def generate_query_vector(text: str) -> List[float]:
    """生成查询向量（带异常处理）"""
    try:
        model = load_model()
        return model.encode(text, normalize_embeddings=True).tolist()
    except Exception as e:
        logger.error(f"向量生成失败: {str(e)}")
        return []


def optimize_search_query(
        query: str,
        query_vector: List[float],
        top_k: int = 50,
        min_score: float = 0.5
) -> Tuple[Dict, Dict]:
    """
    构建优化的复合查询结构
    返回 (查询参数, 评分分析参数)
    """
    # 权重配置（可调整）
    field_weights = {
        "Question": 1.0
         # "Response": 1.3
         # "Complex_CoT":0.2
    }

    vector_config = {

        "Question_vector": 0.8
          # "Response_vector": 0.5

    }

    # 构建组合查询
    search_query = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    # 传统文本匹配（加权）
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [f"{field}^{weight}" for field, weight in field_weights.items()],
                            "type": "best_fields",
                            "tie_breaker": 0.3
                        }
                    },
                    # 向量搜索：内容向量
                    {
                        "knn": {
                            "field": "Question_vector",
                            "query_vector": query_vector,
                            "num_candidates": 100,
                            "boost": vector_config["Question_vector"],
                            "similarity": 30
                        }
                    }
                    # # 向量搜索：章节标题向量
                    # {
                    #     "knn": {
                    #         "field": "Response_vector",
                    #         "query_vector": query_vector,
                    #         "num_candidates": 100,
                    #         "boost": vector_config["Response_vector"]
                    #     }
                    # }
                ],
                "minimum_should_match": 1,
                "boost": 1.0
            }
        },
        "_source": ["Question", "Response", "Complex_CoT"],
        "min_score": min_score,
        "explain": True  # 开启评分解释
    }

    # 评分分析参数
    score_analysis = {
        "field_weights": field_weights,
        "vector_config": vector_config
    }

    return search_query, score_analysis


def query_es_content(
        index_name: str,
        content_question: str,
        top_k: int = 45,
        min_score: float = 70
) -> List[Dict[str, Any]]:
    """
    执行 Elasticsearch 查询，返回结果
    """
    query_vector = generate_query_vector(content_question)
    if not query_vector:
        return []

    search_query, score_params = optimize_search_query(
        query=content_question,
        query_vector=query_vector,
        top_k=top_k,
        min_score=min_score
    )

    try:
        # 执行搜索（带性能监控）
        logger.info(f"Executing search on index: {index_name}")
        response = es.search(
            index=index_name,
            body=search_query,
            request_timeout=45
        )
        logger.debug(f"Search completed in {response['took']}ms")
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}")
        return []

    # 结果处理和分析
    return process_search_results(response, min_score, score_params)


def process_search_results(
        response: Dict,
        min_score: float,
        score_params: Dict
) -> List[Dict[str, Any]]:
    """处理和分析搜索结果"""
    hits = response.get("hits", {}).get("hits", [])

    # 评分分析
    scores = [hit["_score"] for hit in hits]
    logger.info(f"""
    评分分析：
    - 平均分: {np.mean(scores):.2f}
    - 最高分: {np.max(scores):.2f}
    - 最低分: {np.min(scores):.2f}
    - 有效结果: {len([s for s in scores if s >= min_score])}/{len(scores)}
    """)

    # 构建结果集
    results = []
    for hit in hits:
        if hit["_score"] < min_score:
            continue

        source = hit.get("_source", {})
        results.append({
            "id": hit["_id"],
            "score": hit["_score"],
            "Question": source.get("Question", ""),
            "Response": source.get("Response", ""),
            "Complex_CoT": source.get("Complex_CoT", "")[:5000],
            "explanation": hit.get("_explanation")  # 包含评分解释
        })

    return results





def query_ollama(model, prompt, context):
    """简化版 Ollama 查询函数"""
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": f"你是一个医疗专家，请基于以下内容回答：\n{context}"
            },
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()
    # 2. 构建上下文

#
# if __name__ == "__main__":
#     # 配置参数
#     SEARCH_CONFIG = {
#         "index_name": "medical_cases",
#         "content_question": content_question,
#         "top_k": 45,
#         "min_score": 70 # 根据实际数据调整
#     }
#
#     # 执行搜索
#     documents = query_es_content(**SEARCH_CONFIG)
#
#     # 结果展示
#     print(f"\n检索到 {len(documents)} 条有效文档（评分 > {SEARCH_CONFIG['min_score']}）")
#     for idx, doc in enumerate(documents, 1):
#         print(f"\n文档 {idx} [评分: {doc['score']:.2f}]:")
#         print(f"问题: {doc['Question']}")
#         print(f"回答: {doc['Response']}")
#         # print(f"复杂回答: {doc['Complex_CoT'][:500]}...")
#         print("-" * 50)
#
#     # 构建上下文
#     context = "\n".join([
#         f"[文档{idx}] {doc.get('Question', '')}\n{doc.get('Response', '')}"  # \n{doc.get('Complex_CoT', '')}
#         for idx, doc in enumerate(documents, 1)
#     ])[:5000]  # 简单截断
#
#     # 执行Ollama查询
#     if context:
#         question = content_question
#         print("\n生成回答中...")
#         start_time = time.time()
#
#         result = query_ollama(
#             # model="deepseek-r1:7b",
#             model="deepseek-r1:7b",
#             prompt=question,
#             context=context
#         )
#
#         # 解析结果
#         answer = result.get("message", {}).get("content", "未获得有效回答")
#         print(f"\n回答生成耗时: {time.time() - start_time:.1f}秒")
#         print("\n=== 最终回答 ===")
#         print(answer.replace(". ", ".\n"))
#     else:
#         print("未检索到相关上下文")