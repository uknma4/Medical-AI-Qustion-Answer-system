import streamlit as st

from es_medical_query_utils  import query_es_content  # 使用 query_es_content 函数
from es_medical_query_utils import query_ollama  # 使用 query_es_content 函数
st.title("医疗问答系统")

# 用户输入问题
query = st.text_input("请输入您的问题：")

if  st.button("查询"):
    # 使用 query_es_content 函数进行查询
    search_results = query_es_content(
        index_name="medical_cases",  # Elasticsearch 索引名称
        content_question=query,   #用户输入的问题 ,
        top_k=10,  # 获取最多 10 条相关结果
        min_score=120  # 最低评分阈值
    )

    if search_results:

        # 构建上下文
        context = "\n".join([
            f"[文档{idx}] {doc.get('Question', '')}\n{doc.get('Response', '')}"  # \n{doc.get('Complex_CoT', '')}
            for idx, doc in enumerate(search_results, 1)
        ])[:8000]  # 简单截断
    else:
        context = "未找到相关医疗信息。"



    # 3. 执行Ollama查询
    if context:

        print("\n生成回答中...")

        print(f"context...{context}")
        print(f"query...{query}")
        result = query_ollama(
            # model="deepseek-8b:latest",
            model="deepseek-r1:7b",
            prompt=query,
            context=context
        )

        # 解析结果
        answer = result.get("message", {}).get("content", "未获得有效回答")
        # 在界面显示回答
        st.write(answer)





