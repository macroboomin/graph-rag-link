GraphRAG with Link Prediction
=============
## Overview
기존에 OpenAI 세팅에 최적화되어 있는 GraphRAG를 Phi-3-mini-4k-instruct과 같은 작은 모델로 돌릴 수 있는 방안을 고안한 이후 link prediction을 이용하여 confidence를 도출하여 overconfident 문제 완화
+ model: [https://huggingface.co/microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

## File Overview
paper와 paper_base는 open-ended 문제에 대하여 Workflow를 보여주는데에 사용 
- [https://arxiv.org/abs/2406.08391](Large Language Models Must Be Taught to Know What They Don't Know) 논문 데이터셋 사용
bio와 bio_base는 MMLU 데이터셋으로 여러 지표들을 평가하는데에 사용
- [https://huggingface.co/datasets/enelpol/rag-mini-bioasq](rag-mini-bioasql) 데이터셋을 retrieve에 사용, MMLU 데이터셋을 평가에 사용
- MMLU 데이터셋은 bio 데이터와 관련이 있는 항목들만을 사용 (college_biology, high_school_biology, medical_genetics, professional_medicine ,virology)

### paper_base, bio_base
- `*_retrieval_base.py`: 단순히 chunk로 분할하여 벡터 형태로 FAISS로 저장
- `*_base.py`: 관련된 context를 retrieve하고, Phi3 모델이 answer와 verbalized confidence를 도출
- `rag_base_metrics.py`: ECE, NLL, AUROC와 같은 metrics 도출
- `visual_base.py`: Reliability Diagram 도출

### paper, bio
- `document_split.py`: 주어진 데이터셋을 chunk로 분할
- `graph.py`: chunk로 분할한 데이터셋에서 node와 edge 관계를 도출하여 knowledge graph로 만들음
- `link_prediction.py`: 만들어진 knowledge 그래프로 link prediction을 위한 DistMult 모델 학습
- `graph_rag.py`: 그래프로부터 관련된 subgraph를 retrieve하고, LLM이 생성한 output으로 다시 그래프를 만든 후 link prediction의 평균값으로 confidence 도출
- `rag_metrics.py`: ECE, NLL, AUROC와 같은 metrics 도출
- `visual.py`: Reliability Diagram 도출

## WorkFlow
- File Overview에 적혀 있는 순서대로 진행
- 전반적인 개요는 다음과 같음
<img width="833" alt="image" src="<img width="833" alt="image" src="https://github.com/user-attachments/assets/0089fdd5-cc39-42db-b3ef-37e195e5ea87">
