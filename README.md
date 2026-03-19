# Build_RAG




- 지난 포스팅에서 RAG가 어떤 단계로 구축되는 지, 어떤 매커니즘을 보이는 지 정리했다.
| https://velog.io/@yjhut/Mechanism-of-RAG

- 이번 포스팅에서는 랭체인(langchain)으로 RAG를 직접 구축, 성능까지 체크하고자 한다.

> 💡 `Datasets`
>
> - `RAG Dataset`: 의료 특화 데이터
| 출처 : AI hub
>
>
> - `Test Dataset`: Medical QA
| 출처 : https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset

<br>

> 💡 `Model`
>- `Sentence Embedding`: `jinaai/jina-embeddings-v2-base-en`
>
>
> - `ReRank Model` : `tomaarsen/reranker-ModernBERT-base-gooaq-bce`
>
>
> - `LLM`: `google/gemma-2b-2-it`

<br>

> 💡 `평가 방법`
- 실제 정답과 LLM 응답의 `cosine similarity` 산출, 평균 값을 평가지표로 사용 
>
![](https://velog.velcdn.com/images/yjhut/post/ce219006-8749-4001-b032-b50611cf8a12/image.png)
>
> - RAG에 사용한 여러 방법은 아래에서 자세하게 설명!

- 필자는 프레임워크로 `langchain`을 사용하였고, 벡터 DB는 `Chroma`를 사용

---
 
<br>
<br>

## RAG 구축 과정

- RAG를 구축할 데이터가 확보되었다면 먼저 필요 라이브러리 설치 후 오류 없는지 확인!

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain_chroma import Chroma
```

<br>
<br>

### Chunking

- 청킹은 앞서 RAG 포스팅에서도 다뤘듯, RAG 사용의 성능을 확보하는 가장 중요한 요소이며,

- 실제로 어떤 청킹 방식에 따라 RAG 적용의 성공 여부가 결정된다고 해도 과언이 아니다.
| 출처: https://www.aitimes.com/news/articleView.html?idxno=206222

- `SemanticChunker`를 사용하면 Documents 간 임베딩 유사도를 기반으로 청킹할 수 있는데,

- 이 때 Breakpoint. 즉, 청크 분할 기준으로 사용되는 옵션은 네 가지로 제공된다.
   - `percentile`
      - 지정한 퍼센티지 값(N)을 기준으로 청크 분리(정확히는 상위 (100-N)% 값을 기준)
      
   - `standard deviation`
      - 유사도 평균에서 부여한 표준 편차만큼 멀어진 곳을 청크 분리
      - $mean+n*std$에서 n을 지정
      
   - `interquartile`
      - 유사도 거리값의 중앙 50% 범위를 기준
      
   - `gradient`
      - 유사도 변화의 기울기가 급격해지는 지점을 탐색하여 청크 분리


- 위 방법 중 `standard deviation`을 사용하였으며, n은 0.8, 1.5, 3.0 중 가장 좋은 성능을 보인 `0.8`을 사용하였다.
   - n < 1을 사용한 이유는 과다한 내용까지 함께 묶이지 않기 위해. 즉, 청크 세분화를 의도하였다.

> 💡 `Chunking Result`
- 총 35,709개 데이터 -> 278,496개 청크 생성

---
 
<br>
<br>


### Retrieval

- RAG를 사용하기 위해선 query와 가장 유사한 k개 문서를 LLM에 같이 입력해야 한다.

- 처음 시도는 단순 코사인 유사도를 기반으로 탐색했으나, 오히려 RAG를 사용할 때 성능이 떨어지는 문제 발생

- 일종의 필터링 작업을 거치기 위해 아래 세 방법을 적용하였다.
   - `BM25_Search`
   - `MMR_Search`
   - `Re-ranking`


> 💡 `BM25_Search`
> - `키워드 매칭` 방법으로, 임베딩이 아닌 중복된 키워드를 기반으로 유사한 k개 문서를 찾는다.
> - 의료 용어 특성상 특정 용어를 기반으로 찾기에 적합 
>
> 💡 `MMR_Search`
> - 유사도를 적용한 방법의 심화된 버전으로 MMR은 `Max_Marginal_Relevance_search`를 의미한다.
> - Score = [ $\lambda \times$ 쿼리와의 유사도 ] - [ $(1 - \lambda) \times$ 이미 뽑힌 문서와의 유사도 ]
> - $\lambda$ -> 0~1, 1에 가까울수록 유사도를 중점으로 보고, 0에 가까울 수록 다양성을 중점으로 한다.
   - `다양성`의 의미는 이미 뽑힌 문서와 유사한 내용을 가진 문서가 추가로 뽑히지 않도록!
   
- 위 두 탐색 방법을 동시에 이용하여 context에 추가할 후보를 선정
    - ex) k = 5인 경우, context 후보는 10개


> 💡 `Re-ranking`
- 리랭킹이란, 단어 그대로 순서를 다시 매기는 것
- context 후보를 대상으로 수행
- 이 과정에서 `CrossEncoder` 개념이 사용된다.
   - 문장 유사도를 측정하는 방법
   - 모델 내부에서 문장 간 직접 어텐션 연산을 수행하여, 유사도 측정에 뛰어난 성능을 보인다.
   - ![](https://velog.velcdn.com/images/yjhut/post/83a4af70-014a-4812-bf37-cbf57a6038d7/image.png)
   

- `BM25, MMR 탐색 방법으로 구성한 context에 리랭킹을 다시 적용하여 이 중 유사도가 0.8을 넘는 문서만 최종 사용할 context에 추가`하였다.
 
---
 
<br>
<br>

### Test Prompt

- Vanila와 RAG각 LLM에 사용한 프롬프트는 아래와 같이 구성하였다. 

- `Vanila LLM`
```python
vanila_prompt = f"""
    <bos>[Role]: You are a highly skilled medical professional.
    Answer the user's question question.

    <start_of_turn>user
    {query}<end_of_turn>
    <start_of_turn>model

      """
```

<br>

- `RAG LLM`
```python
RAG_prompt = f"""
     <bos>[Role]: You are a highly skilled medical professional. 
     [Instruction]: Answer the user's question by prioritizing the provided [context]. 
     If the [context] does not contain the answer, use your own extensive medical knowledge to provide a highly accurate response.

     [context]: {context}

    <start_of_turn>user
    {query}<end_of_turn>
    <start_of_turn>model
    
      """
```

---

<br>
<br>

### Result

- Test 데이터는 symtom, information, treatment, causes에 대해 묻는 질문 300개씩, 총 1200개를 사용하였다.

- 최종 결과는 아래 표와 같으며, 기본 모델 대비 `1.41` 향상된 성능을 보였다.


|Model|Performance|
|:---:|:---:|
|Vanila_LLM|84.438|
|RAG_LLM|**85.637**|


---

<br>
<br>

## 정리


> - RAG를 구축하면서 Breakpoint를 지정하는 데 시간이 많이 소요되었는데, 이는 데이터마다 다른 특성을 가져, 방법과 값의 기준이 없는 게 어찌보면 당연하다.
>
>
> - 나의 경우, 표준편차 방법에 n = 0.8 을 사용했지만 다른 방법 또는 다른 값이 더 좋은 성능을 낼 수 있다는 게 제일 모호하고 답답한 포인트였다. 
>
>
> - 물론 Retrieval 단계에서 다양한 서칭 방법 및 리랭킹 등 필터링 방법이 존재하지만, 청킹 과정에서 쓰레기가 만들어지면 LLM은 'garbage in, garbage out'이 발생할 수 밖에 없다.
>
>
> - 이에 따라 청킹 시 `안전 장치`의 중요성을 느꼈고, 추후 이 방법에 대해 탐색하고자 한다.
