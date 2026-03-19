

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

from transformers import AutoTokenizer, AutoModelForCausalLM

from collections import defaultdict

import pandas as pd 
import numpy as np

import random
import time
import contractions

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from zipfile import ZipFile

import warnings
warnings.filterwarnings('ignore')

from huggingface_hub import login
login() # key 입력

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

import torch._dynamo
torch._dynamo.config.suppress_errors = True

set_seed(43)


########################################################################


# 텍스트파일 압축 풀기
extract_path = 'TEXT'

def extracted_file(zip_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)

    with ZipFile(zip_path, 'r') as zip_ref:
        infos = zip_ref.infolist()

        print(f"총 {len(infos)}개의 항목, 압축 해제 시작...")

        for info in tqdm(infos, desc='Extracting'):
            original_filename = info.filename
            try:
                info.filename = original_filename.encode('cp437').decode('cp949')

            except (UnicodeEncodeError, UnicodeDecodeError):
                pass

            zip_ref.extract(info, extract_path)
            
    print(f"{extract_path}에 압축 해체 완료!!")
    print()

for path in data_path_list:
    extracted_file(path, extract_path)


########################################################################


# 하나의 파일이 여러 파일로 분할되어, 이를 병합
directory_path = 'TEXT/'
def get_file_groups(directory_path:str):
    file_groups = defaultdict(list)

    all_file_list = os.listdir(directory_path)

    for file_name in all_file_list:
        match = re.search(r'(cid_\d+)', file_name)
        if match:
            cid_prefix = match.group(1)
            file_groups[cid_prefix].append(file_name)

    for cid in file_groups:
        file_groups[cid].sort()

    return file_groups

file_grouping = get_file_groups(directory_path)


########################################################################


# 시멘틱 청킹 수행 
embedding_model_ckpt = "nomic-ai/modernbert-embed-base"

tokenizer = AutoTokenizer.from_pretrained(embedding_model_ckpt)
langchain_emgbeddings = HuggingFaceEmbeddings(model_name= embedding_model_ckpt,
                                             model_kwargs={'device':'cuda:0'},
                                             encode_kwargs = {'normalize_embeddings': True}
                                             )

text_spliter = SemanticChunker(langchain_emgbeddings,
                              breakpoint_threshold_type = 'standard_deviation',
                              breakpoint_threshold_amount = 0.8
                              )


directory = "TEXT/"
total_chunk = []
for group in tqdm(file_grouping, desc = 'Chunking'):
    per_group_file = file_grouping[group]
    text_temp = []
    
    for file in per_group_file:
        input_path = os.path.join(directory, file)
        item = pd.read_json(input_path, typ = 'series', encoding='utf-8-sig')
        text_temp.append(item.content)

    merge_docs = ' '.join(text_temp)

    tokenization = tokenizer(merge_docs)
    if len(tokenization.input_ids) > 8192:
        print(f"토큰 개수 초과! {len(tokenization.input_ids)}")

    doc_chunk = text_spliter.create_documents(
        [merge_docs]
    )

    total_chunk.extend(doc_chunk)


import gc
gc.collect()
torch.cuda.empty_cache()


vector_db = Chroma(
    embedding_function = langchain_emgbeddings,
    persist_directory = './VectorDB_std_0.8'
)

batch_size = 32
idx = 0
for i in tqdm(range(0, len(total_chunk), batch_size), desc = 'Adding to Chroma'):
    batch = total_chunk[idx : idx + batch_size]
    vector_db.add_documents(documents = batch)
    gc.collect()
    torch.cuda.empty_cache()

    idx += batch_size


########################################################################


# 테스트 데이터 준비 및 벡터 DB 로딩
df = pd.read_csv('medDataset_processed.csv')

def filter_dataframe(df, name:str):
    new_df = df[df['qtype'] == name].reset_index(drop=True)

    return new_df

symptom_df = filter_dataframe(df, 'symptoms')
information_df = filter_dataframe(df, 'information')
treatment_df = filter_dataframe(df, 'treatment')
cause_df = filter_dataframe(df, 'causes')

test_df = pd.concat((symptom_df[:300], information_df[:300], treatment_df[:300], cause_df[:300])).reset_index(drop=True)

test_df.drop('qtype', axis=1, inplace=True)


llm_model_ckpt = "google/gemma-2-2b-it"
embedding_model_ckpt = "nomic-ai/modernbert-embed-base"

langchain_emgbeddings = HuggingFaceEmbeddings(model = embedding_model_ckpt,
                                              model_kwargs = {'device':'cuda:0'},
                                              encode_kwargs = {'normalize_embeddings': True} 
                                             )

llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_ckpt)

load_VDB = Chroma(
    persist_directory="VectorDB_std_0.8/",
    embedding_function=langchain_emgbeddings # 반드시 청킹 시 사용한 임베딩 모델과 query에 적용하는 임베딩 모델은 동일해야함!
)


data_list = load_VDB.get() 
all_chunks = [
    Document(page_content=text)
    for text in data_list['documents']
]  # BM25 사용위해 청크의 모든 청크 준비 


########################################################################


# context 후보 생성 및 리랭킹 적용
rerank_embedding_ckpt = "tomaarsen/reranker-ModernBERT-base-gooaq-bce"
rerank_model = CrossEncoder(rerank_embedding_ckpt)

bm25 = BM25Retriever.from_documents(all_chunks)
bm25.k = 5

def medical_collator(batch, use_rag=False):
    formatted_prompts = []
    for item in batch:
        query = item['question']
        if use_rag:
            similarity_docs = load_VDB.max_marginal_relevance_search(query, k = 5)
            bm25_docs =  bm25.get_relevant_documents(query)
            
            add_inform = []
            for i in range(len(similarity_docs)):
                add_inform.append(similarity_docs[i].page_content)
                add_inform.append(bm25_docs[i].page_content)

            rerank = rerank_model.rank(   
                query,
                add_inform
            )  # query와 context 후보 간 유사도 점수 가짐

            context = []
            for c in rerank:
                if c['score'] > 0.8:
                    temp_idx = c['corpus_id']
                    context.append(add_inform[temp_idx])

            prompt = f"""
                <bos>[Role]: You are a highly skilled medical professional. 
                [Instruction]: Answer the user's question by prioritizing the provided [context]. 
                If the [context] does not contain the answer, use your own extensive medical knowledge to provide a highly accurate response.
                
                [context]: {context}
                
                <start_of_turn>user
                {query}<end_of_turn>
                <start_of_turn>model
                
                """

 
        else:
            
            prompt = f"""
                <bos>[Role]: You are a highly skilled medical professional.
                Answer the user's question question.
               
                <start_of_turn>user
                {query}<end_of_turn>
                <start_of_turn>model
                
                """
        
        formatted_prompts.append(prompt)

    token_length = 3000 if use_rag else 50
    
    return llm_tokenizer(
                formatted_prompts,
                padding=True,
                truncation=True,
                max_length=token_length,
                return_tensors='pt'
           ).to(device)


########################################################################


# Test
class for_TEST(nn.Module):
    def __init__(self):
        super().__init__()
          
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_ckpt,
                                                             torch_dtype=torch.float16).to(device)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_ckpt)
        self.sentence_embedding = SentenceTransformer(embedding_model_ckpt).to(device)

        
        self.cs = nn.CosineSimilarity(dim =1, eps=1e-8)

    def vanila_LLM(self, input_ids, attention_mask):
        input_len = input_ids.shape[1]
        
        generate_token_ids = self.llm_model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = 2000,
            repetition_penalty=1.2,
            do_sample = False, 
            eos_token_id = self.llm_tokenizer.eos_token_id,
            pad_token_id = self.llm_tokenizer.pad_token_id
        ) # [batch_size, generate_sequence]

        only_gen_ids = generate_token_ids[:,input_len:]

        v_generate_text = self.llm_tokenizer.batch_decode(only_gen_ids, 
                                  skip_special_tokens=True)   # batch_decode
        
        return v_generate_text

    
    def RAG_LLM(self, input_ids, attention_mask):
        input_len = input_ids.shape[1]
        
        generate_token_ids = self.llm_model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = 2000,
            repetition_penalty=1.2,
            do_sample = False,
            eos_token_id = self.llm_tokenizer.eos_token_id,
            pad_token_id = self.llm_tokenizer.pad_token_id
        ) 

        only_gen_ids = generate_token_ids[:,input_len:]

        rag_generate_text = self.llm_tokenizer.batch_decode(only_gen_ids, 
                                  skip_special_tokens=True)   # batch_decode
        
        return rag_generate_text
        
    
    def v_test_code(self):
        self.eval()
        v_result = []
        for batch in tqdm(vanila_dataloader, total=len(vanila_dataloader), desc='vanilaLLM generation'):
            with torch.no_grad():

                v_generate_text = self.vanila_LLM(**batch)
                v_result.append(v_generate_text)
     
        return v_result

    
    def rag_test_code(self):
        self.eval()
        rag_result = []
        for batch in tqdm(rag_dataloader, total=len(rag_dataloader), desc='RAG + LLM generation'):
            with torch.no_grad():

                rag_generate_text = self.RAG_LLM(**batch)
                rag_result.append(rag_generate_text)
     
        return rag_result


########################################################################


# 결과 산출 
vanila_answer = []
for i in range(len(v_text)):
    for j in range(len(v_text[i])):
        vanila_answer.append(v_text[i][j]) 

rag_answer = []
for i in range(len(r_text)):
    for j in range(len(r_text[i])):
        rag_answer.append(r_text[i][j])

test_df['vanila_answer'] = vanila_answer
test_df['rag_answer'] = rag_answer


def make_embedding(column_name : str):
    
    result = []
    for i in range(len(test_df)):
        result.append(test_df[column_name][i])

    embedding_model = SentenceTransformer(embedding_model_ckpt).to(device)

    mass_embedding = embedding_model.encode(result)

    return torch.tensor(mass_embedding)


premise_embedding = make_embedding('Answer')
vanila_embedding = make_embedding('vanila_answer')
rag_embedding = make_embedding('rag_answer')


def extract_average_cosine(embedding_1, embedding_2):
    cosine_list = F.cosine_similarity(embedding_1, embedding_2, dim=1)
    return torch.mean(cosine_list)

vanila_sim_results = extract_average_cosine(premise_embedding, vanila_embedding)
rag_sim_results = extract_average_cosine(premise_embedding, rag_embedding)

print(f'COSINE_SIMILARITY By Vanila_LLM: {vanila_sim_results*100:.3f}')
print(f'COSINE_SIMILARITY By RAG_LLM: {rag_sim_results*100:.3f}')
