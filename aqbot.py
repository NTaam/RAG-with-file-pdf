from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores.faiss import FAISS


# cấu hình
model_file = "model/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

# load LLM
def load_llm(model_file):
    llm = CTransformers(model=model_file, 
                        model_type="llama", 
                        n_gpu_layers=60, # số lượng lớp LLM sử dụng GPU
                        temperature=0.4, # độ sáng tạo của mô hình
                        max_new_tokens=50)
    return llm

# tạo prompt template
def create_prompt(template):
    prompt = PromptTemplate(template=template,
                            input_variables=["context", "question"])
    return prompt

# tạo simple chain
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs = {"k": 1}, max_tokens_limit=512),
        return_source_documents = False,
        chain_type_kwargs = {"prompt": prompt}
    )
    return llm_chain

# 
def read_vector_db():
    # 
    embedding_model = GPT4AllEmbeddings(model = "models/all-MiniLM-L6-v2-f16.gguf")
    # db = FAISS.load_local(vector_db_path, embedding_model)
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db


# 
db = read_vector_db()
llm = load_llm(model_file)
#
template = """ <|im_start|>system
# Bạn là một trợ lí AI trực tuyến của shop quần áo. 
# Hãy dựa vào thông tin trong ngữ cảnh để trả lời các câu hỏi của người dùng. 
# Nếu bạn không biết câu trả lời, hãy nói là "Tôi không biết".
# Hãy trả lời gắn gọn và đầy đủ thông tin trong khoảng 3 câu. Không được tự ý thêm thông tin ngoài ngữ cảnh.
Bạn là trợ lý AI của shop quần áo. Hãy dựa vào thông tin được cung cấp để trả lời câu hỏi của người dùng.
Trả lời ngắn gọn và đầy đủ, không thêm thông tin ngoài ngữ cảnh.
{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
prompt = create_prompt(template)
llm_chain = create_qa_chain(prompt, llm, db)
question = "phí vận chuyển?"
response = llm_chain.invoke({"query": question})
print(response)