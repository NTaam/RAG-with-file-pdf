from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# cấu hình
model_file = "model/vinallama-7b-chat_q5_0.gguf"

# load LLM
def load_llm(model_file):
    llm = CTransformers(model=model_file, 
                        model_type="llama", 
                        # n_gpu_layers=1000, 
                        temperature=0.01, # độ sáng tạo của mô hình
                        max_new_tokens=1024)
    return llm

# tạo prompt template
def create_prompt(template):
    prompt = PromptTemplate(template=template,
                            input_variables=["question"])
    return prompt

# tạo simple chain
def create_simple_chain(prompt, llm):
    llm_chain = LLMChain(prompt=prompt,
                     llm=llm)
    return llm_chain

# chạy thử
# template = """Bạn là một trợ lý ảo thông minh. Hãy trả lời một cách chính xác."""
template = """ <|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
prompt = create_prompt(template)
llm = load_llm(model_file)
llm_chain = create_simple_chain(prompt, llm)

question = "Thành phố Hà Nội có bao nhiêu quận?"
response = llm_chain.invoke({"question": question})
print(response)