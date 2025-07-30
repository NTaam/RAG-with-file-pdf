from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings



# Khai báo biến
pdf_data_bath = "data"
vector_db_path = "vectorstores/db_faiss"

# Tao ra vector db từ 1 đoạn text
def create_db_from_text():
    raw_text = """Quán Nướng Linh Anh là lựa chọn lý tưởng nếu bạn muốn thưởng thức các món nướng Hàn-Việt tại Hà Đông với giá cả hợp lý, 
    không gian điều hòa rộng rãi và đồ ăn tươi ngon. Nhà hàng phù hợp cho nhóm bạn hoặc gia đình, 
    cần lưu ý kiểm soát đơn gọi thêm (như rau) để tránh phát sinh không mong muốn."""
    # chia nhỏ đoạn text thành các chunk nhỏ hơn
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 30,
        chunk_overlap = 10,
        length_function = len
    )

    chunks = text_splitter.split_text(raw_text)

    # embed các chunk
    embedding_model = GPT4AllEmbeddings(model = "models/all-MiniLM-L6-v2-f16.gguf")

    # Đưa vào Faiss vector db
    db = FAISS.from_texts(texts = chunks, 
                          embedding = embedding_model)
    db.save_local(vector_db_path)
    return 

def create_db_from_files():
    # khai bao loader de doc all data 
    loader = DirectoryLoader(pdf_data_bath, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    chunks = text_splitter.split_documents(documents)
    # embed các chunk
    embedding_model = GPT4AllEmbeddings(model = "models/all-MiniLM-L6-v2-f16.gguf")

    # db = FAISS.from_documents(chunks, embedding_model)
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    db.save_local(vector_db_path)
    return db

create_db_from_files()