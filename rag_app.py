#-> streamlit run rag_app.py

## for db
import chromadb  # 0.5.0
## for ai
import ollama  # 0.5.0
## for app
import streamlit as st  # 1.35.0
## for extracting text and embedding PDF
import pytesseract
from tqdm import tqdm
import pdf2image
import re

######################## Backend ##############################
class AI():
    def __init__(self):
        db = chromadb.PersistentClient()
        self.collection = db.get_or_create_collection("vector_db")

    def query(self, q, top=10):
        res_db = self.collection.query(query_texts=[q])["documents"][0][0:top]
        context = ' '.join(res_db).replace("\n", " ")
        return context

    def respond(self, lst_messages, model="llama3", use_knowledge=False):
        q = lst_messages[-1]["content"]
        context = self.query(q)

        if use_knowledge:
            prompt = "Give the most accurate answer using your knowledge and the following additional information: \n" + context
        else:
            prompt = "Give the most accurate answer using only the following information: \n" + context

        res_ai = ollama.chat(model=model,
                             messages=[{"role": "system", "content": prompt}] + lst_messages,
                             stream=True)
        for res in res_ai:
            chunk = res["message"]["content"]
            app["full_response"] += chunk
            yield chunk

    def extract_headings(self, page_text):
        # Example heading extraction from a single page
        return re.findall(r'\n(\d+\.?\d*)\s*(.+?)\n', page_text)

    def create_title_map(self, doc_text):
        title_map = {}
        current_title = None
        start_page = 1

        for page_number, page in enumerate(doc_text):
            headings = self.extract_headings(page)
            
            if headings:
                for heading in headings:
                    page_number = int(page_number) + 1  # Adjust to 1-based indexing
                    title = heading[1].strip()
                    if current_title:
                        title_map[f"{start_page}-{page_number-1}"] = current_title
                    start_page = page_number
                    current_title = title
                    
                # For the last heading in the document
                if current_title:
                    title_map[f"{start_page}-{page_number}"] = current_title
        
        return title_map

    def upload_document(self, file):
        collection_name = "vector_db"
        db = chromadb.PersistentClient()
        self.collection = db.get_or_create_collection("vector_db")
        images = pdf2image.convert_from_bytes(file.read())
        doc_text = []
        
        # Create progress bar
        with st.spinner("Processing document..."):
            progress_bar = st.progress(0)
            
            for i, img in enumerate(tqdm(images)):
                doc_text.append(pytesseract.image_to_string(img))
                # Update progress bar
                progress_bar.progress((i + 1) / len(images))
            
            # Complete progress bar
            progress_bar.progress(1.0)
            st.spinner("Processing complete!")
        
        # Debugging: Check the length of doc_text
        print(f"Number of pages processed: {len(doc_text)}")

        # Create title map
        title_map = self.create_title_map(doc_text)
        lst_docs, lst_ids, lst_metadata = [], [], []

        for n, page in enumerate(doc_text):
            try:
                # Get title from title_map
                title = [v for k, v in title_map.items() if n + 1 in range(int(k.split("-")[0]), int(k.split("-")[1]) + 1)]
                if title:
                    title = title[0]
                    # Clean page
                    page = page.replace("Table of Contents", "")
                    # Get paragraphs
                    for i, p in enumerate(page.split('\n\n')):
                        if len(p.strip()) > 5:
                            lst_docs.append(p.strip())
                            lst_ids.append(f"{n}_{i}")
                            lst_metadata.append({"title": title})
            except Exception as e:
                print(f"Error processing page {n}: {e}")
                continue

        # Debugging: Check the lengths of lists
        print(f"Number of documents: {len(lst_docs)}")
        print(f"Number of IDs: {len(lst_ids)}")
        print(f"Number of metadata entries: {len(lst_metadata)}")

        if len(lst_docs) == 0 or len(lst_ids) == 0:
            raise ValueError("No documents or IDs generated. Check document processing.")

        if collection_name in [c.name for c in db.list_collections()]:
            db.delete_collection(collection_name)
            print("--- deleted ---")

        collection = db.get_or_create_collection(name=collection_name,
                                                 embedding_function=chromadb.utils.embedding_functions.DefaultEmbeddingFunction())

        collection.add(documents=lst_docs, ids=lst_ids, metadatas=lst_metadata,
                       images=None, embeddings=None)


ai = AI()

######################## Frontend #############################
## Layout
st.title('💬 Write your questions')

# Chat section
app = st.session_state

if "messages" not in app:
    app["messages"] = [{"role": "assistant", "content": "I'm ready to retrieve information"}]

if 'full_response' not in app:
    app['full_response'] = '' 

## Keep messages in the Chat
for msg in app["messages"]:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="😎").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message(msg["role"], avatar="👾").write(msg["content"])

## Chat
if txt := st.chat_input():
    ### User writes
    app["messages"].append({"role": "user", "content": txt})
    st.chat_message("user", avatar="😎").write(txt)

    ### AI responds with chat stream
    app["full_response"] = ""
    st.chat_message("assistant", avatar="👾").write_stream(ai.respond(app["messages"]))
    app["messages"].append({"role": "assistant", "content": app["full_response"]})

st.sidebar.title("Document Upload")
# Keep document upload always at the top
with st.sidebar.container():
    st.header("Upload Document")
    document = st.file_uploader("Choose a document")
    if st.button("Upload Document"):
        if document:
            try:
                st.sidebar.write("Uploading document...")
                ai.upload_document(document)
                st.sidebar.success("Document uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
        else:
            st.sidebar.error("Please choose a document.")
