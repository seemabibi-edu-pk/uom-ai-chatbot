import streamlit as st
import faiss
import json
import numpy as np
import re

from PIL import Image
import pytesseract

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ==========================================
# PAGE
# ==========================================

st.set_page_config(
    page_title="University of Malakand AI Assistant",
    page_icon="🎓"
)

st.title("🎓 University of Malakand AI Assistant")

st.write(
    "Ask questions about the University of Malakand "
    "using text or an image."
)


# ==========================================
# LOAD DATABASE
# ==========================================

@st.cache_resource
def load_database():

    index = faiss.read_index(
        "uom.index"
    )

    with open(
        "chunks.json",
        "r",
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    with open(
        "sources.json",
        "r",
        encoding="utf-8"
    ) as f:
        sources = json.load(f)

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return (
        index,
        chunks,
        sources,
        embedding_model
    )


index, all_chunks, chunk_sources, embedding_model = (
    load_database()
)


# ==========================================
# LOAD LLM
# ==========================================

@st.cache_resource
def load_llm():

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    return tokenizer, model


tokenizer, model = load_llm()


# ==========================================
# RETRIEVAL
# ==========================================

def retrieve(question, top_k=5):

    question_words = set(
        re.findall(
            r'\b[a-zA-Z]+\b',
            question.lower()
        )
    )

    stop_words = {
        "how", "many", "what", "which",
        "where", "when", "who", "why",
        "is", "are", "the", "in", "of",
        "to", "a", "an", "for", "on",
        "does", "do", "can", "could",
        "please", "university",
        "malakand", "any", "there"
    }

    question_words = {
        word
        for word in question_words
        if word not in stop_words
        and len(word) > 2
    }

    # Semantic search

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    question_embedding = np.array(
        question_embedding,
        dtype="float32"
    )

    semantic_scores, ids = index.search(
        question_embedding,
        len(all_chunks)
    )

    semantic_scores = semantic_scores[0]
    ids = ids[0]

    results = []

    # Keyword + semantic search

    for semantic_score, idx in zip(
        semantic_scores,
        ids
    ):

        chunk = all_chunks[idx]

        chunk_lower = chunk.lower()

        title_match = re.search(
            r"Title:\s*(.*?)(?:\n|Category:)",
            chunk,
            re.IGNORECASE
        )

        if title_match:
            title = (
                title_match
                .group(1)
                .strip()
                .lower()
            )
        else:
            title = ""

        keyword_score = 0

        for word in question_words:

            if word in title:
                keyword_score += 5

            elif word in chunk_lower:
                keyword_score += 1

        results.append({
            "semantic_score": float(
                semantic_score
            ),
            "keyword_score": keyword_score,
            "text": chunk,
            "source": chunk_sources[idx]
        })

    results.sort(
        key=lambda x: (
            x["keyword_score"],
            x["semantic_score"]
        ),
        reverse=True
    )

    return results[:top_k]


# ==========================================
# ANSWER GENERATION
# ==========================================

def generate_answer(question, top_k=5):

    # Retrieve relevant information from UOM dataset
    results = retrieve(question, top_k=top_k)

    # Build context
    context_parts = []

    for result in results:
        context_parts.append(
            f"Title: {result['text']}\n"
            f"Source: {result['source']}"
        )

    context = "\n\n".join(context_parts)

    # Prompt for the language model
    prompt = f"""
You are the University of Malakand AI Assistant.

Answer the user's question using ONLY the information provided
in the UOM CONTEXT.

Important rules:
1. Carefully read all the context before answering.
2. Give a direct answer to the question.
3. If the context clearly contains the answer, use it.
4. Do not say "I don't know" when the answer is present.
5. Do not use outside knowledge.
6. Do not invent facts.
7. If the answer is not present in the context, say:
   "I could not find this information in the available
   University of Malakand data."
8. For yes/no questions, answer Yes or No first and then
   give a short explanation.

UOM CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    # Generate answer
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        num_beams=4,
        early_stopping=True
    )

    # Convert model output to text
    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    # Use the highest-ranked result as the source
    source = results[0]["source"] if results else ""

    return answer, source
```



# ==========================================
# TEXT QUESTION
# ==========================================

st.subheader("💬 Ask a question")

question = st.chat_input(
    "Example: What scholarships are available?"
)

if question:

    with st.chat_message("user"):

        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching UOM information..."
        ):

            answer, source = generate_answer(
                question
            )

        st.write(answer)

        st.markdown(
            f"**Source:** [{source}]({source})"
        )


# ==========================================
# IMAGE QUESTION
# ==========================================

st.subheader("🖼️ Ask using an image")

uploaded_image = st.file_uploader(
    "Upload a screenshot or image containing your question",
    type=["png", "jpg", "jpeg"]
)

if uploaded_image:

    image = Image.open(
        uploaded_image
    )

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    with st.spinner(
        "Reading text from image..."
    ):

        extracted_text = pytesseract.image_to_string(
            image
        )

    st.write("### Extracted question")

    st.write(extracted_text)

    if extracted_text.strip():

        with st.spinner(
            "Searching UOM information..."
        ):

            answer, source = generate_answer(
                extracted_text
            )

        st.write("### Answer")

        st.write(answer)

        st.markdown(
            f"**Source:** [{source}]({source})"
        )

    else:

        st.warning(
            "I could not read any text from the image."
        )
