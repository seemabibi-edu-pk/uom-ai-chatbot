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
    
    print("\n===== RETRIEVED RESULTS =====")

    for result in results[:top_k]:
        print("TITLE/CONTENT:")
        print(result["text"])
        print("KEYWORD SCORE:", result["keyword_score"])
        print("SEMANTIC SCORE:", result["semantic_score"])
        print("SOURCE:", result["source"])
        print("----------------------------")

    return results[:top_k]


# ==========================================
# ANSWER GENERATION
# ==========================================

def generate_answer(question, top_k=5):

    # Retrieve relevant information from UOM dataset
    results = retrieve(question, top_k=top_k)

    st.write("### Retrieved Information")

    for result in results:
        st.write(result["text"])
        st.write("Keyword score:", result["keyword_score"])
        st.write("Semantic score:", result["semantic_score"])
        st.write("Source:", result["source"])
        st.write("---")

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
Answer the user's question using ONLY the retrieved University of Malakand information.

Rules:
1. Give a direct answer.
2. If the question is a Yes/No question, start with "Yes." or "No."
3. If the question asks "how many", give the number directly.
4. If the question asks "what", give the requested information directly.
5. If the question asks "which", list the requested items directly.
6. Do not use "Yes" or "No" for questions that are not Yes/No questions.
7. Keep the answer short and simple.
8. If the retrieved information contains the answer, always answer from it.
9. Do not invent information.
10. Do not mention keyword scores or semantic scores in the answer.
11. Do not provide the source inside the answer.

Retrieved Information:
{context}

Question:
{question}

Answer:
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
