import os
import tempfile
import streamlit as st
from pdf_processing import extract_pages_from_pdf
from chunking import chunk_pages
from embeddings import get_embedding_model, build_index_from_documents
from edubot import answer_question_simple

# Configuration constants
MAX_PDF_SIZE_MB = 50
MIN_QUESTION_LENGTH = 3
DEFAULT_K_RESULTS = 3

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="EDUBOT - AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    body {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 600;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton > button {
        background-color: #4f7cff;
        color: white;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #3d5fcc;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def generate_memory_trick(answer: str):
    """Generate memory tricks based on actual answer content"""
    
    # Extract sentences and clean them
    sentences = [s.strip() for s in answer.split('.') if s.strip() and len(s.strip()) > 10]
    
    if not sentences:
        sentences = [line.strip() for line in answer.split('\n') if line.strip() and len(line.strip()) > 10]
    
    st.write("### 📌 Key Points from Answer:")
    
    # Display up to 5 key points
    key_phrases = sentences[:5]
    for i, phrase in enumerate(key_phrases, 1):
        # Clean the phrase
        clean_phrase = phrase[:100] + "..." if len(phrase) > 100 else phrase
        st.write(f"**{i}.** {clean_phrase}")
    
    # Create acronym from key points
    st.write("---")
    st.write("### 🔤 Acronym Method:")
    
    words = []
    for phrase in key_phrases:
        # Get first meaningful word
        words_in_phrase = phrase.split()
        if words_in_phrase:
            words.append(words_in_phrase[0][0].upper())
    
    if words:
        acronym = ''.join(words)
        st.write(f"**Acronym: `{acronym}`**")
        st.write(f"💡 Remember '{acronym}' to recall all these key concepts!")
        
        # Explain each letter
        st.write("**Breaking it down:**")
        for letter, phrase in zip(words, key_phrases):
            st.write(f"- **{letter}** = {phrase[:80]}...")
    
    # Memory techniques
    st.write("---")
    st.write("### 💡 Memory Techniques:")
    st.write("1. **Visualization:** Create a mental picture or diagram of these concepts")
    st.write("2. **Association:** Connect each point with something you already know")
    st.write("3. **Repetition:** Read these points multiple times")
    st.write("4. **Rhyming:** Try to remember using rhymes or patterns")
    st.write("5. **Story Method:** Create a story connecting all these points")
    
    # Summary box
    st.write("---")
    st.write("### 📝 Quick Summary:")
    st.write("This concept covers:")
    for i, phrase in enumerate(key_phrases, 1):
        main_concept = phrase.split()[0] if phrase.split() else "concept"
        st.write(f"- {main_concept.capitalize()}")


def generate_flowchart(question: str):
    """Generate a flowchart"""
    st.code("""
┌──────────────────────┐
│   ❓ Your Question   │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  🔍 Search in PDF   │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  🧠 Process & Match │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  📝 Generate Answer │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  ✅ Answer Ready    │
└──────────────────────┘
    """)


def generate_key_points(answer: str):
    """Generate key points from actual answer"""
    sentences = [s.strip() for s in answer.split('.') if s.strip()]
    
    st.write("### 📌 Important Points:")
    
    if sentences:
        for i, sentence in enumerate(sentences[:8], 1):
            clean_sentence = sentence[:150] + "..." if len(sentence) > 150 else sentence
            st.write(f"**{i}.** {clean_sentence}")
    else:
        st.write("Key points will be extracted from the answer.")


def generate_viva_questions(answer: str):
    """Generate viva questions based on answer"""
    
    # Extract main topic from answer
    words = answer.split()
    main_concept = words[0] if words else "concept"
    
    st.write("### 🎤 Possible Viva Questions:")
    
    st.write(f"**Q1.** Define {main_concept.lower()} in your own words?")
    st.write(f"**A1.** Based on the answer: The key definition is present in the provided content.")
    st.write("---")
    
    st.write(f"**Q2.** What are the real-life applications of {main_concept.lower()}?")
    st.write("**A2.** Look for application points in the answer content.")
    st.write("---")
    
    st.write(f"**Q3.** How does {main_concept.lower()} relate to other concepts?")
    st.write("**A3.** Find connections and relationships in the provided explanation.")
    st.write("---")
    
    st.write(f"**Q4.** Can you provide examples of {main_concept.lower()}?")
    st.write("**A4.** Examples are included in the detailed answer above.")


def generate_exam_answers(answer: str):
    """Generate exam-style answers based on actual answer"""
    
    sentences = [s.strip() for s in answer.split('.') if s.strip()]
    
    st.write("### 📝 Answer Formats for Different Marks:")
    
    # 2 Marks Answer
    st.write("---")
    st.write("### ⏱️ 2 Marks Answer:")
    st.write("Keep it brief and to the point. Use 2-3 sentences maximum.")
    if sentences:
        st.write(f"**Answer:** {sentences[0]}")
    else:
        st.write("**Answer:** Provide a brief definition of the concept.")
    
    # 5 Marks Answer
    st.write("---")
    st.write("### ⏱️ 5 Marks Answer:")
    st.write("Include definition, 1-2 explanations, and 1 example.")
    if sentences:
        st.write(f"**Answer:**")
        st.write(f"1. **Definition:** {sentences[0]}")
        if len(sentences) > 1:
            st.write(f"2. **Explanation:** {sentences[1]}")
        if len(sentences) > 2:
            st.write(f"3. **Example:** {sentences[2][:100]}...")
    else:
        st.write("**Answer:** Expand with explanation and examples.")
    
    # 10 Marks Answer
    st.write("---")
    st.write("### ⏱️ 10 Marks Answer:")
    st.write("Provide complete definition, detailed explanation, multiple examples, and related concepts.")
    st.write("**Answer:**")
    if sentences:
        for i, sentence in enumerate(sentences[:5], 1):
            st.write(f"{i}. {sentence[:100]}...")
    else:
        st.write("Provide a detailed comprehensive answer with all aspects covered.")
    
    st.write("\n*Note: Refer to the Explanation tab for the complete detailed answer.*")


# ==================== MAIN UI ====================

st.title("📚 EDUBOT")
st.subheader("Learn Faster with Your PDFs")
st.caption("Upload • Ask • Understand")

st.write("---")

# Features Section
st.subheader("✨ Key Features")
cols = st.columns(8)
features = [
    ("🔍", "Search"),
    ("❓", "QA"),
    ("💡", "Smart"),
    ("🧠", "Memory"),
    ("🌳", "Flowchart"),
    ("📝", "Exam"),
    ("🎤", "Viva"),
    ("⚡", "Fast")
]

for col, (icon, label) in zip(cols, features):
    with col:
        st.write(f"{icon}\n**{label}**")

st.write("---")

# PDF Upload Section
st.subheader("📄 Upload Your PDF")
uploaded_file = st.file_uploader("Select PDF file", type=["pdf"])

if uploaded_file:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    file_name = uploaded_file.name
    st.success(f"✅ {file_name} ({file_size_mb:.2f} MB) - Ready")

st.write("---")

# Question Section
st.subheader("❓ Ask Your Question")
question = st.text_area(
    "Enter your question here:",
    placeholder="Example: What is photosynthesis? What is the water cycle?",
    height=80
)

st.write("---")

# Generate Button
generate_button = st.button("🔵 Generate Answer", use_container_width=True)

# ==================== PROCESSING ====================

if generate_button:
    errors = []
    
    # Validation
    if uploaded_file is None:
        errors.append("❌ Please upload a PDF file")
    
    if not question.strip():
        errors.append("❌ Please ask a question")
    elif len(question.strip()) < MIN_QUESTION_LENGTH:
        errors.append(f"❌ Question must be at least {MIN_QUESTION_LENGTH} characters")
    
    if uploaded_file and uploaded_file.size > MAX_PDF_SIZE_MB * 1024 * 1024:
        errors.append(f"❌ PDF must be smaller than {MAX_PDF_SIZE_MB}MB")
    
    # Show errors
    if errors:
        for error in errors:
            st.error(error)
    else:
        # Start processing
        with st.spinner("⏳ Processing PDF..."):
            tmp_path = None
            
            try:
                # Save PDF temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                # Extract Pages
                pages = extract_pages_from_pdf(tmp_path)
                
                if not pages or all(not page.strip() for page in pages):
                    st.error("❌ PDF is empty or text could not be extracted")
                else:
                    # Chunk Text
                    docs = chunk_pages(pages)
                    
                    if not docs:
                        st.error("❌ No text found in PDF")
                    else:
                        # Build Embeddings
                        model = get_embedding_model()
                        index, vectors = build_index_from_documents(model, docs)
                        
                        # Generate Answer
                        answer = answer_question_simple(
                            index=index,
                            docs=docs,
                            query=question,
                            k=DEFAULT_K_RESULTS
                        )
                        
                        st.success("✅ Answer Generated Successfully!")
                        st.write("---")
                        
                        # Display in tabs
                        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                            "📖 Explanation",
                            "🧠 Memory Trick",
                            "🌳 Flowchart",
                            "🎯 Key Points",
                            "🎤 Viva Q&A",
                            "📝 Exam Answer"
                        ])
                        
                        with tab1:
                            st.write("### 💡 Complete Explanation")
                            st.write(answer)
                        
                        with tab2:
                            st.write("### 🧠 Memory Techniques")
                            generate_memory_trick(answer)
                        
                        with tab3:
                            st.write("### 🌳 Process Flowchart")
                            generate_flowchart(question)
                        
                        with tab4:
                            st.write("### 🎯 Essential Points")
                            generate_key_points(answer)
                        
                        with tab5:
                            st.write("### 🎤 Viva Questions & Answers")
                            generate_viva_questions(answer)
                        
                        with tab6:
                            st.write("### 📝 Exam-Format Answers")
                            generate_exam_answers(answer)
            
            except FileNotFoundError as e:
                st.error(f"❌ File Error: {str(e)}")
            except RuntimeError as e:
                st.error(f"❌ Runtime Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info(f"Details: {str(e)}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass