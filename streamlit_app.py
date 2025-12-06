"""
Interactive RAG System Demo - Streamlit UI
Allows testing and comparison of all 5 QA systems on Basketball in Africa dataset
Usage: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Basketball Africa RAG System Demo",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #004E89;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B35;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_data():
    """Load all predictions, evaluations, questions, and answers"""
    
    systems = ['None_gpt4omini', 'Azure_gpt4omini', 'Local_gpt4omini', 
               'Azure_flant5', 'Local_flant5']
    
    # Load questions
    questions_df = pd.read_csv('data/question.tsv', sep='\t', header=None, 
                               names=['question', 'type'])
    
    # Load gold answers
    answers_df = pd.read_csv('data/answer.tsv', sep='\t', header=None)
    gold_answers = []
    for idx, row in answers_df.iterrows():
        answers = [str(val) for val in row.values if pd.notna(val) and str(val).strip()]
        gold_answers.append(", ".join(answers))
    
    # Load all system data
    all_data = {}
    
    for system in systems:
        # Load predictions
        pred_file = f'output/prediction/{system}.tsv'
        pred_df = pd.read_csv(pred_file, sep='\t', header=None)
        predictions = pred_df[0].tolist()
        retrieved_docs = pred_df[1].tolist() if len(pred_df.columns) > 1 else [None] * len(predictions)
        
        # Load evaluation results
        eval_file = f'output/evaluation/{system}.tsv'
        eval_df = pd.read_csv(eval_file, sep='\t', header=None,
                             names=['llm_score', 'exact_match', 'f1_score'])
        
        all_data[system] = {
            'predictions': predictions,
            'retrieved_docs': retrieved_docs,
            'evaluations': eval_df
        }
    
    return questions_df, gold_answers, all_data, systems


@st.cache_data
def load_analysis_results():
    """Load Part 5 analysis results"""
    
    analysis_files = {
        'complete_error': 'output/analysis/complete_error_analysis.csv',
        'question_type': 'output/analysis/question_type_analysis.csv',
        'retrieval_quality': 'output/analysis/retrieval_quality_summary.csv',
        'hardest_questions': 'output/analysis/hardest_20_questions.csv',
        'easiest_questions': 'output/analysis/easiest_20_questions.csv',
        'calibration': 'output/analysis/calibration_analysis.csv'
    }
    
    results = {}
    for key, filepath in analysis_files.items():
        if os.path.exists(filepath):
            results[key] = pd.read_csv(filepath)
    
    return results


def display_score_card(llm_score, exact_match, f1_score):
    """Display evaluation scores as colored cards"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color = "🟢" if llm_score >= 4 else "🟡" if llm_score >= 3 else "🔴"
        st.metric("LLM Score", f"{llm_score}/5 {color}")
    
    with col2:
        color = "🟢" if exact_match == 1 else "🔴"
        st.metric("Exact Match", f"{int(exact_match*100)}% {color}")
    
    with col3:
        color = "🟢" if f1_score >= 0.7 else "🟡" if f1_score >= 0.4 else "🔴"
        st.metric("F1 Score", f"{f1_score:.2f} {color}")


def main():
    # Load data
    questions_df, gold_answers, all_data, systems = load_all_data()
    
    try:
        analysis_results = load_analysis_results()
    except:
        analysis_results = {}
    
    # Header
    st.markdown('<p class="main-header">🏀 Basketball Africa RAG System Demo</p>', 
                unsafe_allow_html=True)
    st.markdown("**Interactive demonstration of 5 Question-Answering systems on Basketball in Africa dataset**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("# 🏀 Basketball Africa")
        st.markdown("### RAG System Demo")
        st.markdown("### 🎯 Navigation")
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "🔍 Try Systems", "📊 Compare Systems", 
             "📈 Analytics", "🎓 About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.info(f"""
        **Questions:** 159  
        **Question Types:** 3  
        - Factoid: 53
        - List: 53  
        - Multiple Choice: 53
        
        **Systems:** 5  
        **Corpus Docs:** 53
        """)
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(questions_df, all_data, systems)
    elif page == "🔍 Try Systems":
        show_try_systems_page(questions_df, gold_answers, all_data, systems)
    elif page == "📊 Compare Systems":
        show_compare_page(questions_df, gold_answers, all_data, systems)
    elif page == "📈 Analytics":
        show_analytics_page(analysis_results)
    elif page == "🎓 About":
        show_about_page()


def show_home_page(questions_df, all_data, systems):
    """Home page with overview and quick stats"""
    
    st.markdown('<p class="sub-header">📋 Project Overview</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to the Basketball Africa QA System Demo!
        
        This interactive demo showcases **5 different question-answering systems** trained on 
        the Basketball in Africa dataset. The systems combine different retrieval and generation 
        approaches to answer questions about:
        
        - 🏀 Basketball Africa League (BAL)
        - 🌍 FIBA Africa competitions
        - 👥 African basketball players
        - 🏗️ Infrastructure and development
        - 📚 Historical context
        
        **Test each system, compare their performance, and explore the analytics!**
        """)
        
        st.success("✅ All 159 questions from the test set are available for exploration")
    
    with col2:
        st.markdown("### 🏆 System Overview")
        
        system_info = {
            "None_gpt4omini": "No Retrieval + GPT-4o-mini",
            "Azure_gpt4omini": "Azure Embeddings + GPT-4o-mini",
            "Local_gpt4omini": "Local Embeddings + GPT-4o-mini",
            "Azure_flant5": "Azure Embeddings + FLAN-T5",
            "Local_flant5": "Local Embeddings + FLAN-T5"
        }
        
        for sys, desc in system_info.items():
            st.markdown(f"**{sys}**")
            st.caption(desc)
    
    st.markdown("---")
    st.markdown('<p class="sub-header">📊 Quick Performance Stats</p>', unsafe_allow_html=True)
    
    # Compute overall stats
    cols = st.columns(5)
    
    for idx, system in enumerate(systems):
        with cols[idx]:
            evals = all_data[system]['evaluations']
            avg_llm = evals['llm_score'].mean()
            avg_em = evals['exact_match'].mean() * 100
            
            st.metric(
                label=system.replace('_', ' ').title(),
                value=f"{avg_llm:.2f}/5",
                delta=f"{avg_em:.0f}% EM"
            )


def show_try_systems_page(questions_df, gold_answers, all_data, systems):
    """Page to try different systems on selected questions"""
    
    st.markdown('<p class="sub-header">🔍 Try the Systems</p>', unsafe_allow_html=True)
    st.markdown("Select a question and system to see the prediction and evaluation.")
    
    # System selector
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_system = st.selectbox(
            "**Select System:**",
            systems,
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col2:
        question_type_filter = st.selectbox(
            "**Filter by Type:**",
            ["All", "factoid", "list", "multiple choice"]
        )
    
    # Filter questions
    if question_type_filter != "All":
        filtered_questions = questions_df[questions_df['type'] == question_type_filter]
    else:
        filtered_questions = questions_df
    
    st.markdown(f"**{len(filtered_questions)} questions available**")
    
    # Question selector
    question_options = [f"Q{idx+1}: {row['question'][:80]}..." 
                       for idx, row in filtered_questions.iterrows()]
    
    selected_q_display = st.selectbox(
        "**Select Question:**",
        question_options,
        key="question_selector"
    )
    
    # Get actual question index
    q_num = int(selected_q_display.split(":")[0].replace("Q", ""))
    q_idx = q_num - 1
    
    st.markdown("---")
    
    # Display question details
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Question")
        st.info(f"**Type:** {questions_df.loc[q_idx, 'type'].title()}")
        st.markdown(f"**{questions_df.loc[q_idx, 'question']}**")
    
    with col2:
        st.markdown("### ✅ Gold Answer")
        st.success(gold_answers[q_idx])
    
    st.markdown("---")
    
    # Display prediction
    st.markdown("### 🤖 System Prediction")
    
    prediction = all_data[selected_system]['predictions'][q_idx]
    retrieved = all_data[selected_system]['retrieved_docs'][q_idx]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Predicted Answer:**")
        st.markdown(f'<div class="success-box"><b>{prediction}</b></div>', 
                   unsafe_allow_html=True)
        
        if selected_system != "None_gpt4omini":
            st.markdown("**Retrieved Documents:**")
            st.caption(str(retrieved))
    
    with col2:
        st.markdown("**Evaluation Scores:**")
        llm_score = all_data[selected_system]['evaluations'].loc[q_idx, 'llm_score']
        em = all_data[selected_system]['evaluations'].loc[q_idx, 'exact_match']
        f1 = all_data[selected_system]['evaluations'].loc[q_idx, 'f1_score']
        
        display_score_card(llm_score, em, f1)
    
    # Analysis
    st.markdown("---")
    st.markdown("### 📊 Analysis")
    
    if llm_score == 5 and em == 1:
        st.markdown('<div class="success-box">✅ <b>Perfect answer!</b> System got it exactly right.</div>', 
                   unsafe_allow_html=True)
    elif llm_score >= 4:
        st.markdown('<div class="success-box">✅ <b>Good answer!</b> Semantically correct with minor differences.</div>', 
                   unsafe_allow_html=True)
    elif llm_score >= 3:
        st.markdown('<div class="warning-box">⚠️ <b>Acceptable answer.</b> Partially correct but missing key information.</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-box">❌ <b>Poor answer.</b> Mostly incorrect or irrelevant.</div>', 
                   unsafe_allow_html=True)


def show_compare_page(questions_df, gold_answers, all_data, systems):
    """Page to compare all systems side-by-side"""
    
    st.markdown('<p class="sub-header">📊 Compare All Systems</p>', unsafe_allow_html=True)
    st.markdown("See how all 5 systems perform on the same question.")
    
    # Question selector
    question_options = [f"Q{idx+1}: {row['question'][:80]}..." 
                       for idx, row in questions_df.iterrows()]
    
    selected_q_display = st.selectbox(
        "**Select Question to Compare:**",
        question_options,
        key="compare_question_selector"
    )
    
    q_num = int(selected_q_display.split(":")[0].replace("Q", ""))
    q_idx = q_num - 1
    
    st.markdown("---")
    
    # Display question
    st.markdown("### 📝 Question")
    st.info(f"**Type:** {questions_df.loc[q_idx, 'type'].title()}")
    st.markdown(f"**{questions_df.loc[q_idx, 'question']}**")
    
    st.markdown(f"**Gold Answer:** {gold_answers[q_idx]}")
    
    st.markdown("---")
    st.markdown("### 🤖 System Predictions")
    
    # Create comparison table
    comparison_data = []
    
    for system in systems:
        prediction = all_data[system]['predictions'][q_idx]
        llm_score = all_data[system]['evaluations'].loc[q_idx, 'llm_score']
        em = all_data[system]['evaluations'].loc[q_idx, 'exact_match']
        f1 = all_data[system]['evaluations'].loc[q_idx, 'f1_score']
        
        comparison_data.append({
            'System': system.replace('_', ' ').title(),
            'Prediction': prediction[:100] + "..." if len(str(prediction)) > 100 else prediction,
            'LLM Score': f"{llm_score}/5",
            'Exact Match': "✅" if em == 1 else "❌",
            'F1 Score': f"{f1:.2f}"
        })
    
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    # Visualization
    st.markdown("---")
    st.markdown("### 📈 Score Comparison")
    
    fig = go.Figure()
    
    for system in systems:
        llm_score = all_data[system]['evaluations'].loc[q_idx, 'llm_score']
        fig.add_trace(go.Bar(
            name=system.replace('_', ' ').title(),
            x=['LLM Score'],
            y=[llm_score],
            text=[f"{llm_score}/5"],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="LLM Scores for This Question",
        yaxis_title="Score",
        yaxis_range=[0, 5],
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_analytics_page(analysis_results):
    """Page showing Part 5 analysis results"""
    
    st.markdown('<p class="sub-header">📈 Performance Analytics</p>', unsafe_allow_html=True)
    st.markdown("Comprehensive analysis of all 5 systems across 159 questions.")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overall Performance", 
        "🎯 Question Types", 
        "🔍 Retrieval Quality",
        "📉 Difficulty Analysis"
    ])
    
    with tab1:
        st.markdown("### Overall System Performance")
        
        # Load and display charts
        if os.path.exists('output/analysis/overall_comparison.png'):
            st.image('output/analysis/overall_comparison.png', use_container_width=True)
        
        if os.path.exists('output/analysis/component_analysis.png'):
            st.image('output/analysis/component_analysis.png', use_container_width=True)
        
        st.markdown("""
        **Key Findings:**
        - GPT-4o-mini systems significantly outperform FLAN-T5 systems
        - Azure and Local retrievers show similar performance
        - Generator choice has larger impact than retriever choice
        """)
    
    with tab2:
        st.markdown("### Performance by Question Type")
        
        if os.path.exists('output/analysis/question_type_heatmap.png'):
            st.image('output/analysis/question_type_heatmap.png', use_container_width=True)
        
        if 'question_type' in analysis_results:
            st.dataframe(analysis_results['question_type'], use_container_width=True)
    
    with tab3:
        st.markdown("### Retrieval Quality Analysis")
        
        if 'retrieval_quality' in analysis_results:
            st.dataframe(analysis_results['retrieval_quality'], use_container_width=True)
            
            st.markdown("""
            **Insights:**
            - Shows how often retrievers found the correct document
            - Compares performance when retrieval was correct vs incorrect
            - Helps identify if retriever is the bottleneck
            """)
    
    with tab4:
        st.markdown("### Question Difficulty Analysis")
        
        if os.path.exists('output/analysis/difficulty_distribution.png'):
            st.image('output/analysis/difficulty_distribution.png', use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'hardest_questions' in analysis_results:
                st.markdown("#### 🔴 Hardest Questions (Top 10)")
                hardest = analysis_results['hardest_questions'].head(10)
                for idx, row in hardest.iterrows():
                    st.caption(f"Q{row['question_idx']}: {row['question'][:100]}...")
                    st.caption(f"   Avg LLM: {row['avg_llm_score']:.2f}/5")
        
        with col2:
            if 'easiest_questions' in analysis_results:
                st.markdown("#### 🟢 Easiest Questions (Top 10)")
                easiest = analysis_results['easiest_questions'].head(10)
                for idx, row in easiest.iterrows():
                    st.caption(f"Q{row['question_idx']}: {row['question'][:100]}...")
                    st.caption(f"   Avg LLM: {row['avg_llm_score']:.2f}/5")


def show_about_page():
    """About page with project details"""
    
    st.markdown('<p class="sub-header">🎓 About This Project</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📚 HW6: Retrieval-Augmented Generation for Question Answering
    
    **Course:** 11-697 Introduction to Question Answering  
    **Institution:** Carnegie Mellon University Africa  
    **Topic:** Basketball in Africa
    
    ---
    
    ### 🎯 Project Goals
    
    This project explores different approaches to building Question-Answering systems:
    
    1. **Baseline System** (No Retrieval)
       - Tests generator's inherent knowledge
    
    2. **RAG Systems** (4 combinations)
       - API vs Open-weight Retrievers
       - API vs Open-weight Generators
       - Tests impact of each component
    
    3. **Evaluation** (3 metrics)
       - LLM-as-Judge (semantic quality)
       - Exact Match (strict correctness)
       - F1 Score (token overlap)
    
    4. **Advanced Analysis**
       - Error analysis (failure types)
       - Question type performance
       - Retrieval quality
       - Difficulty patterns
       - Calibration analysis
    
    ---
    
    ### 🛠️ Technologies Used
    
    **Retrievers:**
    - Azure text-embedding-3-small (API)
    - sentence-transformers/all-MiniLM-L6-v2 (Open-weight)
    
    **Generators:**
    - GPT-4o-mini (API)
    - google/flan-t5-base (Open-weight)
    
    **Evaluation:**
    - GPT-4o-mini as judge
    - Custom metrics (EM, F1)
    
    **Visualization:**
    - Matplotlib, Seaborn, Plotly
    - Streamlit for interactive demo
    
    ---
    
    ### 📊 Dataset Statistics
    
    - **Total Questions:** 159
    - **Question Types:** 
      - Factoid: 53 (33.3%)
      - List: 53 (33.3%)
      - Multiple Choice: 53 (33.3%)
    - **Corpus Documents:** 53
    - **Total Words:** ~11,263
    - **Topics:** BAL, FIBA Africa, Players, Infrastructure, History
    
    ---
    
    ### 🏆 Key Findings
    
    1. **Generator matters more than Retriever**
       - GPT-4o-mini: ~4.8/5 LLM score
       - FLAN-T5: ~2.3/5 LLM score
    
    2. **Retrieval helps significantly**
       - No retrieval: 16.4% exact match
       - With retrieval: 31.4% exact match
    
    3. **Question types have different difficulty**
       - Factoid questions: Slightly easier
       - List questions: More challenging
    
    4. **Retrieval accuracy matters**
       - Correct retrieval: +0.5 LLM score improvement
    
    ---
    
    ### 👨‍💻 Author
    
    **Joel Maison**  
    MSIT Student, CMU-Africa  
    
    ---
    ### 📄 References
    """)


if __name__ == "__main__":
    main()
