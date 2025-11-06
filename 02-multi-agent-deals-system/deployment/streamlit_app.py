import streamlit as st
import sys
import os
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from framework import DealAgentFramework

# Page config
st.set_page_config(
    page_title="Multi-Agent Deals System",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #0A1628;
        margin-bottom: 1rem;
    }
    .agent-card {
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 0.5rem 0;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f5f5f5;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🤖 Multi-Agent Deals System</div>', unsafe_allow_html=True)
st.markdown("**Autonomous AI Framework for Product Deal Discovery**")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/0A1628/FFFFFF?text=Prasad+Pagade", use_column_width=True)
    
    st.markdown("### System Status")
    st.success("🟢 All Agents Online")
    
    st.markdown("### Quick Actions")
    if st.button("🚀 Run Discovery", use_container_width=True):
        with st.spinner("Running agent framework..."):
            framework = DealAgentFramework()
            results = framework.run()
            st.success(f"✅ Found {len(results)} opportunities!")
            st.rerun()
    
    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This dashboard monitors a multi-agent AI system that discovers and evaluates product deals.
    
    **7 Agents Working Together:**
    - Planning Agent
    - Scanner Agent  
    - Random Forest Agent
    - Frontier Agent
    - Ensemble Agent
    - Messaging Agent
    - Specialist Agent
    """)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 Agents", "💎 Opportunities", "🔬 Analysis"])

with tab1:
    # Load memory
    memory_file = Path(__file__).parent.parent / "src" / "memory.json"
    if memory_file.exists():
        with open(memory_file, "r") as f:
            opportunities = json.load(f)
    else:
        opportunities = []
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Opportunities", len(opportunities))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        high_conf = len([o for o in opportunities if o.get('confidence', 0) > 0.8])
        st.metric("High Confidence", high_conf)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        categories = set([o.get('category', 'Unknown') for o in opportunities])
        st.metric("Categories", len(categories))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_price = sum([o.get('price', 0) for o in opportunities]) / len(opportunities) if opportunities else 0
        st.metric("Avg Price", f"${avg_price:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Opportunities by Category")
        if opportunities:
            df = pd.DataFrame(opportunities)
            category_counts = df['category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Distribution of Deals"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No opportunities yet. Run discovery to populate!")
    
    with col2:
        st.subheader("Confidence Distribution")
        if opportunities:
            df = pd.DataFrame(opportunities)
            fig = px.histogram(
                df,
                x="confidence",
                nbins=20,
                title="Deal Confidence Scores"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No opportunities yet. Run discovery to populate!")

with tab2:
    st.subheader("🤖 Agent Architecture")
    
    st.markdown("""
    ### 7 Specialized Agents
    
    Each agent has a specific role in the discovery and evaluation pipeline:
    """)
    
    agents = [
        {"name": "Planning Agent", "icon": "🎯", "role": "Orchestrates overall strategy", "status": "Active"},
        {"name": "Scanner Agent", "icon": "🔍", "role": "Discovers new deals via semantic search", "status": "Active"},
        {"name": "Random Forest Agent", "icon": "🌲", "role": "ML-based deal classification", "status": "Active"},
        {"name": "Frontier Agent", "icon": "🚀", "role": "LLM-powered nuanced evaluation", "status": "Active"},
        {"name": "Ensemble Agent", "icon": "🎭", "role": "Combines multiple model predictions", "status": "Active"},
        {"name": "Messaging Agent", "icon": "📱", "role": "Sends real-time notifications", "status": "Active"},
        {"name": "Specialist Agent", "icon": "🎓", "role": "Category-specific expertise", "status": "Active"},
    ]
    
    for agent in agents:
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            st.markdown(f"<h1 style='text-align: center;'>{agent['icon']}</h1>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{agent['name']}**")
            st.caption(agent['role'])
        with col3:
            st.success(f"✅ {agent['status']}")
    
    st.markdown("---")
    
    st.subheader("Agent Coordination Flow")
    st.code("""
    1. Planning Agent → Analyzes state, decides strategy
    2. Scanner Agent → Queries vector DB for candidates
    3. Specialist Agents → Evaluate each candidate
    4. Ensemble Agent → Aggregates recommendations
    5. Messaging Agent → Sends notifications
    6. Memory → Persists opportunities
    """, language="text")

with tab3:
    st.subheader("💎 Current Opportunities")
    
    if opportunities:
        # Sort by confidence
        sorted_opps = sorted(opportunities, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for i, opp in enumerate(sorted_opps[:10], 1):
            with st.expander(f"#{i} - {opp.get('product_name', 'Unknown')} ({opp.get('category', 'N/A')})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Price", f"${opp.get('price', 0):.2f}")
                
                with col2:
                    st.metric("Rating", f"{opp.get('rating', 0):.1f} ⭐")
                
                with col3:
                    conf = opp.get('confidence', 0)
                    st.metric("Confidence", f"{conf*100:.0f}%")
                
                st.markdown(f"**Description:** {opp.get('description', 'No description available')}")
                st.caption(f"Discovered: {opp.get('timestamp', 'Unknown')}")
    else:
        st.info("📭 No opportunities found yet. Click 'Run Discovery' in the sidebar!")

with tab4:
    st.subheader("🔬 Vector Space Analysis")
    
    st.markdown("""
    This section visualizes the product embedding space using t-SNE dimensionality reduction.
    Products with similar features cluster together.
    """)
    
    try:
        with st.spinner("Loading vector data..."):
            documents, vectors, colors = DealAgentFramework.get_plot_data(max_datapoints=1000)
            
            if len(vectors) > 0:
                # Create 3D scatter plot
                fig = go.Figure(data=[go.Scatter3d(
                    x=vectors[:, 0],
                    y=vectors[:, 1],
                    z=vectors[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=colors,
                        opacity=0.8
                    ),
                    text=documents[:100],  # Only show first 100 for performance
                    hoverinfo='text'
                )])
                
                fig.update_layout(
                    title="Product Embeddings in 3D Space",
                    scene=dict(
                        xaxis_title="Component 1",
                        yaxis_title="Component 2",
                        zaxis_title="Component 3"
                    ),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✅ Visualizing {len(vectors)} products across {len(set(colors))} categories")
            else:
                st.warning("Vector database is empty. Initialize with sample data first.")
    except Exception as e:
        st.error(f"Error loading vector data: {str(e)}")
        st.info("Make sure the vector database is initialized with `python src/setup_vectorstore.py`")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Built by Prasad Pagade | Multi-Agent AI Systems<br>
    <a href='https://github.com/prasadpagade'>GitHub</a> | 
    <a href='https://linkedin.com/in/prasadpagade'>LinkedIn</a>
</div>
""", unsafe_allow_html=True)
