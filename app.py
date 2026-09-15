import streamlit as st
from agent_backend import AVAILABLE_PERSONAS, run_agent_turn

st.set_page_config(page_title="Deep Agent Workspace", layout="centered")

st.title("🤖 Deep Agent Production Workspace")

# --- Sidebar: User Authentication & Persona Selection ---
st.sidebar.header("Session Settings")

# 1. Multi-User Identification Log-in
user_id = st.sidebar.text_input("User Login / Email", value="default_user")

# 2. Persona Selector with default greeting options
persona_choice = st.sidebar.selectbox(
    "Select Agent Persona",
    options=list(AVAILABLE_PERSONAS.keys()),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Active User:** `{user_id}`")
st.sidebar.markdown(f"**Current Persona:** `{persona_choice}`")

# Clear chat history when user or persona changes to keep context clean
session_key = f"{user_id}_{persona_choice}"
if "current_session" not in st.session_state or st.session_state["current_session"] != session_key:
    st.session_state["current_session"] = session_key
    # Default initial message addressing the persona capabilities
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": f"Hello! I am your **{persona_choice}**. How can I help you today? (I can also switch personas in the sidebar if you need a Code Architect or Financial Analyst instead!)"
    }]

# --- Main Chat Interface ---
st.subheader(f"Chatting with your {persona_choice}")

# Display chat messages from history on app rerun
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like assistance with today?"):
    # Add user message to chat history view
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate agent response with spinner
    with st.chat_message("assistant"):
        with st.spinner(f"Agent ({persona_choice}) is working..."):
            try:
                response_content = run_agent_turn(
                    user_id=user_id,
                    persona_name=persona_choice,
                    user_message=prompt
                )
                st.markdown(response_content)
                # Append assistant response to history view
                st.session_state["messages"].append({"role": "assistant", "content": response_content})
            except Exception as e:
                st.error(f"Error running agent: {e}")