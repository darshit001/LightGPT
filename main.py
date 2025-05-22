import streamlit as st
import asyncio
import traceback
import json
import logging
import uuid
from datetime import datetime
from mcp import ClientSession
from mcp.client.sse import sse_client
from groq import Groq
import os
import psycopg2 
from psycopg2 import sql  
from dotenv import load_dotenv 
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import tempfile 
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io


load_dotenv()

 
SERVER_URL = os.getenv("SERVER_URL")  # Default if not found
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")  
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "chatbot_db") 
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "darshit")


logging.basicConfig(
    filename='mcp_interactions.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# PostgreSQL Database functions
def get_db_connection():
    """Create and return a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except psycopg2.Error as e:
        logging.error(f"Database connection error: {str(e)}")
        return None 

def init_database():
    """Initialize database with required tables if they don't exist."""
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        with conn.cursor() as cur:
            # Create a table for chat sessions
            cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id VARCHAR(36) PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)    
            
            # Create a table for chat interactions
            cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_interactions (
                interaction_id SERIAL PRIMARY KEY,
                session_id VARCHAR(36) REFERENCES chat_sessions(session_id),
                user_question TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                tool_used VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Drop the old table if it exists (be careful with this in production!)
            cur.execute("""
            DROP TABLE IF EXISTS chat_messages CASCADE
            """)
            
            conn.commit()
            return True
    except psycopg2.Error as e:
        logging.error(f"Database initialization error: {str(e)}")
        return False
    finally:
        conn.close()

def save_chat_interaction(session_id, user_question, assistant_response=None, tool_used=None):
    """
    Save a chat interaction to the database.
    
    Args:
        session_id (str): The session ID
        user_question (str): The user's question
        assistant_response (str, optional): The assistant's response. If None, this is 
            stored as a placeholder to be updated later.
        tool_used (str, optional): The name of the tool used to generate the response
            
    Returns:
        bool: True if successful, False otherwise
    """
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        # If assistant_response is None, we're just recording the user question
        # and will update with the response later
        response_text = assistant_response or "Processing..."
        
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_interactions (session_id, user_question, assistant_response, tool_used)
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, user_question, response_text, tool_used)
            )
            conn.commit()
            return True
    except psycopg2.Error as e:
        logging.error(f"Error saving interaction: {str(e)}")
        return False
    finally:
        conn.close()

def create_chat_session():
    """Create a new chat session and return its ID."""
    session_id = str(uuid.uuid4())
    conn = get_db_connection()
    if conn is None:
        return None
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_sessions (session_id) VALUES (%s)",
                (session_id,)
            )
            conn.commit()
            return session_id
    except psycopg2.Error as e:
        logging.error(f"Error creating chat session: {str(e)}")
        return None
    finally:
        conn.close()

def get_chat_sessions():
    """
    Retrieve all chat sessions from the database, ordered by most recent first.
    
    Returns:
        list: List of tuples containing (session_id, creation_timestamp)
    """
    conn = get_db_connection()
    if conn is None:
        return []
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT session_id, created_at 
                FROM chat_sessions 
                ORDER BY created_at DESC
                """
            )
            return cur.fetchall()
    except psycopg2.Error as e:
        logging.error(f"Error retrieving chat sessions: {str(e)}")
        return []
    finally:
        conn.close()

def get_chat_interactions(session_id):
    """
    Retrieve all interactions for a specific chat session.
    
    Args:
        session_id (str): The session ID to fetch interactions for
        
    Returns:
        list: List of tuples containing (user_question, assistant_response, tool_used, timestamp)
    """
    conn = get_db_connection()
    if conn is None:
        return []
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_question, assistant_response, tool_used, created_at 
                FROM chat_interactions 
                WHERE session_id = %s 
                ORDER BY created_at
                """,
                (session_id,)
            )
            return cur.fetchall()
    except psycopg2.Error as e:
        logging.error(f"Error retrieving chat interactions: {str(e)}")
        return []
    finally:
        conn.close()

def delete_chat_session(session_id):
    """Delete a chat session and all its interactions from the database."""
    conn = get_db_connection()
    if conn is None:
        logging.error("Failed to get database connection for deleting chat session.")
        return False
    
    try:
        with conn.cursor() as cur:
            # First delete all interactions associated with this session
            cur.execute(
                "DELETE FROM chat_interactions WHERE session_id = %s",
                (session_id,)
            )
            logging.info(f"Deleted interactions for session {session_id}")
            
            # Then delete the session itself
            cur.execute(
                "DELETE FROM chat_sessions WHERE session_id = %s",
                (session_id,)
            )
            logging.info(f"Deleted session {session_id}")
            
            conn.commit()
            return True
    except psycopg2.Error as e:
        logging.error(f"Error deleting chat session: {str(e)}")
        conn.rollback()  # Rollback to maintain database integrity
        return False
    except Exception as e:
        logging.error(f"Unexpected error deleting chat session: {str(e)}")
        conn.rollback()  # Rollback to maintain database integrity
        return False
    finally:
        conn.close()

st.markdown("""
<style>
    .stApp {
        background-color: none !important;
        color: #e0e0e0;
        font-family: 'Söhne', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: none !important;
        gap: 2rem;
        padding: 0 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8e8ea0;
        font-weight: 500;
        background-color: transparent !important;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2a2a2a !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #10a37f;
        background-color: #2a2a2a !important;
        border-bottom: none !important;
    }
    .stChatInput {
        position: fixed;
        bottom: 1rem;
        left: 0;
        right: 0;
        max-width: 800px;
        margin: 0 auto;
        padding: 0.5rem;
        z-index: 1000;
    }
    .stChatInput > div > input {
        color: #e0e0e0 !important;
        background-color: transparent !important;
    }
    .chat-container {
        padding-bottom: 120px !important;
        max-width: 800px;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }
    .stChatMessage {
        max-width: 100%;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1rem;
        border-radius: 0.8rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        clear: both;
        overflow: auto;
        word-wrap: break-word;
    }    .stChatMessage[data-testid="stChatMessage-User"] {
        background-color: #333333;
        color: #ffffff;
        margin-left: auto;
        float: right;
        display: flex;
        justify-content: flex-end;
    }
    .stChatMessage[data-testid="stChatMessage-Assistant"] {
        background-color: #252525;
        color: #e0e0e0;
        margin-right: auto;
        float: left;
        display: flex;
        justify-content: flex-start;
    }
    .block-container {
        padding-bottom: 120px !important;
    }
    .main .block-container {
        padding-bottom: 120px !important;
    }
    h1, h2, h3 {
        color: #e0e0e0;
        margin-bottom: 0rem !important;
    }
    h2 {
        margin-bottom: 0.3rem !important;
    }
    .stTextInput > label {
        color: #e0e0e0;
    }
    .stButton > button {
        background-color: #2a2a2a;
        color: #8e8ea0;
        border: 1px solid #3a3a3a;
        border-radius: 0.4rem;
    }
    .stButton > button:hover {
        background-color: #3a3a3a;
        color: #10a37f;
    }
    /* Style for delete buttons */
    button[data-baseweb="button"][key^="delete_"] {
        background-color: transparent;
        border: none;
        color: #8e8ea0;
        min-width: 30px;
        padding: 0;
    }
    button[data-baseweb="button"][key^="delete_"]:hover {
        color: #ff5252;
        background-color: transparent;
    }
    /* Style for chat session buttons */
    button[data-baseweb="button"][key^="session_"] {
        text-align: left;
        height: auto;
        padding: 8px 10px;
        white-space: normal;
        line-height: 1.2;
    }
    /* Style for active chat session - current session */
    .active-chat {
        background-color: #1e3f34 !important;
        border-left: 3px solid #10a37f;
        font-weight: bold;
    }
    .active-chat:hover {
        background-color: #264940 !important;
    }
    /* Style for current chat indicator */
    .current-chat-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #10a37f;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    /* Active chat container styles */
    .active-chat-container {
        background-color: #1e3f34;
        border-left: 3px solid #10a37f;
        border-radius: 4px;
        padding: 8px 10px;
        margin-bottom: 10px;
    }
    /* Chat history row style */
    [data-testid="column"] {
        padding: 0;
        margin-bottom: 5px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .fixed-input-bar-true {
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        width: 100vw;
        background: #232323;
        padding: 1rem 0.5rem 0.7rem 0.5rem;
        z-index: 10000;
        box-shadow: 0 -2px 12px rgba(0,0,0,0.15);
        display: flex;
        justify-content: center;
    }
    .fixed-input-bar-true .block-container {
        padding-bottom: 120px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the database
init_database()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'tools' not in st.session_state:
    st.session_state.tools = []
if 'tool_used' not in st.session_state:
    st.session_state.tool_used = None
if 'image_paths' not in st.session_state:
    st.session_state.image_paths = []  # Store multiple images
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = create_chat_session()

def llm_client(message: str):
    """
    Send a message to the LLM and return the response, with conversation memory.
    """
    memory_messages = st.session_state.memory.chat_memory.messages
    message_history = [{"role": "system", "content": "You are an intelligent assistant. You will execute tasks as prompted"}]
    
    for msg in memory_messages:
        if isinstance(msg, HumanMessage):
            message_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            message_history.append({"role": "assistant", "content": msg.content})
    
    message_history.append({"role": "user", "content": message})
    
    groq_client = Groq(api_key=GROQ_API_KEY)
    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=message_history,
        max_tokens=250,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def format_tool_response(query: str, raw_response: str):
    """
    Use LLM to reformat the raw tool response for clarity and proper formatting.
    """
    prompt = (
        "You are an assistant tasked with reformatting a tool's response to make it clear, concise, and well-structured. "
        "Ensure the response directly answers the user's question, uses proper grammar, and is formatted in a professional manner. "
        "Avoid adding unnecessary details or altering the factual content unless it improves clarity. "
        "If the raw response is empty or irrelevant, provide a polite fallback message. "
        f"User's Question: {query}\n"
        f"Raw Tool Response: {raw_response}\n"
        "Reformatted Response:"
    )
    return llm_client(prompt)

def get_prompt_to_identify_tool_and_arguments(query, tools, pdf_path=None):
    tools_description = "\n".join([f"- {tool.name}, {tool.description}, {tool.inputSchema} " for tool in tools])
    pdf_instruction = f"\nIf a PDF is uploaded (path: {pdf_path}), use the pdf_qa tool for questions related to the PDF content." if pdf_path else ""
    return (
        """You are a helpful assistant with access to these tools. Your task is to choose the most appropriate tool based on the user's question.

IMPORTANT GUIDELINES:
1. For general questions, learning paths, explanations, or discussions, use the general_qa tool
2. For specific code implementation requests, use the generate_code tool
3. For mathematical calculations, use the math_solver tool
4. For web searches, use the tavily_search tool
5. For casual conversation, use the chat_with_assistant tool
6. For creating prompts, use the generate_prompt tool
7. For generating images, use the generate_image tool
8. For questions about a PDF's content, use the pdf_qa tool"""
        f"{pdf_instruction}\n"
        f"Available tools:\n{tools_description}\n"
        f"User's Question: {query}\n"
        "Choose the most appropriate tool based on the guidelines above.\n"
        "If no tool is needed, reply directly.\n\n"
        "IMPORTANT: When you need to use a tool, you must ONLY respond with "
        "the exact JSON object format below, nothing else:\n"
        "Keep the values in str "
        "{\n"
        '    "tool": "tool-name",\n'
        '    "arguments": {\n'
        '        "argument-name": "value"\n'
        "    }\n"
        "}\n\n"
    )

def extract_image_path(result_text):
    """
    Extract the image path from the result text.
    """
    if "Saved as:" in result_text:
        return result_text.split("Saved as:")[1].strip()
    return None

async def run_query(server_url: str, query: str, pdf_path=None):
    """
    Process the query using SSE connection and return the result.
    """
    logging.info(f"User Query: {query}")
    try:
        async with sse_client(server_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                tools = await session.list_tools()
                st.session_state.tools = tools.tools
                
                prompt = get_prompt_to_identify_tool_and_arguments(query, tools.tools, pdf_path)
                llm_response = llm_client(prompt)
                logging.info(f"LLM response received: {llm_response}")
                
                tool_call = json.loads(llm_response)
                logging.info(f"Tool call parsed: {tool_call}")
                
                result = await session.call_tool(tool_call["tool"], arguments=tool_call["arguments"])
                
                if not result.content:
                    return "No results found. Please try a different query.", tool_call["tool"]
                
                response_text = result.content[0].text if result.content else "No content available"
                logging.info(f"Tool Used: {tool_call['tool']}")
                logging.info(f"Response: {response_text}\n")
                
                return response_text, tool_call["tool"]
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        return f"An error occurred. Please try again. Error: {str(e)}", None

def display_message(message, is_user=False, image_path=None):
    """
    Display a message in the chat interface with appropriate styling.
    If image_path is provided, also display the image after the message.
    """
    avatar = "👤" if is_user else "🤖"
    message_class = "user-message" if is_user else "assistant-message"
    
    message_html = f"""
        <div class="stChatMessage" data-testid="stChatMessage-{'User' if is_user else 'Assistant'}">
            <div class="avatar">{avatar}</div>
            <div class="message-content">{message}</div>
        </div>
    """
    st.markdown(message_html, unsafe_allow_html=True)
    
    # If an image path is provided and this is an assistant message, display the image
    if image_path and not is_user and os.path.exists(image_path):
        with st.container():
            st.image(image_path, caption="Generated Image", width=400)

def export_chat_to_pdf(messages):
    """
    Export chat messages to a PDF file.
    
    Args:
        messages (list): List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        bytes: PDF file content as bytes
    """
    # Create a BytesIO buffer to store the PDF
    buffer = io.BytesIO()
    
    # Create the PDF object with the BytesIO buffer
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Configure text settings
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "MCP Assistant Chat History")
    
    # Add current date
    c.setFont("Helvetica", 10)
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(72, height - 95, f"Exported on: {current_date}")
    
    # Add a line below the header
    c.line(72, height - 110, width - 72, height - 110)
    
    # Start y position for chat messages
    y_position = height - 140
    
    # Add chat messages
    c.setFont("Helvetica-Bold", 11)
    
    for message in messages:
        role = "User" if message['role'] == 'user' else "Assistant"
        content = message['content']
        timestamp = message.get('timestamp', '')
        
        # Draw role and timestamp
        c.setFillColorRGB(0.2, 0.2, 0.8) if role == "User" else c.setFillColorRGB(0.2, 0.6, 0.2)
        c.drawString(72, y_position, f"{role} - {timestamp}")
        y_position -= 20
        
        # Draw message content
        c.setFillColorRGB(0, 0, 0)  # Black text for content
        c.setFont("Helvetica", 10)
        
        # Process content in chunks to handle line breaks
        content_lines = content.split("\n")
        for line in content_lines:
            # Wrap text to fit page width
            text_width = width - 144  # 72 points margin on each side
            chars_per_line = int((text_width / 7) * 1.8)  # Approximate chars that fit
            
            # Simple text wrapping
            while len(line) > chars_per_line:
                # Find the last space before the character limit
                space_pos = line[:chars_per_line].rfind(' ')
                if space_pos == -1:  # No space found, just cut at char limit
                    space_pos = chars_per_line
                
                c.drawString(72, y_position, line[:space_pos])
                line = line[space_pos:].lstrip()
                y_position -= 14
                
                # Check if we need a new page
                if y_position < 72:
                    c.showPage()
                    y_position = height - 72
                    c.setFont("Helvetica", 10)
            
            # Draw remaining text
            if line:
                c.drawString(72, y_position, line)
                y_position -= 14
            
            # Check if we need a new page
            if y_position < 72:
                c.showPage()
                y_position = height - 72
                c.setFont("Helvetica", 10)
        
        # Add spacing between messages
        y_position -= 10
        
        # Add a separator line between messages
        c.setStrokeColorRGB(0.8, 0.8, 0.8)
        c.line(72, y_position, width - 72, y_position)
        
        y_position -= 20
        
        # Check if we need a new page
        if y_position < 72:
            c.showPage()
            y_position = height - 72
            c.setFont("Helvetica", 10)
    
    # Save the PDF to the buffer
    c.save()
    
    # Get the PDF content from the buffer
    buffer.seek(0)
    return buffer.getvalue()

def format_timestamp(timestamp):
    """Format database timestamp to a more readable format."""
    if isinstance(timestamp, str):
        # Parse string timestamp (if it comes that way from DB)
        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            try:
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return timestamp
    else:
        dt = timestamp
    
    # Format as "Jan 15, 2023, 3:45 PM"
    return dt.strftime("%b %d, %Y, %I:%M %p")


def load_chat_history(session_id):
    """Load chat history for a specific session from the database and update session state."""
    interactions = get_chat_interactions(session_id)
    
    # Clear current messages
    st.session_state.messages = []
    
    # Initialize memory
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    
    # Clear image paths when switching sessions
    st.session_state.image_paths = []
    
    # Add each interaction to the messages and memory
    for user_question, assistant_response, tool_used, timestamp in interactions:
        st.session_state.messages.append({'role': 'user', 'content': user_question})
        st.session_state.messages.append({'role': 'assistant', 'content': assistant_response})
        
        # Also update the memory
        st.session_state.memory.chat_memory.add_user_message(user_question)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)
        
        # If this interaction involved image generation, try to recover the image path
        if tool_used == "generate_image":
            image_path = extract_image_path(assistant_response)
            if image_path and os.path.exists(image_path):
                st.session_state.image_paths.append(image_path)
    
    # Set the current session ID
    st.session_state.session_id = session_id


def get_session_preview(session_id):
    """Get a preview of the first message in a chat session for display in the sidebar."""
    interactions = get_chat_interactions(session_id)
    
    if interactions:
        first_question = interactions[0][0]  # first message's user question
        # Truncate if too long
        if len(first_question) > 40:
            return first_question[:37] + "..."
        return first_question
    
    return "Empty chat"


def main():
    # Initialize the database if it doesn't exist
    init_database()
    
    # Initialize session state for chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'session_id' not in st.session_state:
        # Create a new chat session
        session_id = create_chat_session()
        if session_id:
            st.session_state.session_id = session_id
        else:
            st.error("Failed to create a chat session. Please try again.")
            return
    
    # Initialize conversation memory
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    
    # Initialize PDF path
    if 'pdf_path' not in st.session_state:
        st.session_state.pdf_path = None
    
    # Initialize image path
    if 'image_path' not in st.session_state:
        st.session_state.image_path = None
    
        st.title("MCP Assistant")
    # st.write("Chat with the MCP Assistant powered by various tools.")
    
    # Chat history
    chat_container = st.container()
    with chat_container:
        # Display all previous messages with images where applicable
        for i, message in enumerate(st.session_state.messages):
            is_user = message['role'] == 'user'
            
            # Check if this is an assistant message about image generation
            if not is_user and "Tool used: generate_image" in message['content'] and i > 0:
                # Find the associated image path from stored image paths
                # This assumes the order of st.session_state.image_paths matches the order of image messages
                image_index = sum(1 for m in st.session_state.messages[:i] 
                                    if m['role'] == 'assistant' and "Tool used: generate_image" in m['content'])
                
                # If we have an image for this message, display it with the message
                if image_index < len(st.session_state.image_paths):
                    display_message(message['content'], is_user, st.session_state.image_paths[image_index])
                else:
                    display_message(message['content'], is_user)
            else:
                display_message(message['content'], is_user)
    
    server_url = SERVER_URL  # Use the environment variable

    with st.sidebar:
        st.header("Chat Sessions")
          # New Chat button
        if st.button("➕  New Chat"):
            # Create a new chat session
            new_session_id = create_chat_session()
            if new_session_id:
                # Clear messages and set new session ID
                st.session_state.messages = []
                st.session_state.session_id = new_session_id
                st.session_state.memory = ConversationBufferMemory(return_messages=True)
                st.session_state.pdf_path = None
                st.session_state.image_paths = []  # Clear image paths for new chat
                st.rerun()  # Refresh the page
            else:
                st.error("Failed to create a new chat session. Please try again.")
        st.divider()
        st.subheader("Chat History")
        chat_sessions = get_chat_sessions()
        
        # Create a container for session management
        session_container = st.container()
        
        # Set up a session state to track which session to delete
        if 'delete_session_id' not in st.session_state:
            st.session_state.delete_session_id = None
            
        # Process any pending deletions
        if st.session_state.delete_session_id:
            session_to_delete = st.session_state.delete_session_id
            if delete_chat_session(session_to_delete):
                st.success(f"Chat deleted successfully!")
                # Reset the delete session ID
                st.session_state.delete_session_id = None
                # Rerun to refresh the list
                st.rerun()
            else:
                st.error("Failed to delete chat. Please try again.")
                st.session_state.delete_session_id = None
        
        # Display all chat sessions with current one highlighted
        with session_container:
            for session_id, created_at in chat_sessions:
                # Format the timestamp
                formatted_time = format_timestamp(created_at)
                
                # Get a preview of the first message
                preview = get_session_preview(session_id)
                
                # Create a container for each chat session with a button and delete icon
                chat_row = st.container()
                
                # Check if this is the current active session
                is_active = session_id == st.session_state.session_id
                
                with chat_row:
                    col1, col2 = st.columns([5, 1])
                    
                    with col1:
                        if is_active:
                            # Display current chat with special styling
                            active_chat_label = f"<div class='active-chat-container'><span class='current-chat-indicator'></span><strong>{preview}</strong><br><small>{formatted_time}</small> <span style='background-color: #1e3f34; color: #10a37f; padding: 2px 6px; border-radius: 10px; font-size: 10px;'>ACTIVE</span></div>"
                            st.markdown(active_chat_label, unsafe_allow_html=True)
                        else:
                            # Create a button for other sessions with the preview and time
                            session_button_label = f"{preview}\n{formatted_time}"
                            if st.button(session_button_label, key=f"session_{session_id}", use_container_width=True):
                                # Load selected chat history
                                load_chat_history(session_id)
                                st.rerun()  # Refresh the page
                      # Add delete button with trash icon
                    with col2:
                        if is_active:
                            st.markdown("<div style='height: 38px; display: flex; align-items: center;'>", unsafe_allow_html=True)
                            if st.button("🗑️", key=f"delete_active_{session_id}", help="Delete this chat"):
                                # For active chat, create new session before deleting
                                new_session_id = create_chat_session()
                                if new_session_id:
                                    st.session_state.session_id = new_session_id
                                    st.session_state.messages = []
                                    st.session_state.memory = ConversationBufferMemory(return_messages=True)
                                    st.session_state.image_paths = []  # Clear image paths
                                st.session_state.delete_session_id = session_id
                                st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            if st.button("🗑️", key=f"delete_{session_id}", help="Delete this chat"):
                                # Set the session ID to delete
                                st.session_state.delete_session_id = session_id
                                st.rerun()
        
        st.divider()
        
        st.header("Tool Controls")
        
        selected_tool = st.radio(
            "Select Tool",
            options=["Default", "Deep Research", "Image Generation", "PDF QA"],
            help="Choose the tool you want to use"
        )
        
        # Deep Research Controls
        if selected_tool == "Deep Research":
            research_depth = st.slider(
                "Research Depth",
                min_value=1,
                max_value=10,
                value=5,
                help="Higher depth provides more comprehensive results but takes longer"
            )
        
        # PDF QA Controls
        if selected_tool == "PDF QA":
            st.subheader("Upload PDF")
            uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=False, help="Upload a PDF to ask questions about its content")
            if uploaded_pdf is not None:
                uploaded_dir="./uploaded_pdfs"
                os.makedirs(uploaded_dir, exist_ok=True)
                pdf_path = os.path.join(uploaded_dir, uploaded_pdf.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_pdf.read())
                st.session_state.pdf_path = pdf_path
                st.success("PDF uploaded successfully!")
            else:
                st.session_state.pdf_path = None
                st.warning("No PDF uploaded. Please upload a PDF to use the PDF QA tool.")    # Add a 'Clear Chat' button in the sidebar
    with st.sidebar:
        if st.button("🗑️ Clear Current Chat"):
            # Clear chat history for current session only
            st.session_state.messages = []
            st.session_state.memory = ConversationBufferMemory(return_messages=True)
            st.session_state.image_paths = []  # Clear image paths

            # Clear uploaded PDFs and database directories
            uploaded_dir = "./uploaded_pdfs"
            database_dir = "./database"
            image_path="./image"            
            for folder in [uploaded_dir, database_dir, image_path]:
                if os.path.exists(folder):
                    for file in os.listdir(folder):
                        file_path = os.path.join(folder, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            import shutil
                            shutil.rmtree(file_path)

            st.success("Chat history and stored files have been cleared.")    # Add Export Chat Button in Sidebar
    with st.sidebar:
        if st.button("📄 Export Chat as PDF"):
            if st.session_state.messages:
                # Add timestamps to messages if not already present
                for message in st.session_state.messages:
                    if "timestamp" not in message:
                        message["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Generate PDF
                pdf_content = export_chat_to_pdf(st.session_state.messages)

                # Provide download link
                st.download_button(
                    label="Download Chat History",
                    data=pdf_content,
                    file_name="chat_history.pdf",
                    mime="application/pdf",
                )
            else:
                st.warning("No chat history to export.")
                
        # Add a button to show all generated images
        if st.session_state.image_paths:
            if st.button("🖼️ Show All Generated Images"):
                show_all_images = True
                with st.sidebar.expander("Generated Images", expanded=True):
                    # Use columns to display images in a grid
                    num_images = len(st.session_state.image_paths)
                    cols_per_row = 1  # Number of columns in the sidebar grid
                    
                    # Display images in rows
                    for i in range(0, num_images, cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            if i + j < num_images:
                                with cols[j]:
                                    img_path = st.session_state.image_paths[i + j]
                                    st.image(
                                        img_path,
                                        caption=f"Image {i + j + 1}",
                                        use_column_width=True
                                    )# Chat input at the bottom
    if prompt := st.chat_input("Type your message here..."):        # Add user message to chat history
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        # We'll save the full interaction after getting the response
        user_question = prompt
        with chat_container:
            display_message(prompt, is_user=True)
        with st.spinner("Processing your query..."):
            if selected_tool == "Deep Research":
                # result, tool_used = asyncio.run(run_query(server_url, prompt))
                # if tool_used != "deep_research":
                async def force_deep_research(query):
                    async with sse_client(server_url) as streams:
                        async with ClientSession(streams[0], streams[1]) as session:
                            await session.initialize()
                            result = await session.call_tool(
                                    "deep_research", 
                                    arguments={
                                        "query": query,
                                        "depth": research_depth
                                    }
                                )
                            return result.content[0].text, "deep_research"
                result, tool_used = asyncio.run(force_deep_research(prompt))
            elif selected_tool == "Image Generation":
                async def generate_image_with_prompt(query):
                    async with sse_client(server_url) as streams:
                        async with ClientSession(streams[0], streams[1]) as session:
                            await session.initialize()
                            result = await session.call_tool(
                                "generate_image",
                                arguments={
                                    "prompt": query
                                }
                            )
                            return result.content[0].text, "generate_image"
                result, tool_used = asyncio.run(generate_image_with_prompt(prompt))
            elif selected_tool == "PDF QA" and st.session_state.pdf_path:
                # Use pdf_qa tool if a PDF is uploaded and PDF QA is selected 
                async def query_pdf(query, pdf_path):
                    async with sse_client(server_url) as streams:
                        async with ClientSession(streams[0], streams[1]) as session:
                            await session.initialize()
                            result = await session.call_tool(
                                "pdf_qa",
                                arguments={
                                    "query": query,
                                    "pdf_path": pdf_path
                                }
                            )
                            return result.content[0].text, "pdf_qa"
                result, tool_used = asyncio.run(query_pdf(prompt, st.session_state.pdf_path))
            else:
                # Use regular query processing, considering PDF if uploaded
                result, tool_used = asyncio.run(run_query(server_url, prompt, st.session_state.pdf_path))
            if tool_used == "deep_research":
                # Don't format Deep Research responses
                formatted_result = result
            else:
                # Format responses from other tools
                formatted_result = format_tool_response(prompt, result)
            
            if tool_used == "generate_image":
                image_path = extract_image_path(result)
                if image_path and os.path.exists(image_path):
                    # Add the new image to the list of images
                    st.session_state.image_paths.append(image_path)         
                    
            assistant_message = f"Tool used: {tool_used}\n\n{formatted_result}"
            st.session_state.messages.append({'role': 'assistant', 'content': assistant_message})
            
            # Save assistant message to database with tool information
            save_chat_interaction(st.session_state.session_id, user_question, formatted_result, tool_used)
            
            st.session_state.memory.chat_memory.add_user_message(prompt)
            st.session_state.memory.chat_memory.add_ai_message(formatted_result)
            
            with chat_container:
                # If it's an image generation, display the message with the image
                if tool_used == "generate_image" and image_path and os.path.exists(image_path):
                    display_message(assistant_message, is_user=False, image_path=image_path)
                else:
                    display_message(assistant_message, is_user=False)

if __name__ == "__main__":
    main()
