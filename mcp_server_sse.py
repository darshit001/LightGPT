import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route, Mount
from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INTERNAL_ERROR, INVALID_PARAMS
from mcp.server.sse import SseServerTransport
from groq import Groq
import math
import operator
import requests
from tavily import TavilyClient
import urllib.parse
import os
from datetime import datetime
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
FIRE_CRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Create an MCP server instance
mcp = FastMCP("MCP Assistant")

@mcp.tool()
def math_solver(expression: str) -> str:
    """
    Solve mathematical expressions safely with support for various mathematical operations.

    Args:
        expression (str): A string containing the mathematical expression to solve.
            Examples: "2 + 2", "sin(pi/2)", "sqrt(16)"

    Returns:
        str: A string containing either the result of the calculation or an error message.
    """
    safe_dict = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        'math': math,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'sqrt': math.sqrt,
        'log': math.log,
        'log10': math.log10,
        'pi': math.pi,
        'e': math.e,
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        '**': operator.pow,
        '%': operator.mod,
    }
    
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"The result of {expression} is {result}"
    except Exception as e:
        raise McpError(ErrorData(INVALID_PARAMS, f"Error evaluating expression: {str(e)}"))

@mcp.tool()
def generate_code(code_request: str, language: str = "python") -> str:
    """
    Generate code based on natural language description using LLM capabilities.

    Args:
        code_request (str): A natural language description of the code to generate.
        language (str, optional): The programming language to generate code in.

    Returns:
        str: The generated code as a string with explanations.
    """
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        prompt = f"""You are an expert programmer. Generate {language} code based on the following request:
        {code_request}
        
        Requirements:
        1. First provide a detailed explanation of the solution approach
        2. Then provide the complete code with necessary imports
        3. After the code, explain how the code works step by step
        4. Include any important notes or best practices
        5. Follow best practices for {language}
        
        Format your response as follows:
        
        EXPLANATION:
        [Your detailed explanation of the approach]
        
        CODE:
        [Your complete code with imports]
        
        HOW IT WORKS:
        [Step by step explanation of the code]
        
        NOTES:
        [Any additional notes or best practices]
        """
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"You are an expert {language} programmer. Generate clean, efficient, and well-documented code with detailed explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Error generating code: {str(e)}"))

@mcp.tool()
def tavily_search(query: str) -> str:
    """
    Perform web searches using the Tavily API and format results using LLM.

    Args:
        query (str): The search query to look up on the web.

    Returns:
        str: A well-formatted summary of the search results.
    """
    try:
        client = TavilyClient(TAVILY_KEY)
        response = client.search(
            query=query,
            max_results=10,
            search_depth="advanced",
        )

        if not response:
            return "No results found."

        search_results = response["results"]
        results_text = "Search Results:\n\n"
        for idx, result in enumerate(search_results, 1):
            results_text += f"Result {idx}:\n"
            results_text += f"Title: {result.get('title', 'No title')}\n"
            results_text += f"Content: {result.get('content', 'No content')}\n"

        groq_client = Groq(api_key=GROQ_API_KEY)
        
        llm_prompt = f"""Please analyze and format these search results about '{query}' in a clear, organized way.
        Include:
        1. A brief summary of the main findings
        2. Key points or insights
        3. Important details or context
        
        Here are the raw search results:
        {results_text}
        
        Please format this information in a clear, readable way."""

        llm_response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant that formats and summarizes search results in a clear, organized way."},
                {"role": "user", "content": llm_prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )

        return llm_response.choices[0].message.content.strip()
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Error during search: {str(e)}"))

@mcp.tool()
def chat_with_assistant(message: str) -> str:
    """
    Engage in conversational interactions with an AI assistant.

    Args:
        message (str): The user's message or question to respond to.

    Returns:
        str: A conversational response from the AI assistant.
    """
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a friendly, helpful AI assistant."},
                {"role": "user", "content": message}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Error in chat: {str(e)}"))

@mcp.tool()
def generate_prompt(topic: str, purpose: str = "general") -> str:
    """
    Generate creative and detailed prompts for various purposes.

    Args:
        topic (str): The main subject or theme for the prompt.
        purpose (str, optional): The intended use for the prompt.

    Returns:
        str: A creative, detailed prompt.
    """
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        full_prompt = (
            f"Write a creative, detailed prompt for the following purpose: {purpose}. "
            f"The topic is: '{topic}'. "
            "Make the prompt clear, inspiring, and suitable for the intended use."
        )
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are my prompt expert. You write the best prompts for any purpose."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=120,
            temperature=0.85
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Error generating prompt: {str(e)}"))

@mcp.tool()
def generate_image(prompt: str) -> str:
    """
    Generate images based on text prompts using the Pollinations AI API.

    Args:
        prompt (str): A text description of the image to generate.

    Returns:
        str: The path to the generated image file.
    """
    try:
        # Create image directory if it doesn't exist
        os.makedirs("image", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image/generated_image_{timestamp}.jpg"
        
        encoded_prompt = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        
        response = requests.get(url)
        response.raise_for_status()
        
        with open(filename, "wb") as f:
            f.write(response.content)
        
        abs_path = os.path.abspath(filename)
        return f"Image generated successfully! Saved as: {abs_path}"
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Error generating image: {str(e)}"))

@mcp.tool()
def general_qa(question: str) -> str:
    """
    Answer general questions and provide information on various topics using the Groq LLM.

    Args:
        question (str): The user's question or topic they want to learn about.

    Returns:
        str: A detailed, informative response to the user's question.
    """
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        prompt = f"""You are a knowledgeable and helpful AI assistant. Please provide a detailed, accurate, 
        and informative response to the following question. If you're not sure about something, be honest about it.
        If the question requires specialized tools or capabilities, suggest which tools might be more appropriate.
        
        User's question: {question}
        
        Please structure your response in a clear, organized way. Include:
        1. A direct answer to the question
        2. Relevant details and explanations
        3. Examples or analogies if helpful
        4. Additional context or related information if relevant
        
        Keep the response informative but concise."""
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a knowledgeable and helpful AI assistant that provides accurate, detailed, and well-structured responses to general questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Error in general QA: {str(e)}"))

@mcp.tool()
def deep_research(query: str) -> str:
    """
    Perform deep research on a given query using FirecrawlApp and return a summary and source count.

    Args:
        query (str): The research question or topic.

    Returns:
        str: The final analysis and number of sources found.
    """
    try:
        firecrawl = FirecrawlApp(api_key=FIRE_CRAWL_API_KEY)
        results = firecrawl.deep_research(
            query=query,
            max_depth=10,
            time_limit=180,
            max_urls=15
        )
        final_analysis = results['data']['finalAnalysis']
        num_sources = len(results['data']['sources'])
        return f"Final Analysis: {final_analysis}\n\nSources: {num_sources} references"
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Error in deep research: {str(e)}"))

@mcp.tool()
def pdf_qa(query: str, pdf_path: str) -> str:
    """
    Answer questions about the content of a specific PDF file using LlamaIndex.

    Args:
        query (str): The question about the PDF content.
        pdf_path (str): The file path to the PDF document.

    Returns:
        str: Answer to the question based on the PDF content.
    """
    try:
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document, StorageContext, load_index_from_storage
        from llama_index.llms.groq import Groq
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        import os
        
        # Set up Groq LLM and embeddings
        groq_api_key = GROQ_API_KEY
        llm = Groq(api_key=groq_api_key, model="llama3-70b-8192")

        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        Settings.embed_model = embed_model
        Settings.llm = llm
        Settings.chunk_size = 512
        
        pdf_filename = os.path.basename(pdf_path).replace(" ", "_").replace(".", "_")
        persist_dir = f"./database/pdf_{pdf_filename}"
        
        # Check if index already exists for this PDF
        if not os.path.exists(persist_dir):
            # If index doesn't exist, create it
            print(f"Creating new index for {pdf_path}...")
            
            # Load PDF
            if os.path.isfile(pdf_path):
                documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
            else:
                return f"Error: PDF file not found at {pdf_path}"
                
            if not documents:
                return "Error: No content extracted from the PDF"
                
            # Create index
            index = VectorStoreIndex.from_documents(documents)
            
            # Save index to disk
            os.makedirs(persist_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=persist_dir)
        else:
            print(f"Loading existing index for {pdf_path}...")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
        
        # Create query engine
        query_engine = index.as_query_engine()
        
        # Process query
        response = query_engine.query(query)
        
        return str(response)
    
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Error in PDF QA: {str(e)}"))

# Set up the SSE transport for MCP communication
sse = SseServerTransport("/messages/")

async def handle_sse(request: Request) -> None:
    _server = mcp._mcp_server
    async with sse.connect_sse(
        request.scope,
        request.receive,
        
        request._send,
    ) as (reader, writer):
        await _server.run(reader, writer, _server.create_initialization_options())

# Create the Starlette app with SSE endpoints
app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ],
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000) 
    