import os
import json
import re
import traceback
from typing import TypedDict, List, Dict, Any, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, END

from utils import load_api_keys
from pdf_processor import chunk_text

# --- API Key Loading ---
# Keep this section BUT maybe don't set os.environ here globally
# Instead, check os.environ directly when initializing components
try:
    GOOGLE_API_KEY, TAVILY_API_KEY, _ = load_api_keys()
    # Optional: Log what keys were found initially from .env
    print(f"Initial keys from .env/system: Google {'Found' if GOOGLE_API_KEY else 'Missing'}, Tavily {'Found' if TAVILY_API_KEY else 'Missing'}") # Message unchanged
except ValueError as e:
    print(f"Configuration Error loading initial keys: {e}") # Message unchanged
    GOOGLE_API_KEY = None
    TAVILY_API_KEY = None

# --- Lazy Initialization Helper Functions ---
# These functions will check environment variables when called

def get_google_embeddings():
    google_key = os.environ.get("GOOGLE_API_KEY")
    if google_key:
        try:
            # Set it just for this initialization if needed by LangChain internals
            # os.environ["GOOGLE_API_KEY"] = google_key # Sometimes needed
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception as e:
            print(f"Error initializing Google Embeddings (key might be invalid): {e}") # Message unchanged
            return None
    else:
        print("ERROR: Google API Key not found in environment for Embeddings.") # Message unchanged
        return None

def get_google_llm():
    google_key = os.environ.get("GOOGLE_API_KEY")
    if google_key:
        try:
            # os.environ["GOOGLE_API_KEY"] = google_key # Sometimes needed
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5, top_p=0.9)
        except Exception as e:
            print(f"Error initializing Google LLM (key might be invalid): {e}") # Message unchanged
            return None
    else:
        print("ERROR: Google API Key not found in environment for LLM.") # Message unchanged
        return None

def get_tavily_search():
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        try:
            # os.environ["TAVILY_API_KEY"] = tavily_key # Sometimes needed
            return TavilySearchResults(max_results=3)
        except Exception as e:
            print(f"Error initializing Tavily Search (key might be invalid): {e}") # Message unchanged
            return None
    else:
        # This is not an error, just info
        print("Info: Tavily API Key not found in environment. Web search tool disabled.") # Message unchanged
        return None

# --- Graph State Definition (Keep existing) ---
class GraphState(TypedDict):
    # ... (no changes needed here) ...
    pdf_text_chunks: Optional[List[str]] = None
    vector_store: Optional[Any] = None
    retrieved_docs: Optional[List[str]] = None
    generation_query: str = "Generate examination questions (and their answers) based on the provided content." # Prompt-like string unchanged
    search_enabled: bool = False
    num_questions: int = 5
    randomize: bool = False
    web_search_results: Optional[str] = None
    generated_questions: List[Dict[str, str]] = []
    error: Optional[str] = None

# --- Graph Node Functions ---

def setup_vector_store(state: GraphState) -> GraphState:
    """
    Initializes the FAISS vector store from PDF text chunks using Google embeddings.

    Args:
        state (GraphState): The current graph state containing 'pdf_text_chunks'.

    Returns:
        GraphState: The updated state with 'vector_store' populated or 'error' set on failure.
    """
    print("--- NODE: setup_vector_store ---") # Message unchanged
    pdf_chunks = state.get("pdf_text_chunks")
    # Get embeddings model LAZILY
    embeddings_model = get_google_embeddings()

    if not pdf_chunks or not embeddings_model:
        print("Error: No PDF chunks or embedding model available (check API Key?).") # Message unchanged
        return {**state, "error": "Cannot create vector store without text chunks or embedding model (API Key may be missing/invalid)."} # Error text unchanged
    try:
        print(f"Creating FAISS index for {len(pdf_chunks)} chunks...") # Message unchanged
        vector_store = FAISS.from_texts(pdf_chunks, embeddings_model)
        print("FAISS index created successfully.") # Message unchanged
        return {**state, "vector_store": vector_store, "error": None}
    except Exception as e:
        error_message = str(e)
        print(f"Error creating FAISS vector store: {e}") # Message unchanged
        traceback.print_exc()

        # More explicit detection of API key errors
        if "API key not valid" in error_message or "API_KEY_INVALID" in error_message:
            return {**state, "error": "API_KEY_INVALID: The Google API key provided is invalid or has expired."} # Error text unchanged

        return {**state, "error": f"FAISS Error: {e}"} # Error text unchanged

def retrieve_documents(state: GraphState) -> GraphState:
    """
    Retrieves relevant document chunks from the vector store based on the generation query.

    Args:
        state (GraphState): The current graph state containing 'vector_store' and 'generation_query'.

    Returns:
        GraphState: The updated state with 'retrieved_docs' populated or 'error' set on failure.
    """
    print("--- NODE: retrieve_documents ---") # Message unchanged

    # First check if an API key error already exists
    if state.get("error") and "API_KEY_INVALID" in state.get("error"):
        # Propagate the API key error without changing it
        return state

    vector_store = state.get("vector_store")
    generation_query = state.get("generation_query", "Relevant content for exam questions") # Fallback query text unchanged
    # Ensure the vector store exists.
    if not vector_store:
         return {**state, "error": "Vector store not initialized for retrieval."} # Error text unchanged
    try:
        # Create a retriever from the vector store. 'k' specifies the number of documents to retrieve.
        retriever = vector_store.as_retriever(search_kwargs={'k': 10})
        print(f"Retrieving documents for query: '{generation_query}'") # Message unchanged
        # Invoke the retriever with the query.
        retrieved_docs_objs = retriever.invoke(generation_query)
        # Extract the page content (the actual text) from the retrieved Document objects.
        retrieved_docs = [doc.page_content for doc in retrieved_docs_objs]
        print(f"{len(retrieved_docs)} documents retrieved.") # Message unchanged
        if not retrieved_docs:
             print("Warning: No relevant documents found in the Vector Store.") # Message unchanged
        # Update state with the retrieved document texts.
        return {**state, "retrieved_docs": retrieved_docs, "error": None}
    except Exception as e:
        error_message = str(e)
        # Handle errors during retrieval.
        print(f"Error retrieving from vector store: {e}") # Message unchanged
        traceback.print_exc()

        # Check if the error is related to an invalid API key
        if "API key not valid" in error_message or "API_KEY_INVALID" in error_message:
            return {**state, "error": "API_KEY_INVALID: The Google API key provided is invalid or has expired."} # Error text unchanged

        return {**state, "error": f"Retrieval Error: {e}"} # Error text unchanged


def format_docs(docs: List[str]) -> str:
    """
    Formats a list of document strings into a single string separated by separators.

    Args:
        docs (List[str]): A list of strings, where each string is a document chunk.

    Returns:
        str: A single string concatenating all documents with separators.
    """
    return "\n\n---\n\n".join(docs)

def generate_questions_node(state: GraphState) -> GraphState:
    """
    Generates questions and answers based on retrieved documents and optional web search results.

    Args:
        state (GraphState): The current graph state.

    Returns:
        GraphState: The updated state with 'generated_questions' populated or 'error' set.
    """
    print("--- NODE: generate_questions ---") # Message unchanged

    # First check if an API key error already exists
    if state.get("error") and "API_KEY_INVALID" in state.get("error"):
        # Propagate the API key error without changing it
        return state

    # Get LLM LAZILY
    llm = get_google_llm()
    if not llm:
        return {**state, "error": "LLM not available for generation (check API Key?)."} # Error text unchanged

    retrieved_docs = state.get("retrieved_docs", [])
    web_search_results = state.get("web_search_results")
    search_enabled = state.get("search_enabled", False)
    num_questions_to_gen = state.get("num_questions", 5) # Use .get with default
    randomize = state.get("randomize", False)

    # --- Prepare Context ---
    if not retrieved_docs and not (search_enabled and web_search_results):
         return {**state, "generated_questions": [], "error": "No content (PDF or Web) available for generation."} # Error text unchanged

    context_str = format_docs(retrieved_docs) if retrieved_docs else "No PDF context provided." # Context text unchanged

    # --- Construct Prompts (Keep existing logic) ---
    # System prompt defines the LLM's role and constraints
    system_message = f"""You are an expert quiz creation assistant. Your task is to generate EXACTLY {num_questions_to_gen} questions and answers based *only* on the provided context.
It is absolutely IMPERATIVE that you return a list containing PRECISELY {num_questions_to_gen} JSON objects. No more, no less.""" # Prompt unchanged
    web_info_prompt = ""
    if search_enabled and web_search_results:
        print("Using PDF context AND web search results for generation.") # Message unchanged
        web_info_prompt = f"\n\nHere is additional information from a web search. Use it to enrich or validate questions if relevant, but remain focused on the exact number ({num_questions_to_gen}) of questions:\n{web_search_results}" # Prompt unchanged
    elif retrieved_docs:
         print("Using PDF context only for generation.") # Message unchanged
    else: # Only web results available
         print("Using Web search results only for generation.") # Message unchanged
         context_str = "Only web search results provided as context." # Context text unchanged


    randomize_text = "Try to cover different aspects of the content." if randomize else "" # Prompt text unchanged
    # User prompt provides the context and formatting instructions
    user_content = f"""Provided Context:
{context_str}
{web_info_prompt}

{randomize_text}

STRICT FORMATTING INSTRUCTIONS:
1. Generate a valid JSON list.
2. The list must contain EXACTLY {num_questions_to_gen} objects.
3. Each object in the list must have exactly two keys: "question" (string) and "answer" (string).
4. Return NOTHING else but the JSON list itself (no introductory text, no ```json, just the list starting with '[' and ending with ']').

FINAL, ABSOLUTELY CRUCIAL REMINDER: The JSON list MUST contain exactly {num_questions_to_gen} question/answer pairs.

JSON list of the {num_questions_to_gen} questions/answers:
""" # Prompt unchanged
    messages = [ SystemMessage(content=system_message), HumanMessage(content=user_content) ]

    # --- Invoke LLM and Process Response (Keep existing robust JSON parsing logic) ---
    try:
        print(f"Calling LLM to generate {num_questions_to_gen} questions...") # Message unchanged
        llm_result = llm.invoke(messages)
        response_content = llm_result.content if llm_result else ""
        print("RAW LLM Response (start):", response_content[:250] if response_content else "Empty") # Message unchanged

        # --- Robust JSON Parsing Logic (Keep existing) ---
        json_text = ""
        try:
            # Try finding ```json ... ``` block first
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_content, re.DOTALL)
            if json_match:
                json_text = json_match.group(1).strip()
            else:
                # Fallback: find first '[' and last ']'
                start_index = response_content.find('[')
                end_index = response_content.rfind(']')
                if start_index != -1 and end_index != -1 and start_index < end_index:
                    json_text = response_content[start_index : end_index + 1].strip()
                else:
                    # Last resort: assume the whole response might be JSON
                    json_text = response_content.strip()

            # Clean potential control characters that break json.loads
            json_text = re.sub(r'[\x00-\x1F\x7F]', '', json_text)
            questions_data = json.loads(json_text)

            if isinstance(questions_data, list):
                valid_questions = []
                for item in questions_data:
                    # Validate structure and non-empty content
                    if isinstance(item, dict) and \
                       "question" in item and isinstance(item["question"], str) and item["question"].strip() and \
                       "answer" in item and isinstance(item["answer"], str) and item["answer"].strip():
                        valid_questions.append({"question": item["question"].strip(), "answer": item["answer"].strip()})
                    else:
                        print(f"  -> Invalid/empty Q/A object ignored: {item}") # Message unchanged

                num_generated_valid = len(valid_questions)
                print(f"LLM provided {num_generated_valid} valid questions.") # Message unchanged

                # Enforce exact number or take what's available
                if num_generated_valid >= num_questions_to_gen:
                    final_questions = valid_questions[:num_questions_to_gen]
                    if num_generated_valid > num_questions_to_gen:
                        print(f"WARNING: LLM generated {num_generated_valid}, truncated to {num_questions_to_gen}.") # Message unchanged
                else:
                    final_questions = valid_questions
                    if num_generated_valid < num_questions_to_gen:
                         print(f"WARNING: LLM generated only {num_generated_valid}, requested {num_questions_to_gen}. Using available.") # Message unchanged

                print(f"Final number of questions: {len(final_questions)}") # Message unchanged
                return {**state, "generated_questions": final_questions, "error": None}
            else:
                print(f"Error: Parsed JSON is not a list (type: {type(questions_data)}).") # Message unchanged
                return {**state, "error": "LLM response format error (not a JSON list)."} # Error text unchanged

        except json.JSONDecodeError as e:
            print(f"Severe JSON parsing error: {e}") # Message unchanged
            print(f"Content that failed parsing (start): '{json_text[:500]}'") # Message unchanged
            # --- Attempt Regex Repair (Keep existing) ---
            try:
                # Look for individual {"question": "...", "answer": "..."} patterns
                matches = re.findall(r'\{\s*"question"\s*:\s*"((?:\\.|[^"\\])*)"\s*,\s*"answer"\s*:\s*"((?:\\.|[^"\\])*)"\s*\}', response_content)
                if matches:
                    repaired_questions = [{"question": q.strip(), "answer": a.strip()} for q, a in matches if q.strip() and a.strip()]
                    num_repaired = len(repaired_questions)
                    print(f"Regex repair found {num_repaired} Q/A pairs.") # Message unchanged
                    if num_repaired > 0:
                        # Take up to the requested number
                        if num_repaired >= num_questions_to_gen:
                             final_questions = repaired_questions[:num_questions_to_gen]
                        else:
                             final_questions = repaired_questions
                        print(f"Final number after repair: {len(final_questions)}") # Message unchanged
                        # Return recovered questions with a warning
                        return {**state, "generated_questions": final_questions, "error": f"JSON parsing failed, but {len(final_questions)} questions recovered."} # Error text unchanged
            except Exception as repair_e:
                 print(f"Regex repair failed: {repair_e}") # Message unchanged
            # If parsing and repair fail, return error
            return {**state, "error": f"JSON Parsing Error: {e}. Could not retrieve questions."} # Error text unchanged
        except Exception as inner_e:
             # Catch any other unexpected errors during processing
             print(f"Unexpected error during response processing: {inner_e}") # Message unchanged
             traceback.print_exc()
             return {**state, "error": f"Internal Post-LLM Error: {inner_e}"} # Error text unchanged

    except Exception as e:
        error_message = str(e)
        print(f"Major error during LLM call: {e}") # Message unchanged
        traceback.print_exc()

        # Check if the error is related to an invalid API key
        if "API key not valid" in error_message or "API_KEY_INVALID" in error_message:
            return {**state, "error": "API_KEY_INVALID: The Google API key provided is invalid or has expired."} # Error text unchanged

        return {**state, "error": f"LLM Call Error: {e}"} # Error text unchanged


def web_search_node(state: GraphState) -> GraphState:
    """
    Performs a web search using Tavily based on context or generated questions (if available).

    Args:
        state (GraphState): The current graph state.

    Returns:
        GraphState: The updated state with 'web_search_results' populated or 'error' set on search failure.
    """
    print("--- NODE: web_search ---") # Message unchanged

    # First check if an API key error already exists
    if state.get("error") and ("API_KEY_INVALID" in state.get("error") or "TAVILY_KEY_INVALID" in state.get("error")):
         # Propagate the API key error without changing it
         # Don't attempt search if a relevant key is known to be invalid
         return state

    # Get search tool LAZILY
    web_search_tool = get_tavily_search()
    if not web_search_tool:
        print("Web search skipped: Tavily tool not available/configured (check API Key?).") # Message unchanged
        # If search was intended but tool isn't there, maybe set a warning?
        # If state['search_enabled'] is True:
        #     state['error'] = "Web search was enabled, but Tavily tool/key is missing." # Error text unchanged (commented out)
        return {**state, "web_search_results": None} # Don't overwrite existing error if any

    retrieved_docs = state.get("retrieved_docs", [])
    generated_questions = state.get("generated_questions", [])

    # --- Dynamic Search Query (Keep existing) ---
    # Construct a search query dynamically based on available information
    search_query = "Relevant topics based on the provided document" # Prompt text unchanged
    if generated_questions:
        # Use first few generated questions if available
        topics = " ".join([q.get("question", "") for q in generated_questions[:2]])
        search_query = f"Fact-checking or related information on: {topics}" if topics else search_query # Prompt text unchanged
    elif retrieved_docs:
        # Use the beginning of the first retrieved document
        search_query = f"Information related to: {retrieved_docs[0][:150]}..." # Prompt text unchanged

    try:
        print(f"Executing web search for: '{search_query}'") # Message unchanged
        # Invoke the Tavily search tool
        results = web_search_tool.invoke({"query": search_query}) # Ensure input format matches tool

        # Format results: Tavily usually returns a list of dictionaries
        if isinstance(results, list) and all(isinstance(res, dict) for res in results):
             formatted_results = "\n\n".join([f"Source: {res.get('url', 'N/A')}\nContent: {res.get('content', '')}" for res in results]) # Prompt-like text unchanged
        elif isinstance(results, str): # Sometimes it might return a formatted string
             formatted_results = results
        else:
             print(f"Warning: Unexpected Tavily result format: {type(results)}") # Message unchanged
             formatted_results = str(results) # Fallback

        print(f"Search results obtained ({len(formatted_results)} chars).") # Message unchanged
        # Don't clear existing errors, just add results
        return {**state, "web_search_results": formatted_results}
    except Exception as e:
        error_message = str(e)
        print(f"Error during Tavily web search: {e}") # Message unchanged
        traceback.print_exc()

        # Check if the error is related to an invalid API key
        if "API key not valid" in error_message or "API_KEY_INVALID" in error_message:
            return {**state, "error": "TAVILY_KEY_INVALID: The Tavily API key provided is invalid or has expired."} # Error text unchanged

        # Store as non-blocking error, keep existing web_search_results
        return {**state, "error": f"Tavily Error (non-blocking): {e}"} # Error text unchanged


# --- Conditional Edge Logic (Keep existing) ---
def decide_to_search_or_generate(state: GraphState) -> str:
    """
    Decides the next step after document retrieval based on whether an error occurred.

    Args:
        state (GraphState): The current graph state.

    Returns:
        str: The name of the next node ("generate_questions_node" or "end_node").
    """
    print("--- CONDITIONAL EDGE: decide_to_search_or_generate ---") # Message unchanged
    if state.get("error"):
        print(f"Error detected ({state['error']}), proceeding to end node.") # Message unchanged
        return "end_node"
    # If retrieval was successful, proceed to generation
    print("Decision: Proceed to question generation.") # Message unchanged
    return "generate_questions_node"

def decide_after_generation(state: GraphState) -> str:
    """
    Decides the next step after question generation.
    Proceeds to web search if enabled and possible, otherwise ends the graph.

    Args:
        state (GraphState): The current graph state.

    Returns:
        str: The name of the next node ("web_search_node", "end_node", or END).
    """
    print("--- CONDITIONAL EDGE: decide_after_generation ---") # Message unchanged
    generation_error = state.get("error")
    generated_questions = state.get("generated_questions", [])

    # Check if a MAJOR error occurred preventing any generation (ignore recovery warnings)
    if generation_error and not generated_questions and "recovered" not in generation_error.lower():
         print(f"Major generation error ({generation_error}), proceeding to end node.") # Message unchanged
         return "end_node"

    # Check if web search is enabled *in the state* (reflects user choice)
    search_is_actually_enabled = state.get("search_enabled", False)
    # Check if Tavily key is actually present in the environment
    tavily_key_present = bool(os.environ.get("TAVILY_API_KEY")) # Check env var set by app.py

    if search_is_actually_enabled and tavily_key_present:
        print("Web search is enabled and key is present, proceeding to web search node.") # Message unchanged
        return "web_search_node"
    else:
        if search_is_actually_enabled and not tavily_key_present:
             print("Web search was requested but key is missing. Skipping search and ending.") # Message unchanged
        else:
             print("Web search disabled or key missing. Ending graph execution.") # Message unchanged
        # If generation succeeded (or partially recovered) but no search, end successfully
        if generated_questions or (generation_error and "recovered" in generation_error.lower()):
             return END # Signal successful completion
        else: # No questions generated, no search -> go to end node to report failure explicitly
             if not generation_error: # Set error if not already set
                  state["error"] = "No questions generated and web search not enabled/possible." # Error text unchanged
             return "end_node" # Go to terminal error node


# --- Graph Construction (Keep existing) ---
def build_rag_graph() -> Any:
    """
    Builds and compiles the LangGraph StateGraph for the RAG pipeline.

    Returns:
        Compiled LangGraph application.
    """
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("setup_vector_store", setup_vector_store)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("generate_questions_node", generate_questions_node)
    workflow.add_node("web_search_node", web_search_node)
    # Define a basic end node for explicit termination paths
    workflow.add_node("end_node", lambda state: print(f"--- GRAPH TERMINATED (Via end_node) --- Error='{state.get('error')}', Qs={len(state.get('generated_questions', []))}") or state) # Message unchanged

    # Define the edges and entry point
    workflow.set_entry_point("setup_vector_store")
    workflow.add_edge("setup_vector_store", "retrieve_documents")

    # Conditional edge after retrieval: proceed if ok, end if error
    workflow.add_conditional_edges(
        "retrieve_documents",
        decide_to_search_or_generate,
        {
            "generate_questions_node": "generate_questions_node",
            "end_node": "end_node"
        }
    )
    # Conditional edge after generation: search if enabled, otherwise end
    workflow.add_conditional_edges(
        "generate_questions_node",
        decide_after_generation,
        {
            "web_search_node": "web_search_node",
            "end_node": "end_node", # Go to explicit end node on major generation failure without search
            END: END # Go to implicit successful end if generation ok and no search needed
        }
    )
    # After web search, always end the graph
    workflow.add_edge("web_search_node", END)
    # Explicit end node connection
    workflow.add_edge("end_node", END)

    print("Compiling LangGraph...") # Message unchanged
    app = workflow.compile()
    print("Graph compiled.") # Message unchanged
    return app

# Compile the graph when the module is loaded.
compiled_graph = build_rag_graph()


# --- Main Execution Function ---
# Keep run_generation_pipeline as is, it expects app.py to set env vars correctly
def run_generation_pipeline(texts: List[str], search_enabled: bool = False, num_questions: int = 5, randomize: bool = False) -> Dict[str, Any]:
    """
    Runs the full RAG question generation pipeline using the compiled LangGraph app.

    It handles text chunking, setting up the initial state, invoking the graph,
    and returning the final results (questions and/or errors).

    Args:
        texts (List[str]): A list containing the text content, potentially pre-chunked or as a single string.
        search_enabled (bool): Whether to enable the web search step (requires Tavily API key).
        num_questions (int): The target number of questions to generate.
        randomize (bool): Whether to add a randomization hint to the generation prompt.

    Returns:
        Dict[str, Any]: A dictionary containing 'generated_questions' (List[Dict[str, str]])
                        and 'error' (Optional[str]).
    """
    # ... (Keep existing logic for chunking and invoking compiled_graph) ...
    # It relies on the environment variables being set correctly *before* this function is called.

    pdf_chunks: List[str] = []
    # Handle input: could be a single string needing chunking, or pre-chunked list
    if isinstance(texts, list) and len(texts) == 1 and isinstance(texts[0], str):
        print("Input text appears to be a single string, chunking...") # Message unchanged
        pdf_chunks = chunk_text(texts[0])
        if not pdf_chunks:
             print("Warning: Text chunking produced no chunks.") # Message unchanged
             return {"error": "Failed to chunk the input text.", "generated_questions": []} # Error text unchanged
    elif isinstance(texts, list) and all(isinstance(t, str) for t in texts):
         # Assume input is already chunked if it's a list of strings
         pdf_chunks = texts
         print(f"Using {len(pdf_chunks)} provided text chunks.") # Message unchanged
    else:
        # Handle invalid input format
        print(f"Error: Unexpected input format for 'texts': {type(texts)}. Expected List[str].") # Message unchanged
        return {"error": "Invalid input data format for the pipeline.", "generated_questions": []} # Error text unchanged

    # Prepare the initial state for the graph
    initial_state = GraphState(
        pdf_text_chunks=pdf_chunks,
        search_enabled=search_enabled, # Reflects user intent; nodes will check key availability
        num_questions=num_questions,
        randomize=randomize,
        vector_store=None,
        retrieved_docs=None,
        web_search_results=None,
        generated_questions=[],
        error=None
    )

    print(f"--- Starting Graph Execution (Web Search: {search_enabled}, Requested Questions: {num_questions}) ---") # Message unchanged
    final_state = None
    try:
        # Before invoking, double-check necessary env vars are set (useful for debugging)
        print(f"DEBUG: GOOGLE_API_KEY set: {bool(os.environ.get('GOOGLE_API_KEY'))}") # Message unchanged
        print(f"DEBUG: TAVILY_API_KEY set: {bool(os.environ.get('TAVILY_API_KEY'))}") # Message unchanged

        # Invoke the compiled graph with the initial state
        # recursion_limit is a safety measure against infinite loops
        final_state = compiled_graph.invoke(initial_state, {"recursion_limit": 25})

        # Process the final state
        error_message = final_state.get("error")
        generated_questions = final_state.get("generated_questions", [])

        if error_message:
            print(f"Graph finished with Error/Warning: {error_message}") # Message unchanged
            # Don't treat "recovered" as a fatal error if questions were actually generated
            is_fatal_error = "recovered" not in error_message.lower() or not generated_questions
            return {
                # Return error message, but also include any questions that might have been generated before the error
                "error": error_message,
                "generated_questions": generated_questions
            }

        if not generated_questions:
            # If the graph finished without error but produced no questions
            print("Graph finished, but no questions were generated.") # Message unchanged
            return {
                "error": "No questions could be generated successfully.", # Error text unchanged
                "generated_questions": []
            }

        # Successful execution with generated questions
        print(f"Graph finished successfully. {len(generated_questions)} questions generated.") # Message unchanged
        return {
            "generated_questions": generated_questions,
            "error": None # Explicitly indicate no error
        }

    except Exception as e:
        # Catch major unexpected errors during the graph invocation itself
        print(f"Major unexpected error during graph execution: {e}") # Message unchanged
        traceback.print_exc()
        error_msg = f"Major System Error during graph execution: {str(e)}" # Error text unchanged
        # Try to return any partial results if possible
        partial_questions = final_state.get("generated_questions", []) if final_state else []
        return { "error": error_msg, "generated_questions": partial_questions }
    finally:
        # Always print this, regardless of success or failure
        print("--- End of Graph Execution ---") # Message unchanged