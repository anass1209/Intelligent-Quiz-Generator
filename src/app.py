import os
import uuid
import traceback # Import traceback

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from werkzeug.utils import secure_filename
from utils import load_api_keys
from pdf_processor import extract_text_from_pdfs
from rag_graph import run_generation_pipeline

# Attempt to load Flask secret key (API keys are now handled mainly by session)
try:
    # We still load GOOGLE_API_KEY/TAVILY_API_KEY from .env here IF they exist,
    _, _, FLASK_SECRET_KEY = load_api_keys()
except ValueError as e:
    print(f"Critical startup error: {e}") # Keep: Server-side message
    FLASK_SECRET_KEY = "default-insecure-key-change-me" # WARNING: Insecure default
    print("WARNING: Using a default, insecure Flask secret key.") # Keep: Server-side message

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = FLASK_SECRET_KEY
app.config['UPLOAD_FOLDER'] = 'uploads'
# Set max upload size to 32MB !
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# Ensure the upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- Flask Routes ---
@app.route('/')
def index():
    """
    Renders the main index page (index.html).
    JavaScript on the frontend handles displaying the API key modal if necessary.
    """
    # Render index.html - The JS will handle showing the modal if keys are missing
    return render_template('index.html')

@app.route('/save-api-keys', methods=['POST'])
def save_api_keys():
    """
    Receives API keys (Gemini, Tavily) from the frontend modal via a JSON POST request.
    Validates the presence of the Gemini key and saves both keys into the user's server-side session.
    Returns a JSON response indicating success or failure.
    """
    try:
        data = request.get_json()
        if not data:
            # Return error if no JSON data is received
            return jsonify({'success': False, 'message': 'No data received.'}), 400 # Message unchanged

        gemini_key = data.get('gemini_api_key')
        tavily_key = data.get('tavily_api_key') # This key is optional, can be None or empty

        # Gemini key is mandatory for core functionality
        if not gemini_key:
             return jsonify({'success': False, 'message': 'Gemini API Key is required.'}), 400 # Message unchanged

        # Store the keys securely in the server-side session
        session['GOOGLE_API_KEY'] = gemini_key
        session['TAVILY_API_KEY'] = tavily_key # Store even if it's None or empty

        print("API Keys saved to session.") # Keep: Server-side log confirmation
        # Confirm successful save to the frontend
        return jsonify({'success': True, 'message': 'API Keys saved successfully.'}) # Message unchanged

    except Exception as e:
        # Log any unexpected errors during the save process
        print(f"Error saving API keys: {e}") # Keep: Server-side error log
        traceback.print_exc() # Keep: Detailed traceback for debugging
        # Inform the frontend about the server error
        return jsonify({'success': False, 'message': f'Server error: {e}'}), 500 # Message unchanged


# Modified /upload route to handle API key errors from the RAG pipeline
@app.route('/upload', methods=['POST'])
def upload():
    """
    Handles PDF file uploads via POST request.
    Checks for required API keys in the session.
    Validates uploaded files, extracts text content.
    Temporarily sets API keys in the environment for the RAG pipeline.
    Calls the `run_generation_pipeline` from `rag_graph.py`.
    Processes the results, handles potential API key errors reported by the pipeline,
    stores generated questions in the session, and redirects to the quiz page or back to index on error.
    Cleans up uploaded files afterwards.
    """
    # --- API Key Check (from Session) ---
    session_google_key = session.get('GOOGLE_API_KEY')
    session_tavily_key = session.get('TAVILY_API_KEY') # May be None or empty string

    # Gemini key is essential, redirect if missing
    if not session_google_key:
        flash('Google API Key is missing. Please configure API keys using the gear icon.', 'error') # Message unchanged
        # Redirect back to index; frontend JS should prompt for keys.
        return redirect(url_for('index'))

    # --- File Upload Validation (Keep existing logic) ---
    # Check if the file part is present in the request
    if 'pdf_files' not in request.files:
        flash('No file part in the request.', 'error') # Message unchanged
        return redirect(url_for('index'))
    # Get the list of files from the form field 'pdf_files'
    files = request.files.getlist('pdf_files')
    # Check if the list is empty or contains only files with empty filenames
    if not files or all(f.filename == '' for f in files):
        flash('No PDF files selected for upload.', 'error') # Message unchanged
        return redirect(url_for('index'))

    # --- Retrieve Number of Questions (Keep existing logic) ---
    try:
        # Get 'num_questions' from form, default to 10 if missing or invalid
        num_questions = int(request.form.get('num_questions', 10))
        # Ensure the number is within a reasonable range (e.g., 3 to 20)
        num_questions = max(3, min(20, num_questions))
    except (ValueError, TypeError):
        # Fallback to default if conversion fails
        num_questions = 10

    # --- Determine if Web Search is Enabled ---
    # Check if the 'search_enabled' checkbox was checked in the form
    search_enabled_form = 'search_enabled' in request.form
    # Enable search feature ONLY if the checkbox was checked AND the Tavily API key is present in the session
    search_enabled = search_enabled_form and bool(session_tavily_key)

    # If the user requested search but the key is missing, inform them via flash message
    if search_enabled_form and not session_tavily_key:
        flash('Web search was requested but is disabled because the Tavily API Key is missing.', 'warning') # Message unchanged

    pdf_paths = [] # List to store paths of successfully saved PDF files
    saved_files = [] # List to track all saved files for cleanup in finally block

    try:
        # --- Process and Save Uploaded Files (Keep existing logic) ---
        for file in files:
            # Process only if a file exists and has a .pdf extension
            if file and file.filename.endswith('.pdf'):
                base_filename = secure_filename(file.filename) # Sanitize filename for security
                # Create a unique filename using UUID to prevent collisions
                unique_filename = f"{uuid.uuid4()}_{base_filename}"
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(pdf_path) # Save the uploaded file to the designated folder
                pdf_paths.append(pdf_path) # Add path to list for text extraction
                saved_files.append(pdf_path) # Add path to list for cleanup
            else:
                 # Inform user about any ignored non-PDF files
                 flash(f"Ignored file (not PDF or invalid): {file.filename}", 'info') # Message unchanged

        # If after processing all files, no valid PDFs were found
        if not pdf_paths:
             flash('No valid PDF files were uploaded.', 'error') # Message unchanged
             return redirect(url_for('index'))

        # --- PDF Text Extraction (Keep existing logic) ---
        print("Extracting text from PDFs...") # Keep: Server-side log message
        try:
            # Call the function to extract text from the list of PDF paths
            # Assumes it returns a single string concatenating text from all PDFs
            extracted_texts = extract_text_from_pdfs(pdf_paths)
            # If extraction result is empty, inform user and redirect
            if not extracted_texts:
                 flash("Could not extract any text from the provided PDFs.", 'error') # Message unchanged
                 return redirect(url_for('index'))
        except Exception as e:
             # Handle potential errors during the text extraction process
             print(f"Error during text extraction: {e}") # Keep: Server-side log
             flash(f"Error reading PDF files: {e}", 'error') # Message unchanged
             return redirect(url_for('index'))

        # --- Run RAG Generation Pipeline ---
        print("Launching LangGraph generation pipeline...") # Keep: Server-side log message
        print(f"Search Enabled: {search_enabled} (Form: {search_enabled_form}, Key Present: {bool(session_tavily_key)})") # Keep: Server-side log message
        print(f"Number of Questions: {num_questions}") # Keep: Server-side log message

        # --- IMPORTANT: Temporarily set environment variables for the pipeline run ---
        # The RAG graph nodes use os.environ.get() for lazy initialization of clients.
        # We need to set these based on the current user's session data.
        original_google_key = os.environ.get("GOOGLE_API_KEY")
        original_tavily_key = os.environ.get("TAVILY_API_KEY")

        os.environ["GOOGLE_API_KEY"] = session_google_key # Set Gemini key
        if session_tavily_key:
            os.environ["TAVILY_API_KEY"] = session_tavily_key # Set Tavily key if present
        elif "TAVILY_API_KEY" in os.environ:
             # If Tavily key is NOT in session but IS in environment, remove it
             # Prevents using a key from a previous request or system env
             del os.environ["TAVILY_API_KEY"]

        generation_result = {} # Initialize result dict
        try:
            # Call the pipeline function with extracted text(s) and configuration
            # Pass extracted_texts as a list containing one large string
            generation_result = run_generation_pipeline(
                [extracted_texts],
                search_enabled=search_enabled, # Pass the determined search status
                num_questions=num_questions, # Pass the desired number of questions
                randomize=True # Example setting, could be form-controlled
            )
        except Exception as e:
            # Catch unexpected errors *during* the pipeline's execution
            error_message = str(e)
            print(f"Error during generation pipeline execution: {error_message}") # Keep: Server-side log message
            traceback.print_exc() # Keep: Detailed traceback

            # Check if the exception indicates an invalid API key
            if "API key not valid" in error_message or "API_KEY_INVALID" in error_message:
                # Remove the potentially invalid key from the user's session
                session.pop('GOOGLE_API_KEY', None)
                # Provide a specific error message to the user
                flash("The Gemini API key you provided is invalid. Please enter a valid key.", 'error') # Message unchanged
            else:
                # Provide a general error message for other pipeline failures
                flash(f"An error occurred during quiz generation: {error_message}", 'error') # Message unchanged

            # Redirect back to the index page after handling the exception
            return redirect(url_for('index'))
        finally:
            # --- Restore original environment variables ---
            # This is crucial to ensure subsequent requests are not affected by this request's keys.
            if original_google_key:
                os.environ["GOOGLE_API_KEY"] = original_google_key
            elif "GOOGLE_API_KEY" in os.environ:
                # Only delete if it was set by this request and not originally present
                del os.environ["GOOGLE_API_KEY"]

            if original_tavily_key:
                os.environ["TAVILY_API_KEY"] = original_tavily_key
            elif "TAVILY_API_KEY" in os.environ:
                # Only delete if it was set/unset by this request and not originally present
                del os.environ["TAVILY_API_KEY"]
        # --- End Environment Variable Handling ---

        # --- Process Generation Results (returned dictionary) ---
        generated_questions = generation_result.get("generated_questions", [])
        error_message = generation_result.get("error") # Get error message reported *by the pipeline*
        print(f"Number of questions received from generation: {len(generated_questions)}") # Keep: Server-side log

        # --- Handle API Key Errors Reported by RAG Graph ---
        # This checks the 'error' field in the dictionary returned by the pipeline.
        # It catches errors detected *within* the graph nodes (e.g., invalid key used in API call).
        if error_message and ("API key not valid" in error_message or "API_KEY_INVALID" in error_message or "TAVILY_KEY_INVALID" in error_message):
            # If the error message indicates an invalid Google/Gemini key
            if "GOOGLE" in error_message or "API_KEY_INVALID" in error_message:
                 session.pop('GOOGLE_API_KEY', None) # Remove invalid key from session
                 flash("The Gemini API key you provided is invalid. Please enter a valid key.", 'error') # Message unchanged
            # If the error message indicates an invalid Tavily key
            if "TAVILY" in error_message or "TAVILY_KEY_INVALID" in error_message:
                 session.pop('TAVILY_API_KEY', None) # Remove invalid key from session
                 flash("The Tavily API key you provided seems invalid. Web search disabled.", 'warning') # Message unchanged

            # Redirect to index if an API key was identified as invalid by the pipeline
            return redirect(url_for('index'))

        # --- Handle Other Errors or Lack of Questions ---
        if error_message:
            # If an error message exists, display it. Use 'warning' if some questions were generated despite the error.
            flash_level = 'warning' if generated_questions else 'error'
            flash(f"Generation Info/Error: {error_message}", flash_level) # Message unchanged
            # If there was an error AND absolutely no questions were generated, redirect back.
            if not generated_questions:
                 return redirect(url_for('index'))
        # If no error message was reported, but the questions list is still empty
        elif not generated_questions:
             flash("No questions could be generated from the documents.", 'info') # Message unchanged
             # Redirect if pipeline finished "successfully" but yielded no questions
             return redirect(url_for('index'))

        # --- Store Results and Redirect to Quiz Page on Success ---
        # If questions were generated (potentially with warnings), proceed to the quiz.
        session['quiz_questions'] = generated_questions # Store questions in session
        session['current_question_index'] = 0 # Reset index for the new quiz
        flash(f"{len(generated_questions)} questions generated successfully!", 'success') # Message unchanged
        return redirect(url_for('quiz')) # Redirect user to the quiz display page

    except Exception as e:
        # --- General Catch-All Error Handling for /upload Route ---
        # Catches unexpected errors not handled by specific try/except blocks above.
        print(f"Unexpected error in /upload: {e}") # Keep: Server-side log
        traceback.print_exc() # Keep: Detailed traceback

        # Perform a final check for API key errors in the exception message string
        error_message_str = str(e)
        if "API key not valid" in error_message_str or "API_KEY_INVALID" in error_message_str:
            session.pop('GOOGLE_API_KEY', None)
            flash("The Gemini API key you provided is invalid. Please enter a valid key.", 'error') # Message unchanged
        elif "TAVILY_KEY_INVALID" in error_message_str:
            session.pop('TAVILY_API_KEY', None)
            flash("The Tavily API key you provided seems invalid. Web search disabled.", 'warning') # Message unchanged
        else:
            # General error message for the user if it's not an identified API key issue
            flash(f"A server error occurred: {e}", 'error') # Message unchanged

        # Redirect to index page after logging the error and flashing a message
        return redirect(url_for('index'))

    finally:
        # --- Temporary File Cleanup (Keep existing logic) ---
        # This block executes regardless of whether an error occurred or not.
        print("Cleaning up temporary files...") # Keep: Server-side log
        for file_path in saved_files:
            try:
                os.remove(file_path) # Attempt to delete each saved file
                print(f"Deleted: {file_path}") # Keep: Server-side log
            except OSError as e:
                # Log deletion errors but don't stop the process or alert the user
                print(f"Error deleting file {file_path}: {e}") # Keep: Server-side log

@app.route('/quiz')
def quiz():
    """
    Displays the quiz interface (quiz.html).
    Retrieves the generated questions from the user's session.
    If no questions are found in the session, redirects the user back to the index page with a message.
    """
    # --- Keep existing quiz route logic ---
    questions = session.get('quiz_questions') # Retrieve questions from session
    if not questions:
        # If session does not contain quiz questions, redirect to index
        flash("No quiz is currently loaded. Please upload PDFs first.", 'info') # Message unchanged
        return redirect(url_for('index'))
    # Render the quiz template, passing the list of questions to it
    return render_template('quiz.html', questions=questions)

@app.route('/validate-api-key', methods=['POST'])
def validate_api_key():
    """
    Validates a provided Gemini API key by attempting a simple, low-cost API call (embedding).
    Receives the key via a JSON POST request from the frontend.
    Temporarily sets the key in the environment for the test.
    Returns a JSON response indicating whether the key is 'valid' (True/False) and a corresponding 'message'.
    Handles specific API key errors and general exceptions during validation.
    """
    try:
        data = request.get_json()
        if not data:
            # Handle case where no JSON data is received
            return jsonify({'valid': False, 'message': 'No data received.'}) # Message unchanged

        gemini_key = data.get('gemini_api_key')

        if not gemini_key:
            # Handle case where the key is missing in the received data
            return jsonify({'valid': False, 'message': 'No API key provided.'}) # Message unchanged

        # Temporarily set the environment variable GOOGLE_API_KEY for the validation call
        original_key = os.environ.get("GOOGLE_API_KEY")
        os.environ["GOOGLE_API_KEY"] = gemini_key

        is_valid = False # Default to invalid
        validation_message = "Unknown validation error." # Default message

        try:
            # Attempt a simple API call to test the key's validity
            # Using embeddings is generally a quick and low-cost check
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            embeddings.embed_query("test") # Perform a dummy embedding request

            # If the above call succeeds without throwing an exception, the key is likely valid
            is_valid = True
            validation_message = 'API key is valid.' # Message unchanged

        except Exception as e:
            # Catch exceptions during the API call attempt
            error_message_str = str(e)
            print(f"API Key Validation Error: {error_message_str}") # Keep: Log the specific error server-side

            # Check if the error message specifically indicates an invalid key
            if "API key not valid" in error_message_str or "API_KEY_INVALID" in error_message_str:
                is_valid = False
                validation_message = 'The provided API key is invalid.' # Message unchanged
            else:
                # For other errors (e.g., network issues, quota exceeded temporarily), report as invalid/error
                is_valid = False
                validation_message = f'Error testing API key: {error_message_str}' # Message unchanged

        finally:
            # --- Restore original environment variable state ---
            # Ensure the environment is clean for subsequent operations
            if original_key:
                os.environ["GOOGLE_API_KEY"] = original_key
            elif "GOOGLE_API_KEY" in os.environ:
                # Only delete if it was set by this function and wasn't there before
                del os.environ["GOOGLE_API_KEY"]

        # Return the validation result to the frontend
        return jsonify({'valid': is_valid, 'message': validation_message})

    except Exception as e:
        print(f"Server error during API key validation: {e}") # Keep: Server-side log
        traceback.print_exc() # Keep: Detailed traceback
        return jsonify({'valid': False, 'message': f'Server error: {str(e)}'}) # Message unchanged


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True) # Use debug=True ONLY for development