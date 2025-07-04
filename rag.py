import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import google.generativeai as genai
import logging
import time
from dotenv import load_dotenv
import sys
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class IndianRecipeRAG:
    def __init__(self, dataset_path: str):
        """
        Initialize the Indian Recipe RAG system
        
        Args:
            dataset_path: Path to Excel dataset file
        """
        self.dataset_path = dataset_path
        self.df = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
        # Load and preprocess dataset
        self.load_dataset()
        
        # Create search index
        self.create_search_index()
        
    def load_dataset(self):
        """Load and preprocess the Indian food dataset"""
        try:
            logger.info(f"Loading dataset from {self.dataset_path}")
            self.df = pd.read_excel(self.dataset_path)
            
            # Clean and preprocess data
            # Fill NaN values with empty strings
            self.df = self.df.fillna('')
            
            # Create searchable text by combining relevant columns
            self.df['SearchText'] = self.df.apply(
                lambda row: f"{row.get('RecipeName', '')} {row.get('TranslatedRecipeName', '')} "
                           f"{row.get('Ingredients', '')} {row.get('TranslatedIngredients', '')} "
                           f"{row.get('Cuisine', '')} {row.get('Course', '')} {row.get('Diet', '')}",
                axis=1
            )
            
            logger.info(f"Loaded {len(self.df)} recipes")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def create_search_index(self):
        """Create TF-IDF matrix for recipe search"""
        try:
            # Generate TF-IDF matrix for recipes
            logger.info("Generating TF-IDF matrix for recipes...")
            texts = self.df['SearchText'].tolist()
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            logger.info(f"Created TF-IDF matrix with shape {self.tfidf_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Error creating search index: {e}")
            raise
    
    def search_recipes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for recipes using TF-IDF and cosine similarity
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching recipes as dictionaries
        """
        try:
            # Transform query to TF-IDF vector
            query_vec = self.vectorizer.transform([query])
            
            # Calculate cosine similarity between query and all recipes
            cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top k results
            top_indices = cosine_similarities.argsort()[-top_k:][::-1]
            
            # Get matching recipes
            results = []
            for idx in top_indices:
                recipe = self.df.iloc[idx].to_dict()
                recipe['score'] = float(cosine_similarities[idx])
                results.append(recipe)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching recipes: {e}")
            return []
    
    def format_recipe_for_prompt(self, recipe: Dict[str, Any]) -> str:
        """Format a recipe for inclusion in a prompt"""
        try:
            # Format recipe details in a structured way
            formatted = f"""
Recipe: {recipe.get('RecipeName', '')}
Translated Name: {recipe.get('TranslatedRecipeName', '')}
Cuisine: {recipe.get('Cuisine', '')}
Course: {recipe.get('Course', '')}
Diet: {recipe.get('Diet', '')}
Prep Time: {recipe.get('PrepTimeInMins', '')} mins
Cook Time: {recipe.get('CookTimeInMins', '')} mins
Total Time: {recipe.get('TotalTimeInMins', '')} mins
Servings: {recipe.get('Servings', '')}

Ingredients:
{recipe.get('Ingredients', '')}

Instructions:
{recipe.get('Instructions', '')}
"""
            return formatted
        
        except Exception as e:
            logger.error(f"Error formatting recipe: {e}")
            return str(recipe)
    
    def generate_response(self, user_query: str, model) -> str:
        """
        Generate a response using RAG approach
        
        Args:
            user_query: User query about recipes
            model: LLM model to use for generation
            
        Returns:
            Generated response
        """
        try:
            # Search for relevant recipes
            relevant_recipes = self.search_recipes(user_query, top_k=3)
            
            if not relevant_recipes:
                # If no recipes found, generate a response without context
                prompt = f"""
                You are an expert Indian cuisine chef specializing in authentic recipes.
                
                User Query: {user_query}
                
                I couldn't find specific recipes in my database that match this query. 
                Please provide a helpful response about Indian cooking related to this query.
                """
            else:
                # Format recipes for context
                recipe_contexts = [self.format_recipe_for_prompt(recipe) for recipe in relevant_recipes]
                context = "\n\n".join(recipe_contexts)
                
                # Create prompt with context
                prompt = f"""
                You are an expert Indian cuisine chef specializing in authentic recipes.
                
                User Query: {user_query}
                
                Here are relevant recipes from my database:
                
                {context}
                
                Based on these recipes and your knowledge of Indian cuisine, please provide a helpful response to the user's query.
                If they're asking for a recipe, provide a clear, step-by-step recipe based on the relevant recipes.
                If they're asking for information, provide detailed information about Indian cuisine.
                If they're asking for recommendations, suggest specific dishes based on their preferences.
                Make sure your response is well-structured, informative, and helpful.
                """
            
            # Generate response using Gemini
            response = model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your request. Please try again later."

def initialize_gemini_model():
    """Initialize and return the Gemini model"""
    try:
        # Get API key from environment or Streamlit secrets
        api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        
        if not api_key:
            st.error("GEMINI_API_KEY not found. Please set it in your environment or Streamlit secrets.")
            st.stop()
        
        genai.configure(api_key=api_key)
        MODEL = "gemini-1.5-pro"  # Use the appropriate Gemini model
        
        # Initialize the model
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
        
        model = genai.GenerativeModel(
            model_name=MODEL,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        return model
        
    except Exception as e:
        st.error(f"Error initializing Gemini API: {e}")
        st.stop()

def load_rag_system(dataset_path):
    """Initialize and return the RAG system"""
    with st.spinner("Loading recipe database..."):
        try:
            return IndianRecipeRAG(dataset_path)
        except Exception as e:
            st.error(f"Error loading recipe database: {e}")
            st.stop()

def display_message(role, content, avatar=None):
    """Display a message in the chat interface"""
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

def main():
    st.set_page_config(
        page_title="Indian Recipe Chatbot",
        page_icon="🍛",
        layout="wide"
    )

    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .bot-message {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e6f7ff;
        padding: 15px;
        border-radius: 10px;
        text-align: right;
    }
    .recipe-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    # App header
    st.markdown("""
    <div class="header-container">
        <h1>🍛 Indian Recipe Chatbot 🍛</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Ask me about Indian recipes, cooking techniques, ingredients, or regional cuisines!
    """)

    # Sidebar with info
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/curry.png", width=80)
        st.title("About this app")
        st.markdown("""
        This is an AI-powered chatbot specializing in Indian cuisine. 
        It uses a RAG (Retrieval-Augmented Generation) system to provide
        accurate information about Indian recipes and cooking.
        
        ### Features
        * Recipe recommendations
        * Cooking techniques
        * Ingredient substitutions
        * Regional cuisines
        * Dietary modifications
        
        ### How it works
        1. Your query is processed to find relevant recipes
        2. The AI uses those recipes as context
        3. The AI generates a helpful, accurate response
        """)
        
        # Optional: Add dataset info or other controls
        st.divider()
        st.caption("Powered by Google Gemini API")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_system" not in st.session_state:
        # Path to dataset - change to your file path
        dataset_path = "C:\\Users\\zalaj\\Codes\\FNN project\\pretrained\\FoodDatasetXLS.xlsx" 
        st.session_state.rag_system = load_rag_system(dataset_path)
    
    if "model" not in st.session_state:
        st.session_state.model = initialize_gemini_model()

    # Display chat history
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🍛"
        display_message(message["role"], message["content"], avatar)

    # Chat input
    if prompt := st.chat_input("Ask something about Indian cooking..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt, "🧑‍💻")
        
        # Show thinking indicator
        with st.status("Searching for recipes and crafting response...", expanded=True) as status:
            st.write("Finding relevant recipes...")
            
            # Generate response
            start_time = time.time()
            response = st.session_state.rag_system.generate_response(prompt, st.session_state.model)
            end_time = time.time()
            
            st.write("Generating response...")
            status.update(label="Done!", state="complete", expanded=False)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        display_message("assistant", response, "🍛")
        
        # Display response time
        st.caption(f"Response generated in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()