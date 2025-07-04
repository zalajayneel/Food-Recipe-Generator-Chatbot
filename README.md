# Food-Recipe-Generator-Chatbot
🍲 Food Recipe Generator Chatbot
The Food Recipe Generator Chatbot is an AI-powered assistant designed to provide users with accurate and personalized Indian food recipes through natural conversation. Built using a Retrieval-Augmented Generation (RAG) model, this chatbot merges the strengths of information retrieval and generative language models to produce high-quality, context-aware recipe suggestions.

🧠 How It Works
This project leverages a RAG architecture, which consists of two main components:

Retriever: Given a user’s query (e.g., “How do I make butter paneer?” or “Recipe with rice and tomato”), the retriever searches through a curated dataset of Indian recipes and retrieves the most relevant documents or entries.

Generator: The generator (typically a Transformer-based language model) then conditions its response on the retrieved documents to produce a fluent, human-like answer — a complete recipe or cooking guidance.

This two-step process ensures that the chatbot delivers factually accurate results grounded in the dataset, while maintaining the fluency and adaptability of a generative model.

📦 Features
🍛 Support for diverse Indian dishes: Includes thousands of recipes from the IndianFoodDatasetXLS.csv file, covering snacks, curries, breads, rice dishes, and desserts.

🔍 Retrieval-based grounding: Uses contextually relevant data retrieved in real-time to enhance accuracy.

💬 Natural language interface: Users can chat using simple language like "Tell me a recipe with potatoes and peas" or "How to make dosa?"

🧾 Step-by-step instructions: Output includes ingredients, preparation steps, and cooking time.

📚 Dataset-driven: Built using a well-structured dataset of Indian recipes including ingredients, categories, and preparation instructions.
