# Importing required libraries  
import os  
import json  
import numpy as np  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from typing import Dict, List, Tuple, Optional, Union  
import random  
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity  

# Langchain imports  
from langchain.chat_models import ChatOpenAI  
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain.vectorstores import FAISS  
from langchain.schema import Document  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.prompts import ChatPromptTemplate  
from langchain.chains import LLMChain  
from langchain.document_loaders import DirectoryLoader, TextLoader  

# Set up the LLM using Langchain  
chat_llm = ChatOpenAI(  
    model_name="gpt-3.5-turbo",  
    openai_api_base="https://api.chatanywhere.tech/v1",  
    openai_api_key="sk-bVWnGBsURxbZSojM6pU3Xmogfbiq6q835INsr1OLrHtBWmgy"  
)  

# Set up embedding model - using HuggingFaceEmbeddings instead of client.embeddings.create  
embedding_model = HuggingFaceEmbeddings(  
    model_name="BAAI/bge-small-en",   
    model_kwargs={'device': 'cpu'}  
)  

# Initialize vector store  
vector_store = None  

def load_documents(directory_path: str) -> List[str]:  
    """  
    Load all text documents from the specified directory.  
    
    Args:  
        directory_path (str): Path to the directory containing text files.  
        
    Returns:  
        List[str]: A list of strings, where each string is the content of a text file.  
    """  
    documents = []  
    for filename in os.listdir(directory_path):  
        if filename.endswith(".txt"):  
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:  
                documents.append(file.read())  
    return documents  

def split_into_chunks(documents: List[str], chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:  
    """  
    Split documents into smaller chunks for processing.  
    
    Args:  
        documents (List[str]): List of document texts to split.  
        chunk_size (int): Size of each chunk in characters.  
        chunk_overlap (int): Overlap between consecutive chunks.  
        
    Returns:  
        List[str]: List of text chunks.  
    """  
    text_splitter = RecursiveCharacterTextSplitter(  
        chunk_size=chunk_size,  
        chunk_overlap=chunk_overlap,  
        length_function=len,  
    )  
    
    chunks = []  
    for doc in documents:  
        doc_chunks = text_splitter.split_text(doc)  
        chunks.extend(doc_chunks)  
    
    return chunks  

def preprocess_text(text: str) -> str:  
    """  
    Preprocess text by removing extra whitespace, lowercasing, etc.  
    
    Args:  
        text (str): Text to preprocess.  
        
    Returns:  
        str: Preprocessed text.  
    """  
    # Remove extra whitespace and lowercase the text  
    processed_text = ' '.join(text.split()).lower()  
    return processed_text  

def preprocess_chunks(chunks: List[str]) -> List[str]:  
    """  
    Preprocess all chunks in the list.  
    
    Args:  
        chunks (List[str]): List of text chunks to preprocess.  
        
    Returns:  
        List[str]: List of preprocessed text chunks.  
    """  
    return [preprocess_text(chunk) for chunk in chunks]  

def generate_embeddings_batch(chunks_batch: List[str]) -> List[List[float]]:  
    """  
    Generate embeddings for a batch of text chunks.  
    
    Args:  
        chunks_batch (List[str]): A batch of text chunks to embed.  
        
    Returns:  
        List[List[float]]: List of embeddings for each chunk.  
    """  
    embeddings = []  
    for chunk in chunks_batch:  
        # Use Langchain's embedding model to generate embeddings  
        embedding = embedding_model.embed_query(chunk)  
        embeddings.append(embedding)  
    return embeddings  

def generate_embeddings(chunks: List[str], batch_size: int = 10) -> np.ndarray:  
    """  
    Generate embeddings for all chunks with batching.  
    
    Args:  
        chunks (List[str]): List of text chunks to embed.  
        batch_size (int): Size of each batch for embedding generation.  
        
    Returns:  
        np.ndarray: NumPy array of embeddings.  
    """  
    all_embeddings = []  
    
    # Process chunks in batches  
    for i in range(0, len(chunks), batch_size):  
        batch = chunks[i:i + batch_size]  
        batch_embeddings = generate_embeddings_batch(batch)  
        all_embeddings.extend(batch_embeddings)  
    
    return np.array(all_embeddings)  

def save_embeddings(embeddings: np.ndarray, output_file: str) -> None:  
    """  
    Save embeddings to a JSON file.  
    
    Args:  
        embeddings (np.ndarray): Array of embeddings to save.  
        output_file (str): Path to the output file.  
        
    Returns:  
        None  
    """  
    with open(output_file, 'w', encoding='utf-8') as file:  
        json.dump(embeddings.tolist(), file)  

def add_to_vector_store(embeddings: np.ndarray, chunks: List[str]) -> None:  
    """  
    Add embeddings and their corresponding text chunks to the FAISS vector store.  
    
    Args:  
        embeddings (np.ndarray): Array of embeddings to add.  
        chunks (List[str]): List of text chunks corresponding to the embeddings.  
        
    Returns:  
        None  
    """  
    # Convert to Document objects for Langchain  
    documents = [Document(page_content=chunk) for chunk in chunks]  
    
    # Create FAISS index with these documents  
    global vector_store  
    vector_store = FAISS.from_documents(documents, embedding_model)  

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:  
    """  
    Compute cosine similarity between two vectors.  
    
    Args:  
        vec1 (np.ndarray): First vector.  
        vec2 (np.ndarray): Second vector.  
        
    Returns:  
        float: Cosine similarity score.  
    """  
    # Reshape vectors for sklearn's cosine_similarity  
    vec1_reshaped = vec1.reshape(1, -1)  
    vec2_reshaped = vec2.reshape(1, -1)  
    
    # Compute and return cosine similarity  
    return sklearn_cosine_similarity(vec1_reshaped, vec2_reshaped)[0][0]  

def similarity_search(query_embedding: np.ndarray, top_k: int = 5) -> List[str]:  
    """  
    Perform similarity search in the vector store and return the top_k most similar chunks.  
    
    Args:  
        query_embedding (np.ndarray): Query embedding to search with.  
        top_k (int): Number of top results to return.  
        
    Returns:  
        List[str]: List of the most similar text chunks.  
    """  
    # Using Langchain's similarity search  
    documents = vector_store.similarity_search_by_vector(  
        query_embedding,   
        k=top_k  
    )  
    return [doc.page_content for doc in documents]  

def retrieve_relevant_chunks(query_text: str, top_k: int = 5) -> List[str]:  
    """  
    Retrieve relevant document chunks for a query.  
    
    Args:  
        query_text (str): Query text to search for.  
        top_k (int): Number of top results to return.  
        
    Returns:  
        List[str]: List of relevant text chunks.  
    """  
    # Generate embedding for the query  
    query_embedding = embedding_model.embed_query(query_text)  
    
    # Perform similarity search with the query embedding  
    relevant_chunks = similarity_search(query_embedding, top_k)  
    
    return relevant_chunks  

def construct_prompt(query: str, context_chunks: List[str]) -> ChatPromptTemplate:  
    """  
    Construct a prompt template with context for the LLM.  
    
    Args:  
        query (str): User query.  
        context_chunks (List[str]): List of context chunks to include in the prompt.  
        
    Returns:  
        ChatPromptTemplate: Langchain chat prompt template.  
    """  
    # Combine context chunks into a single context string  
    context = "\n\n".join(context_chunks)  
    
    # Create a chat prompt template with system and user messages  
    prompt_template = ChatPromptTemplate.from_messages([  
        ("system", "You are a helpful AI assistant. Answer the question based on the context provided. "  
                  "If the context doesn't contain relevant information to answer the question, "  
                  "say that you don't have enough information to provide a good answer."),  
        ("user", f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")  
    ])  
    
    return prompt_template  

def generate_response(prompt_template, max_tokens: int = 512, temperature: float = 0.7) -> str:  
    """  
    Generate a response using Langchain's LLM Chain.  
    
    Args:  
        prompt_template: The prompt template to use.  
        max_tokens (int): Maximum number of tokens in the response.  
        temperature (float): Temperature for controlling randomness.  
        
    Returns:  
        str: Generated response.  
    """  
    # Create LLM chain with the prompt template and model  
    chain = LLMChain(  
        llm=chat_llm,  
        prompt=prompt_template  
    )  
    
    # Run the chain  
    response = chain.invoke({})  
    
    return response["text"]  

def basic_rag_pipeline(query: str) -> str:  
    """  
    Implement the basic Retrieval-Augmented Generation (RAG) pipeline using Langchain.  
    
    Args:  
        query (str): User query.  
        
    Returns:  
        str: Generated response.  
    """  
    # Step 1: Retrieve relevant chunks  
    relevant_chunks = retrieve_relevant_chunks(query)  
    
    # Step 2: Construct a prompt  
    prompt_template = construct_prompt(query, relevant_chunks)  
    
    # Step 3: Generate response  
    response = generate_response(prompt_template)  
    
    return response  

def calculate_reward(response: str, ground_truth: str) -> float:  
    """  
    Calculate reward based on response quality compared to ground truth.  
    
    Args:  
        response (str): Generated response.  
        ground_truth (str): Ground truth answer.  
        
    Returns:  
        float: Reward score.  
    """  
    # Generate embeddings for response and ground truth  
    response_embedding = embedding_model.embed_query(response)  
    ground_truth_embedding = embedding_model.embed_query(ground_truth)  
    
    # Calculate cosine similarity as reward  
    similarity = cosine_similarity(  
        np.array(response_embedding),  
        np.array(ground_truth_embedding)  
    )  
    
    return float(similarity)  

def define_state(query: str, context_chunks: List[str]) -> Dict[str, object]:  
    """  
    Define the state representation for reinforcement learning.  
    
    Args:  
        query (str): User query.  
        context_chunks (List[str]): Current context chunks.  
        
    Returns:  
        Dict[str, object]: State representation.  
    """  
    # Get query embedding  
    query_embedding = embedding_model.embed_query(query)  
    
    # Get context embeddings  
    context_embeddings = []  
    for chunk in context_chunks:  
        embedding = embedding_model.embed_query(chunk)  
        context_embeddings.append(embedding)  
    
    # Calculate average context embedding if there are any chunks  
    if context_embeddings:  
        avg_context_embedding = np.mean(context_embeddings, axis=0)  
    else:  
        # Use zero vector if no context chunks  
        avg_context_embedding = np.zeros_like(query_embedding)  
    
    # Calculate query-context similarity  
    query_context_similarity = cosine_similarity(  
        np.array(query_embedding),  
        np.array(avg_context_embedding)  
    ) if len(context_chunks) > 0 else 0.0  
    
    # Create state dictionary  
    state = {  
        "query": query,  
        "query_embedding": query_embedding,  
        "context_chunks": context_chunks,  
        "avg_context_embedding": avg_context_embedding,  
        "query_context_similarity": query_context_similarity,  
        "context_length": len(context_chunks),  
    }  
    
    return state  

def define_action_space() -> List[str]:  
    """  
    Define the action space for reinforcement learning.  
    
    Returns:  
        List[str]: List of possible actions.  
    """  
    return [  
        "rewrite_query",  # Rewrite the query for better retrieval  
        "expand_context", # Retrieve additional context chunks  
        "filter_context", # Filter the context to keep only relevant chunks  
        "generate_response"  # Generate the final response  
    ]  

class PolicyNetwork(nn.Module):  
    """  
    Neural network model for selecting actions based on state.  
    """  
    def __init__(self, input_size: int, output_size: int):  
        """  
        Initialize the policy network.  
        
        Args:  
            input_size (int): Size of input features.  
            output_size (int): Size of output (number of actions).  
        """  
        super(PolicyNetwork, self).__init__()  
        self.fc1 = nn.Linear(input_size, 128)  
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, output_size)  
        self.relu = nn.ReLU()  
        self.softmax = nn.Softmax(dim=0)  
    
    def forward(self, x):  
        """  
        Forward pass through the network.  
        
        Args:  
            x: Input tensor.  
            
        Returns:  
            Tensor: Output probabilities for each action.  
        """  
        x = self.relu(self.fc1(x))  
        x = self.relu(self.fc2(x))  
        x = self.fc3(x)  
        return self.softmax(x)  

def policy_network(state: Dict[str, object], action_space: List[str], model=None) -> str:  
    """  
    Select an action based on the current state using the policy network.  
    
    Args:  
        state (Dict[str, object]): Current state.  
        action_space (List[str]): List of possible actions.  
        model: Optional trained policy network model.  
        
    Returns:  
        str: Selected action.  
    """  
    # If no model is provided, use epsilon-greedy exploration  
    if model is None:  
        # With 20% probability, select a random action for exploration  
        if random.random() < 0.2:  
            return random.choice(action_space)  
        
        # Otherwise, use a heuristic policy for exploitation  
        query_context_sim = state["query_context_similarity"]  
        context_length = state["context_length"]  
        
        if context_length == 0:  
            # If no context yet, expand context  
            return "expand_context"  
        elif context_length < 3:  
            # If limited context, either expand or rewrite based on similarity  
            return "expand_context" if query_context_sim < 0.5 else "rewrite_query"  
        elif context_length >= 5:  
            # If too much context, filter it  
            return "filter_context"  
        else:  
            # Otherwise, generate response  
            return "generate_response"  
    else:  
        # Use the trained model to select an action  
        # Convert state to feature vector  
        features = np.concatenate([  
            state["query_embedding"],  
            state["avg_context_embedding"],  
            [state["query_context_similarity"]],  
            [state["context_length"]]  
        ])  
        
        # Convert to tensor  
        state_tensor = torch.FloatTensor(features)  
        
        # Get action probabilities from model  
        with torch.no_grad():  
            action_probs = model(state_tensor)  
        
        # Select action with highest probability  
        action_idx = torch.argmax(action_probs).item()  
        
        return action_space[action_idx]  

def rewrite_query(query: str, context_chunks: List[str], temperature: float = 0.3) -> str:  
    """  
    Use LLM to rewrite the query for better document retrieval.  
    
    Args:  
        query (str): Original query.  
        context_chunks (List[str]): Current context chunks.  
        temperature (float): Temperature for controlling randomness.  
        
    Returns:  
        str: Rewritten query.  
    """  
    context_text = ' '.join(context_chunks[:2]) if context_chunks else 'No context available yet'  
    
    # Create prompt template for query rewriting  
    rewrite_prompt = ChatPromptTemplate.from_messages([  
        ("system", "You are a query optimization assistant. Your task is to rewrite the given query to make it more effective "  
                  "for retrieving relevant information."),  
        ("user", f"Original query: {query}\n\nBased on the context retrieved so far:\n{context_text}\n\n"  
                "Rewrite the query to be more specific and targeted to retrieve better information.")  
    ])  
    
    # Create rewrite chain  
    rewrite_chain = LLMChain(llm=chat_llm, prompt=rewrite_prompt)  
    
    # Generate rewritten query  
    result = rewrite_chain.invoke({})  
    rewritten_query = result["text"].strip()  
    
    return rewritten_query  

def expand_context(query: str, current_chunks: List[str], top_k: int = 3) -> List[str]:  
    """  
    Expand the context by retrieving additional chunks.  
    
    Args:  
        query (str): User query.  
        current_chunks (List[str]): Current context chunks.  
        top_k (int): Number of additional chunks to retrieve.  
        
    Returns:  
        List[str]: Expanded list of context chunks.  
    """  
    # Optional: Rewrite query to better retrieve additional context  
    rewritten_query = rewrite_query(query, current_chunks)  
    
    # Retrieve additional chunks with rewritten query  
    additional_chunks = retrieve_relevant_chunks(rewritten_query, top_k)  
    
    # Combine with existing chunks, avoiding duplicates  
    expanded_chunks = current_chunks.copy()  
    for chunk in additional_chunks:  
        if chunk not in expanded_chunks:  
            expanded_chunks.append(chunk)  
    
    return expanded_chunks  

def filter_context(query: str, context_chunks: List[str]) -> List[str]:  
    """  
    Filter context to keep only the most relevant chunks.  
    
    Args:  
        query (str): User query.  
        context_chunks (List[str]): Current context chunks.  
        
    Returns:  
        List[str]: Filtered context chunks.  
    """  
    if not context_chunks:  
        return []  
    
    # Get query embedding  
    query_embedding = embedding_model.embed_query(query)  
    
    # Get embeddings for each chunk  
    chunk_embeddings = [embedding_model.embed_query(chunk) for chunk in context_chunks]  
    
    # Calculate relevance scores  
    relevance_scores = []  
    for chunk_embedding in chunk_embeddings:  
        score = cosine_similarity(  
            np.array(query_embedding),   
            np.array(chunk_embedding)  
        )  
        relevance_scores.append(score)  
    
    # Sort by relevance  
    sorted_chunks = [x for _, x in sorted(zip(relevance_scores, context_chunks), reverse=True)]  
    
    # Return top chunks  
    filtered_chunks = sorted_chunks[:min(5, len(sorted_chunks))]  
    return filtered_chunks  

def rl_step(  
    state: Dict[str, object],   
    action_space: List[str],   
    ground_truth: str,   
    model=None  
) -> Tuple[Dict[str, object], str, float, Optional[str]]:  
    """  
    Perform a single reinforcement learning step.  
    
    Args:  
        state (Dict[str, object]): Current state.  
        action_space (List[str]): List of possible actions.  
        ground_truth (str): Ground truth answer for reward calculation.  
        model: Optional trained policy network.  
        
    Returns:  
        Tuple: Containing:  
            - new_state (Dict[str, object]): New state after taking action.  
            - action (str): Action taken.  
            - reward (float): Reward received.  
            - response (Optional[str]): Generated response if action was 'generate_response'.  
    """  
    # Select action based on current state and policy  
    action = policy_network(state, action_space, model)  
    
    # Extract state components  
    query = state["query"]  
    context_chunks = state["context_chunks"]  
    response = None  
    
    # Execute the selected action  
    if action == "rewrite_query":  
        # Rewrite the query  
        rewritten_query = rewrite_query(query, context_chunks)  
        # Update context chunks with new query  
        new_context_chunks = retrieve_relevant_chunks(rewritten_query, top_k=5)  
        # Update state with new query and context  
        new_state = define_state(rewritten_query, new_context_chunks)  
        # No immediate reward for this action  
        reward = 0.0  
        
    elif action == "expand_context":  
        # Expand the context  
        expanded_chunks = expand_context(query, context_chunks)  
        # Update state with expanded context  
        new_state = define_state(query, expanded_chunks)  
        # Small reward for expanding context  
        reward = 0.1  
        
    elif action == "filter_context":  
        # Filter the context  
        filtered_chunks = filter_context(query, context_chunks)  
        # Update state with filtered context  
        new_state = define_state(query, filtered_chunks)  
        # Small reward for filtering context  
        reward = 0.1  
        
    elif action == "generate_response":  
        # Construct prompt  
        prompt_template = construct_prompt(query, context_chunks)  
        # Generate response  
        response = generate_response(prompt_template)  
        # Calculate reward based on response quality  
        reward = calculate_reward(response, ground_truth)  
        # No state update as this is a terminal action  
        new_state = state  
    
    else:  
        # Invalid action  
        new_state = state  
        reward = -0.1  # Penalize invalid actions  
    
    return new_state, action, reward, response  

def initialize_training_params() -> Dict[str, Union[float, int]]:  
    """  
    Initialize parameters for RL training.  
    
    Returns:  
        Dict[str, Union[float, int]]: Dictionary of training parameters.  
    """  
    return {  
        "learning_rate": 0.001,  
        "num_episodes": 20,  
        "discount_factor": 0.95,  
        "epsilon": 0.2,  # For epsilon-greedy exploration  
        "epsilon_decay": 0.95,  
        "min_epsilon": 0.01,  
    }  

def update_policy(  
    model: PolicyNetwork,   
    optimizer: optim.Optimizer,   
    state: Dict[str, object],   
    action: str,   
    reward: float,   
    action_space: List[str]  
) -> None:  
    """  
    Update policy based on reward.  
    
    Args:  
        model (PolicyNetwork): Policy network model.  
        optimizer (optim.Optimizer): Optimizer for model parameters.  
        state (Dict[str, object]): State when action was taken.  
        action (str): Action taken.  
        reward (float): Reward received.  
        action_space (List[str]): List of possible actions.  
        
    Returns:  
        None  
    """  
    # Convert state to feature vector  
    features = np.concatenate([  
        state["query_embedding"],  
        state["avg_context_embedding"],  
        [state["query_context_similarity"]],  
        [state["context_length"]]  
    ])  
    
    # Convert to tensor  
    state_tensor = torch.FloatTensor(features)  
    
    # Get action index  
    action_idx = action_space.index(action)  
    
    # Forward pass to get action probabilities  
    action_probs = model(state_tensor)  
    
    # Create a target distribution that increases probability of the selected action  
    target = torch.zeros_like(action_probs)  
    target[action_idx] = reward  # Set target probability based on reward  
    
    # Calculate loss - use MSE loss  
    loss = nn.MSELoss()(action_probs, target)  
    
    # Backpropagation and optimization  
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  

def track_progress(  
    episode: int,   
    rewards_history: List[float],   
    actions_history: List[List[str]]  
) -> None:  
    """  
    Track and print training progress.  
    
    Args:  
        episode (int): Current episode number.  
        rewards_history (List[float]): History of rewards.  
        actions_history (List[List[str]]): History of actions.  
        
    Returns:  
        None  
    """  
    # Calculate average reward over the last 5 episodes or all if less than 5  
    recent_rewards = rewards_history[-min(5, len(rewards_history)):]  
    avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0  
    
    # Count occurrences of each action in the latest episode  
    if actions_history:  
        action_counts = {}  
        for action in actions_history[-1]:  
            if action in action_counts:  
                action_counts[action] += 1  
            else:  
                action_counts[action] = 1  
    
        # Print progress information  
        print(f"Episode {episode}:")  
        print(f"  Average Reward (last 5): {avg_reward:.4f}")  
        print(f"  Actions in this episode: {action_counts}")  
        print("-" * 40)  

def training_loop(  
    query_text: str,   
    ground_truth: str,   
    params: Optional[Dict[str, Union[float, int]]] = None  
) -> Tuple[PolicyNetwork, List[float], List[List[str]], Optional[str]]:  
    """  
    Implement the training loop for RL-enhanced RAG.  
    
    Args:  
        query_text (str): Input query text.  
        ground_truth (str): Ground truth answer.  
        params (Optional[Dict[str, Union[float, int]]]): Training parameters.  
        
    Returns:  
        Tuple: Containing:  
            - model (PolicyNetwork): Trained policy network.  
            - rewards_history (List[float]): History of rewards.  
            - actions_history (List[List[str]]): History of actions taken.  
            - best_response (Optional[str]): Best response generated during training.  
    """  
    # Initialize training parameters if not provided  
    if params is None:  
        params = initialize_training_params()  
    
    # Initialize action space  
    action_space = define_action_space()  
    
    # Initialize policy network  
    input_size = embedding_model.model_kwargs.get('max_seq_length', 384) * 2 + 2  # query_emb + context_emb + similarity + context_length  
    output_size = len(action_space)  
    model = PolicyNetwork(input_size, output_size)  
    
    # Initialize optimizer  
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])  
    
    # Initialize variables to track progress  
    rewards_history = []  
    actions_history = []  
    best_response = None  
    best_reward = -1  
    
    # Get initial performance from the simple RAG pipeline for comparison  
    simple_response = basic_rag_pipeline(query_text)  
    simple_reward = calculate_reward(simple_response, ground_truth)  
    print(f"Simple RAG reward: {simple_reward:.4f}")  
    
    # Training loop  
    for episode in range(params["num_episodes"]):  
        # Reset the environment with the same query  
        context_chunks = retrieve_relevant_chunks(query_text)  
        state = define_state(query_text, context_chunks)  
        episode_reward = 0  
        episode_actions = []  
        
        # Episode loop  
        max_steps = 10  # Maximum steps per episode to prevent infinite loops  
        for step in range(max_steps):  
            # Take a step in the environment  
            state, action, reward, response = rl_step(state, action_space, ground_truth, model)  
            episode_actions.append(action)  
            
            # Update policy based on reward  
            update_policy(model, optimizer, state, action, reward, action_space)  
            
            # If response was generated, end the episode  
            if response:  
                episode_reward = reward  # Final reward for the episode  
                
                # Track best response  
                if reward > best_reward:  
                    best_reward = reward  
                    best_response = response  
                
                break  
        
        # Update tracking variables  
        rewards_history.append(episode_reward)  
        actions_history.append(episode_actions)  
        
        # Decay epsilon for exploration  
        params["epsilon"] = max(params["min_epsilon"], params["epsilon"] * params["epsilon_decay"])  
        
        # Track progress periodically  
        if episode % 5 == 0 or episode == params["num_episodes"] - 1:  
            track_progress(episode, rewards_history, actions_history)  
    
    # Compare with simple RAG  
    improvement = best_reward - simple_reward  
    print(f"\nTraining completed:")  
    print(f"Simple RAG reward: {simple_reward:.4f}")  
    print(f"Best RL-enhanced RAG reward: {best_reward:.4f}")  
    print(f"Improvement: {improvement:.4f} ({improvement * 100:.2f}%)")  
    
    return model, rewards_history, actions_history, best_response  

def evaluate_relevance(retrieved_chunks: List[str], ground_truth_chunks: List[str]) -> float:  
    """  
    Evaluate the relevance of retrieved chunks compared to ground truth chunks.  
    
    Args:  
        retrieved_chunks (List[str]): Retrieved text chunks.  
        ground_truth_chunks (List[str]): Ground truth text chunks.  
        
    Returns:  
        float: Relevance score.  
    """  
    if not retrieved_chunks or not ground_truth_chunks:  
        return 0.0  
    
    # Get embeddings for retrieved chunks  
    retrieved_embeddings = [embedding_model.embed_query(chunk) for chunk in retrieved_chunks]  
    
    # Get embeddings for ground truth chunks  
    ground_truth_embeddings = [embedding_model.embed_query(chunk) for chunk in ground_truth_chunks]  
    
    # Calculate all pairwise similarities  
    max_similarities = []  
    for gt_embedding in ground_truth_embeddings:  
        similarities = []  
        for ret_embedding in retrieved_embeddings:  
            similarity = cosine_similarity(  
                np.array(gt_embedding),  
                np.array(ret_embedding)  
            )  
            similarities.append(similarity)  
        # Take the maximum similarity for each ground truth chunk  
        max_similarities.append(max(similarities) if similarities else 0.0)  
    
    # Average the maximum similarities  
    avg_relevance = sum(max_similarities) / len(max_similarities) if max_similarities else 0.0  
    return float(avg_relevance)  

def evaluate_accuracy(responses: List[str], ground_truth_responses: List[str]) -> float:  
    """  
    Evaluate the accuracy of generated responses compared to ground truth responses.  
    
    Args:  
        responses (List[str]): Generated responses.  
        ground_truth_responses (List[str]): Ground truth responses.  
        
    Returns:  
        float: Accuracy score.  
    """  
    if not responses or not ground_truth_responses:  
        return 0.0  
    
    # Calculate similarity between each response and its corresponding ground truth  
    accuracies = []  
    for response, ground_truth in zip(responses, ground_truth_responses):  
        accuracy = calculate_reward(response, ground_truth)  
        accuracies.append(accuracy)  
    
    # Average the accuracies  
    avg_accuracy = sum(accuracies) / len(accuracies)  
    return float(avg_accuracy)  

def evaluate_response_quality(responses: List[str]) -> float:  
    """  
    Evaluate the quality of generated responses.  
    
    Args:  
        responses (List[str]): Generated responses.  
        
    Returns:  
        float: Quality score.  
    """  
    if not responses:  
        return 0.0  
    
    # Create a prompt to evaluate response quality  
    quality_prompt = ChatPromptTemplate.from_messages([  
        ("system", "You are an AI assistant tasked with evaluating the quality of text responses. "  
                  "Rate the given response on a scale of 0 to 1, where 0 is poor quality and 1 is excellent quality. "  
                  "Consider factors like relevance, coherence, accuracy, and completeness."),  
        ("user", f"Response to evaluate:\n\n{responses[0]}\n\nPlease provide a quality score between 0 and 1:")  
    ])  
    
    # Create quality evaluation chain  
    quality_chain = LLMChain(llm=chat_llm, prompt=quality_prompt)  
    
    # Generate quality score  
    result = quality_chain.invoke({})  
    
    # Parse the quality score  
    try:  
        # Extract the first number between 0 and 1 in the response  
        import re  
        matches = re.findall(r"0\.\d+|1\.0|1", result["text"])  
        quality_score = float(matches[0]) if matches else 0.5  
    except:  
        # Default to mid-range score if parsing fails  
        quality_score = 0.5  
    
    return quality_score  

def evaluate_rag_performance(  
    queries: List[str],   
    ground_truth_chunks: List[str],   
    ground_truth_responses: List[str]  
) -> Dict[str, float]:  
    """  
    Evaluate the performance of the RAG pipeline using relevance, accuracy, and response quality metrics.  
    
    Args:  
        queries (List[str]): List of query strings to evaluate.  
        ground_truth_chunks (List[str]): List of ground truth text chunks corresponding to the queries.  
        ground_truth_responses (List[str]): List of ground truth responses corresponding to the queries.  
        
    Returns:  
        Dict[str, float]: Dictionary containing the average relevance, accuracy, and quality scores.  
    """  
    # Initialize lists to store scores for each metric  
    relevance_scores = []  
    accuracy_scores = []  
    quality_scores = []  

    # Iterate through each query and its corresponding ground truth data  
    for query, ground_truth_chunk, ground_truth_response in zip(queries, ground_truth_chunks, ground_truth_responses):  
        # Retrieve relevant chunks for the query  
        retrieved_chunks = retrieve_relevant_chunks(query)  
        
        # Evaluate the relevance of the retrieved chunks compared to the ground truth chunk  
        relevance = evaluate_relevance(retrieved_chunks, [ground_truth_chunk])  
        relevance_scores.append(relevance)  

        # Generate a response using the basic RAG pipeline  
        response = basic_rag_pipeline(query)  
        
        # Evaluate the accuracy of the generated response compared to the ground truth response  
        accuracy = evaluate_accuracy([response], [ground_truth_response])  
        accuracy_scores.append(accuracy)  

        # Evaluate the quality of the generated response  
        quality = evaluate_response_quality([response])  
        quality_scores.append(quality)  

    # Calculate the average scores for each metric  
    avg_relevance = np.mean(relevance_scores)  
    avg_accuracy = np.mean(accuracy_scores)  
    avg_quality = np.mean(quality_scores)  

    # Return the average scores as a dictionary  
    return {  
        "average_relevance": avg_relevance,  
        "average_accuracy": avg_accuracy,  
        "average_quality": avg_quality  
    }  

def compare_rag_approaches(query_text: str, ground_truth: str) -> Tuple[str, str, float, float]:  
    """  
    Compare the outputs of simple RAG versus RL-enhanced RAG.  
    
    Args:  
        query_text (str): Input query text.  
        ground_truth (str): Ground truth answer.  
        
    Returns:  
        Tuple: Containing:  
            - simple_response (str): Response from simple RAG.  
            - best_rl_response (str): Best response from RL-enhanced RAG.  
            - simple_similarity (float): Similarity score of simple RAG response.  
            - rl_similarity (float): Similarity score of RL-enhanced RAG response.  
    """  
    print("=" * 80)  
    print(f"Query: {query_text}")  
    print("=" * 80)  
    
    # Generate response with simple RAG  
    simple_response = basic_rag_pipeline(query_text)  
    simple_similarity = calculate_reward(simple_response, ground_truth)  
    
    print("\nSimple RAG Output:")  
    print("-" * 40)  
    print(simple_response)  
    print(f"Similarity to ground truth: {simple_similarity:.4f}")  
    
    # Train RL-enhanced RAG  
    print("\nTraining RL-enhanced RAG model...")  
    params = initialize_training_params()  
    params["num_episodes"] = 5  # Reduce for demonstration  
    
    # Run training loop  
    model, rewards_history, actions_history, best_rl_response = training_loop(  
        query_text, ground_truth, params  
    )  
    
    # If no response was generated during training, generate one  
    if best_rl_response is None:  
        context_chunks = retrieve_relevant_chunks(query_text)  
        prompt_template = construct_prompt(query_text, context_chunks)  
        best_rl_response = generate_response(prompt_template)  
    
    # Calculate similarity score for RL-enhanced response  
    rl_similarity = calculate_reward(best_rl_response, ground_truth)  
    
    print("\nRL-enhanced RAG Output:")  
    print("-" * 40)  
    print(best_rl_response)  
    print(f"Similarity to ground truth: {rl_similarity:.4f}")  
    
    # Evaluation results  
    improvement = rl_similarity - simple_similarity  
    
    print("\nEvaluation Results:")  
    print("-" * 40)  
    print(f"Simple RAG similarity to ground truth: {simple_similarity:.4f}")  
    print(f"RL-enhanced RAG similarity to ground truth: {rl_similarity:.4f}")  
    print(f"Improvement: {improvement * 100:.2f}%")  
    
    # Plot reward history if possible  
    if len(rewards_history) > 1:  
        try:  
            import matplotlib.pyplot as plt  
            plt.figure(figsize=(10, 6))  
            plt.plot(rewards_history)  
            plt.title('Reward History During RL Training')  
            plt.xlabel('Episode')  
            plt.ylabel('Reward')  
            plt.grid(True)  
            plt.savefig('reward_history.png')  
            print("\nReward history plot saved to 'reward_history.png'")  
        except ImportError:  
            print("Matplotlib not available for plotting rewards")  
    
    return simple_response, best_rl_response, simple_similarity, rl_similarity  

# Example usage  
if __name__ == "__main__":  
    # Specify the directory path containing text files  
    directory_path = "Q&A System\data"  
    
    # Load documents  
    print("Loading documents...")  
    documents = load_documents(directory_path)  
    
    # Process documents  
    print("Processing documents...")  
    chunks = split_into_chunks(documents)  
    preprocessed_chunks = preprocess_chunks(chunks)  
    
    # Generate embeddings and add to vector store  
    print("Generating embeddings and building vector store...")  
    embeddings = generate_embeddings(preprocessed_chunks)  
    add_to_vector_store(embeddings, preprocessed_chunks)  
    
    # Load validation data  
    print("Loading validation data...")  
    with open('Q&A System\\data\\val.json', 'r',encoding='utf-8') as file:  
        validation_data = json.load(file)  
    
    # Test the RAG pipeline with a sample query  
    sample_query = validation_data['basic_factual_questions'][0]['question']  
    expected_answer = validation_data['basic_factual_questions'][0]['answer']  
    
    print(f"\nSample Query: {sample_query}")  
    print(f"Expected Answer: {expected_answer}\n")  
    
    # Compare simple RAG vs RL-enhanced RAG  
    print("\nComparing Simple RAG vs RL-enhanced RAG...")  
    simple_response, rl_response, simple_sim, rl_sim = compare_rag_approaches(sample_query, expected_answer)  
    
    # Save results  
    results = {  
        "query": sample_query,  
        "ground_truth": expected_answer,  
        "simple_rag": {  
            "response": simple_response,  
            "similarity": float(simple_sim)  
        },  
        "rl_rag": {  
            "response": rl_response,  
            "similarity": float(rl_sim)  
        },  
        "improvement": float(rl_sim - simple_sim)  
    }  
    
    with open('rl_rag_results.json', 'w') as f:  
        json.dump(results, f, indent=2)  
    
    print("\nResults saved to rl_rag_results.json")  