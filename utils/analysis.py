# utils/analysis.py
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.text_processor import preprocess_text
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.manifold import TSNE


def analyze_vocabulary(texts, min_freq=2):
    """
    Analyze vocabulary distribution in a corpus.
    Returns word frequencies and vocabulary statistics.
    """
    
    # Preprocess texts
    texts = [preprocess_text(text) for text in texts]
    concatenated_text = ' '.join(texts)
    # Tokenize all texts
    
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(texts)
    words = vectorizer.get_feature_names_out()
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Calculate vocabulary statistics
    total_words = len(words)
    unique_words = len(word_freq)
    
    # Create frequency distribution DataFrame
    freq_df = pd.DataFrame(list(word_freq.items()), columns=['word', 'frequency'])
    freq_df['percentage'] = freq_df['frequency'] / total_words * 100
    freq_df = freq_df.sort_values('frequency', ascending=False)
    
    # Calculate cumulative coverage
    freq_df['cumulative_percentage'] = freq_df['percentage'].cumsum()
    
    stats = {
        'total_words': total_words,
        'unique_words': unique_words,
        'words_min_freq': sum(1 for freq in word_freq.values() if freq >= min_freq),
        'coverage_top_1000': freq_df.iloc[:1000]['frequency'].sum() / total_words * 100 if len(freq_df) >= 1000 else 100
    }
    
    return freq_df, stats


def tfidf_analyze_subreddit(posts, max_terms=1000, min_doc_freq=2, include_selftext=False):
    """
    Analyze a single subreddit's posts independently.
    """
    # Combine title and optionally selftext
    texts = [
        preprocess_text(post.get('title', '')) + (' ' + preprocess_text(post.get('selftext', '')) if include_selftext else '')
        for post in posts
    ]
    
    # Analyze vocabulary first
    freq_df, vocab_stats = analyze_vocabulary(texts, min_freq=min_doc_freq)
    # Generate TF-IDF matrix and feature names
    tfidf_matrix, feature_names = generate_tfidf_matrix(texts, max_terms, min_doc_freq)
    
    # Create results object from the matrix and feature names
    results = {
        "tfidf_matrix": tfidf_matrix, 
        "feature_names": feature_names, 
        "freq_df":freq_df, 
        "vocab_stats":vocab_stats}
    
    return results


def generate_tfidf_matrix(texts, max_terms=1000, min_doc_freq=2):
    """
    Generate TF-IDF matrix and feature names from texts.
    """
    stop_words = list(set(stopwords.words('english')))
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=max_terms,
        min_df=min_doc_freq
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return tfidf_matrix, feature_names


def create_posts_dataframe(posts):
    """
    Create DataFrame from Reddit posts with key metadata.
    """
    df = pd.DataFrame([{
        'title': post.get('title'),
        'selftext': post.get('selftext'),
        'url': post.get('url'),
        'domain': post.get('domain'),
        'time': datetime.fromtimestamp(post.get('created_utc', 0)),
        'author': post.get('author')
    } for post in posts])
    return df

def get_mean_tfidf(tfidf_matrix, feature_names=None, return_df=True):
    """
    Calculate mean TF-IDF score for each term in the matrix.
    """
    
    mean_tfidf = tfidf_matrix.mean(axis=0).tolist()[0]

    tfidf_scores = list(zip(feature_names, mean_tfidf))

    tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

    if return_df:
        df = pd.DataFrame(tfidf_scores, columns=['term', 'score'])
        df.set_index('term', inplace=True)
        return df

    return tfidf_scores

def create_report(tfidf_matrix, feature_names, freq_df, vocab_stats):
    """
    Create results object from TF-IDF matrix and feature names.
    """
    
    return {
        'vocab_stats': vocab_stats,
        'freq_distribution': freq_df,
        'tf_idf_scores': get_mean_tfidf(tfidf_matrix, feature_names, return_df=True),
        'vectorizer': None,  # Vectorizer is not needed in the results
        'matrix_shape': tfidf_matrix.shape,
        'matrix_sparsity': 100 * (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
    }

def get_top_terms(tfidf_results, n_terms=5):
    """
    Get top terms from TF-IDF results.
    
    Args:
        tfidf_results: Dictionary of term-tfidf scores
        n_terms: Number of top terms to return
    Returns:
        list: Top n terms
    """
    
    
    if isinstance(tfidf_results, pd.DataFrame):
        tfidf_scores_sorted = tfidf_results.sort_values('score', ascending=False)
    elif isinstance(tfidf_results, (pd.Series, dict)):
        tfidf_scores_sorted = pd.Series(tfidf_results).sort_values(ascending=False)
    else:
        raise ValueError("tfidf_results must be DataFrame, Series or dict")
    return tfidf_scores_sorted.head(n_terms).index.tolist()
    
def plot_word_timeseries(df, terms, figsize=(12, 6), include_selftext=False):
    """
    Plot time series for given terms.
    
    Args:
        df: DataFrame with posts
        terms: List of terms to plot
        figsize: Tuple of figure dimensions
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    # Prepare data
    df['date'] = pd.to_datetime(df['time']).dt.date
    daily_counts = {term: [] for term in terms}
    dates = sorted(df['date'].unique())
    
    # Get vocabulary from all posts
    if include_selftext:
        all_text = ' '.join(df['title'] + ' ' + df['selftext'])
    else:
        all_text = ' '.join(df['title'])
        
    vocab = set(preprocess_text(all_text).split())
    
    # Validate terms
    invalid_terms = [term for term in terms if term not in vocab]
    if invalid_terms:
        raise ValueError(f"Terms not in vocabulary: {invalid_terms}")
    
    # Count terms per day
    for date in dates:
        day_posts = df[df['date'] == date]
        day_text = ' '.join(day_posts['title'] + ' ' + day_posts['selftext'])
        words = preprocess_text(day_text).split()
        word_counts = Counter(words)
        
        for term in terms:
            daily_counts[term].append(word_counts.get(term, 0))
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    for term in terms:
        ax.plot(dates, daily_counts[term], marker='o', label=term)
    
    ax.set_title('Term Frequency Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig, ax



def plot_word_similarities_mds(tfidf_matrix, feature_names, n_terms=10, similarity_threshold=0.3, title=None):
    """
    Plot word similarities using MDS for a single TF-IDF matrix.
    
    Args:
        tfidf_matrix: scipy sparse matrix from TF-IDF vectorization
        feature_names: list of words corresponding to matrix columns
        n_terms: number of top terms to plot
        similarity_threshold: minimum similarity to draw connections
    
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    # Get top n terms based on mean TF-IDF scores
    mean_tfidf = tfidf_matrix.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[-n_terms:][::-1]
    
    # Get vectors for top terms
    term_vectors = tfidf_matrix.T[top_indices].toarray()
    top_terms = feature_names[top_indices]
    
    # Calculate similarities and distances
    similarities = cosine_similarity(term_vectors)
    distances = 1 - similarities
    
    # Use MDS for 2D projection
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distances)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(coords[:, 0], coords[:, 1])
    
    # Add word labels
    for i, term in enumerate(top_terms):
        ax.annotate(
            term, 
            (coords[i, 0], coords[i, 1]), 
            fontsize=16,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7),
            ha='center', va='center')
    
    # Draw lines between similar terms
    for i in range(len(top_terms)):
        for j in range(i+1, len(top_terms)):
            if similarities[i,j] > similarity_threshold:
                ax.plot([coords[i,0], coords[j,0]], 
                       [coords[i,1], coords[j,1]], 
                       'gray', alpha=0.3)
    if title: 
        ax.set_title(f'Word Similarities in {title}')
    else:
        ax.set_title('Word Similarities')
    plt.tight_layout()
    return fig, ax


def plot_word_similarities(tfidf_matrix, feature_names, n_terms=10, similarity_threshold=0.3, title=None):
    """
    Plot word similarities using MDS for a single TF-IDF matrix.
    
    Args:
        tfidf_matrix: scipy sparse matrix from TF-IDF vectorization
        feature_names: list of words corresponding to matrix columns
        n_terms: number of top terms to plot
        similarity_threshold: minimum similarity to draw connections
    
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    # Get top n terms based on mean TF-IDF scores
    mean_tfidf = tfidf_matrix.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[-n_terms:][::-1]
    
    # Get vectors for top terms
    term_vectors = tfidf_matrix.T[top_indices].toarray()
    top_terms = feature_names[top_indices]
    
    # Calculate similarities and distances
    similarities = cosine_similarity(term_vectors)
    distances = 1 - similarities
    
    # Use MDS for 2D projection
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distances)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(coords[:, 0], coords[:, 1])
    
    # Add word labels
    for i, term in enumerate(top_terms):
        ax.annotate(
            term, 
            (coords[i, 0], coords[i, 1]), 
            fontsize=12,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7),
            ha='center', va='center')
    
    # Draw lines between similar terms
    for i in range(len(top_terms)):
        for j in range(i+1, len(top_terms)):
            if similarities[i,j] > similarity_threshold:
                ax.plot([coords[i,0], coords[j,0]], 
                       [coords[i,1], coords[j,1]], 
                       'gray', alpha=0.3)
    if title: 
        ax.set_title(f'Word Similarities in {title}')
    else:
        ax.set_title('Word Similarities')
    plt.tight_layout()
    return fig, ax

def plot_word_similarities_tsne(tfidf_matrix, feature_names, n_highlight=5, perplexity=30, title=None):
    """
    Plot word similarities using t-SNE with all terms but highlighting top N.
    """
    # Get vectors for all terms
    term_vectors = tfidf_matrix.T.toarray()
    
    # Identify top terms
    mean_tfidf = tfidf_matrix.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[-n_highlight:][::-1]
    top_terms = feature_names[top_indices]
    
    # Calculate t-SNE for all terms
    tsne = TSNE(n_components=2, 
                perplexity=min(30, len(feature_names)/4), 
                random_state=42)
    coords = tsne.fit_transform(term_vectors)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot all points in light gray
    ax.scatter(coords[:, 0], coords[:, 1], 
              c='lightgray', alpha=0.5, s=30)
    
    # Highlight top terms
    ax.scatter(coords[top_indices, 0], coords[top_indices, 1], 
              c='red', s=100)
    
    # Add labels for top terms
    for i, term in enumerate(top_terms):
        ax.annotate(term, 
                   (coords[top_indices[i], 0], coords[top_indices[i], 1]),
                   fontsize=14,
                   bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7)
        )
    if title:
        ax.set_title(f'Word Similarities in {title} (Top {n_highlight} Terms Highlighted)')
    else:
        ax.set_title(f'Word Similarities (Top {n_highlight} Terms Highlighted)')
    plt.tight_layout()
    return fig, ax


