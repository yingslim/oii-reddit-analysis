# utils/analysis.py
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.text_processor import *
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS, TSNE
from matplotlib import dates as mdates
import plotly.graph_objects as go
import plotly.express as px

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


def analyze_vocabulary_df(df, text_column, min_freq=2):

    # Preprocess and concatenate text data
    texts = df[text_column].apply(preprocess_text)
    concatenated_text = ' '.join(texts)
    
    # Tokenize and vectorize text data
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
    
    # Gather summary statistics
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


def tfidf_analyze_subreddit_df(df, title_column='post_title', selftext_column='post_body', min_doc_freq=2, max_terms=1000, include_selftext=True):
    
    # Combine title and optionally selftext columns
    texts = [
        preprocess_text(row[title_column]) + (' ' + preprocess_text(row[selftext_column]) if include_selftext and pd.notna(row[selftext_column]) else '')
        for _, row in df.iterrows()
    ]
    
    freq_df, vocab_stats = analyze_vocabulary_df(pd.DataFrame({title_column: texts}), title_column, min_freq=min_doc_freq)
    

    vectorizer = TfidfVectorizer(max_features=max_terms, min_df=min_doc_freq, stop_words = stopwords.words('english'))
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    results = {
        "tfidf_matrix": tfidf_matrix,
        "feature_names": feature_names,
        "freq_df": freq_df,
        "vocab_stats": vocab_stats
    }
    
    return results




def generate_tfidf_matrix(texts, max_terms=1000, min_doc_freq=2):
    """
    Generate TF-IDF matrix and feature names from texts.
    """

    vectorizer = TfidfVectorizer(
        stop_words=stopwords.words('english'),
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


def plot_word_timeseries_df(df, terms, figsize=(12, 6), include_selftext=True):
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
    df['date'] =pd.to_datetime(df['post_datetime'], unit='s')

    
    daily_counts = {term: [] for term in terms}
    dates = sorted(df['date'].unique())

    #if df['post_body'] is NaN, put empty string:
    df['post_body'] = df['post_body'].fillna('')
    
    # Get vocabulary from all posts
    if include_selftext:
        print(df['post_title'])
        print(df['post_body'])
        all_text = ' '.join(df['post_title'] + ' ' + df['post_body'])
    else:
        all_text = ' '.join(df['post_title'])
        
    vocab = set(preprocess_text(all_text).split())
    
    # Validate terms
    invalid_terms = [term for term in terms if term not in vocab]
    if invalid_terms:
        raise ValueError(f"Terms not in vocabulary: {invalid_terms}")
    
    # Count terms per day
    for date in dates:
        day_posts = df[df['date'] == date]
        day_text = ' '.join(day_posts['post_title'] + ' ' + day_posts['post_body'])
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


def plot_word_timeseries_df_cat(df, terms_cat_df, figsize=(12, 6), include_selftext=True):
    """
    Plot time series for all given terms, with shaded colors based on category, starting from darker to lighter.
    
    Args:
        df: DataFrame with posts
        terms_cat_df: DataFrame with terms and their categories (e.g., 'P' or 'C')
        figsize: Tuple of figure dimensions
        include_selftext: Boolean, whether to include 'post_body' in the analysis
    
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    # Extract terms and their categories
    terms = terms_cat_df['term'].tolist()
    categories = terms_cat_df.set_index('term')['category'].to_dict()  # Dictionary mapping term -> category
    
    # Prepare date column and initialize daily counts
    df['date'] = pd.to_datetime(df['post_datetime'], unit='s')
    daily_counts = {term: [] for term in terms}
    dates = sorted(df['date'].unique())

    # Fill missing post_body values with an empty string
    df['post_body'] = df['post_body'].fillna('')

    # Get vocabulary from all posts
    if include_selftext:
        all_text = ' '.join(df['post_title'] + ' ' + df['post_body'])
    else:
        all_text = ' '.join(df['post_title'])
        
    vocab = set(preprocess_text(all_text).split())

    # Validate terms
    invalid_terms = [term for term in terms if term not in vocab]
    if invalid_terms:
        raise ValueError(f"Terms not in vocabulary: {invalid_terms}")
    
    # Count terms per day
    for date in dates:
        day_posts = df[df['date'] == date]
        day_text = ' '.join(day_posts['post_title'] + ' ' + day_posts['post_body'])
        words = preprocess_text(day_text).split()
        word_counts = Counter(words)
        
        for term in terms:
            daily_counts[term].append(word_counts.get(term, 0))
    
    # Define color shades for each category, from darker to lighter
    num_p_terms = sum(1 for term in terms if categories[term] == 'P')
    num_c_terms = len(terms) - num_p_terms

    # Reverse the linspace to go from darker (higher value) to lighter (lower value)
    cmap_p = plt.cm.Blues(np.linspace(1, 0.3, num_p_terms))  # Shades of blue for 'P'
    cmap_c = plt.cm.Reds(np.linspace(1, 0.3, num_c_terms))    # Shades of red for 'C'
    
    fig, ax = plt.subplots(figsize=figsize)
    color_index_p, color_index_c = 0, 0
    
    for term in terms:
        # Choose color shade based on category
        if categories[term] == 'P':
            color = cmap_p[color_index_p]
            color_index_p += 1
        else:
            color = cmap_c[color_index_c]
            color_index_c += 1
            
        # Plot with specific color for each term
        ax.plot(dates, daily_counts[term], marker='o', label=term, color=color)
    
    ax.set_title('Term Frequency Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig, ax

def plot_word_timeseries_df_cat_plotly_test(df, terms_cat_df, figsize=(12, 6), include_selftext=True):
    """
    Plot time series for all given terms, with shaded colors based on category, starting from darker to lighter.
    
    Args:
        df: DataFrame with posts
        terms_cat_df: DataFrame with terms and their categories (e.g., 'P' or 'C')
        figsize: Tuple of figure dimensions
        include_selftext: Boolean, whether to include 'post_body' in the analysis
    
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    # Extract terms and their categories
    terms = terms_cat_df['term'].tolist()
    categories = terms_cat_df.set_index('term')['category'].to_dict()  # Dictionary mapping term -> category
    
    # Prepare date column and initialize daily counts
    df['date'] = pd.to_datetime(df['post_datetime'], unit='s')
    daily_counts = {term: [] for term in terms}
    dates = sorted(df['date'].unique())

    # Fill missing post_body values with an empty string
    df['post_body'] = df['post_body'].fillna('')

    # Get vocabulary from all posts
    if include_selftext:
        all_text = ' '.join(df['post_title'] + ' ' + df['post_body'])
    else:
        all_text = ' '.join(df['post_title'])
        
    vocab = set(preprocess_text(all_text).split())

    # Validate terms
    invalid_terms = [term for term in terms if term not in vocab]
    if invalid_terms:
        raise ValueError(f"Terms not in vocabulary: {invalid_terms}")
    
    # Count terms per day
    for date in dates:
        day_posts = df[df['date'] == date]
        day_text = ' '.join(day_posts['post_title'] + ' ' + day_posts['post_body'])
        words = preprocess_text(day_text).split()
        word_counts = Counter(words)
        
        for term in terms:
            daily_counts[term].append(word_counts.get(term, 0))
    # Define color shades for each category, from darker to lighter
    num_p_terms = sum(1 for term in terms if categories[term] == 'P')
    num_c_terms = len(terms) - num_p_terms
    # Generate color shades
    cmap_p = px.colors.sequential.Greens[::-1][:num_p_terms]  # Shades of orange for 'P', reversed for darker to lighter
    cmap_c = px.colors.sequential.Oranges[::-1][:num_c_terms]   # Shades of blue for 'C', reversed for darker to lighter

    # Create Plotly figures
    fig_p = go.Figure()
    fig_c = go.Figure()
    color_index_p, color_index_c = 0, 0

    # Plot each term with its specific color
    for term in terms:
        if categories[term] == 'P':
            color = cmap_p[color_index_p]
            color_index_p += 1
            fig_p.add_trace(go.Scatter(
                x=pd.to_datetime(dates),
                y=daily_counts[term],
                mode='lines+markers',
                name=term,
                line=dict(color=color, width=2),  # Set line width
                marker=dict(color=color)
            ))
        else:
            color = cmap_c[color_index_c]
            color_index_c += 1
            fig_c.add_trace(go.Scatter(
                x=pd.to_datetime(dates),
                y=daily_counts[term],
                mode='lines+markers',
                name=term,
                line=dict(color=color, width=2),  # Set line width
                marker=dict(color=color)
            ))

    # Set plot layout for 'P' terms
    fig_p.update_layout(
        title={'text':"<span style='color:green;'><b>Political</b></span> Term Frequency Over Time r/China",'x': 0.5,
            'xanchor': 'center'
        },

        xaxis_title="Date",
        yaxis_title="Frequency",
        xaxis=dict(tickangle=45),
        legend_title="Political Terms",
        template="plotly_white",
        width = 1000
    )

    # Set plot layout for 'C' terms
    fig_c.update_layout(

        title={'text':"<span style='color:orange;'><b>Cultural</b></span> Term Frequency Over Time r/China",'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Date",
        yaxis_title="Frequency",
        xaxis=dict(tickangle=45),
        legend_title="Cultural Terms",
        template="plotly_white",
        width = 1000
    )


    fig_p.show()
    fig_c.show()


def plot_word_timeseries_df_cat_grouped(df, terms_cat_df, figsize=(12, 6), include_selftext=True):
    """
    Plot time series for given terms, grouped by category (P, C), with separate lines for each category.
    Args:
        df: DataFrame with posts
        terms_cat_df: DataFrame with terms and their categories (e.g., 'P' or 'C')
        figsize: Tuple of figure dimensions
        include_selftext: Boolean, whether to include 'post_body' in the analysis
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    # Extract terms and their categories
    terms = terms_cat_df['term'].tolist()
    categories = terms_cat_df.set_index('term')['category'].to_dict()  # Dictionary mapping term -> category
    
    # Prepare date column and initialize daily counts
    df['date'] = pd.to_datetime(df['post_datetime'], unit='s')
    dates = sorted(df['date'].unique())
    
    # Fill missing post_body values with an empty string
    df['post_body'] = df['post_body'].fillna('')
    
    # Get vocabulary from all posts
    if include_selftext:
        all_text = ' '.join(df['post_title'] + ' ' + df['post_body'])
    else:
        all_text = ' '.join(df['post_title'])
    vocab = set(preprocess_text(all_text).split())
    
    # Validate terms
    invalid_terms = [term for term in terms if term not in vocab]
    if invalid_terms:
        raise ValueError(f"Terms not in vocabulary: {invalid_terms}")
    
    # Count terms per day and group by category
    p_counts = {term: [] for term in terms if categories[term] == 'P'}
    c_counts = {term: [] for term in terms if categories[term] == 'C'}
    daily_word_counts = []

    for date in dates:
        day_posts = df[df['date'] == date]
        day_text = ' '.join(day_posts['post_title'] + ' ' + day_posts['post_body'])
        words = preprocess_text(day_text).split()
        word_counts = Counter(words)
        daily_word_counts.append(word_counts)
        
        for term, category in categories.items():
            if category == 'P':
                p_counts[term].append(word_counts.get(term, 0))
            elif category == 'C':
                c_counts[term].append(word_counts.get(term, 0))
    
    # Sum counts for each category per day
    p_daily_counts = [sum(p_counts[term][i] for term in p_counts) for i in range(len(dates))]
    c_daily_counts = [sum(c_counts[term][i] for term in c_counts) for i in range(len(dates))]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(dates, p_daily_counts, label='Political Terms', color='blue', linewidth=2.5)
    ax.plot(dates, c_daily_counts, label='Cultural Terms', color='red', linewidth=2.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))


    # Find the two highest peak days for P and C
    max_p_days_idx = np.argsort(p_daily_counts)[-2:]
    max_c_days_idx = np.argsort(c_daily_counts)[-2:]
    max_p_days = [dates[idx] for idx in max_p_days_idx]
    max_c_days = [dates[idx] for idx in max_c_days_idx]
    max_p_values = [p_daily_counts[idx] for idx in max_p_days_idx]
    max_c_values = [c_daily_counts[idx] for idx in max_c_days_idx]
    
    # Annotate the two highest peaks with the three most used political or cultural words
    for i in range(2):
        max_p_words = sorted(p_counts, key=lambda term: p_counts[term][max_p_days_idx[i]], reverse=True)[:2]
        max_c_words = sorted(c_counts, key=lambda term: c_counts[term][max_c_days_idx[i]], reverse=True)[:2]
        print(max_p_words)
        
        ax.annotate('\n'.join(max_p_words), xy=(max_p_days[i], max_p_values[i]), xytext=(max_p_days[i]-5, max_p_values[i] + 5),
                    arrowprops=dict(facecolor='blue', shrink=0.05),ha='center',va='bottom')
        ax.annotate('\n'.join(max_c_words), xy=(max_c_days[i], max_c_values[i]), xytext=(max_c_days[i]-5, max_c_values[i] + 5),
                    arrowprops=dict(facecolor='red', shrink=0.05),ha='center')
    
    ax.set_title('Term Frequency Over Time by Category')
    ax.set_xlabel('Date')
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Categories')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig, ax


import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from datetime import timedelta

def plot_word_timeseries_df_cat_grouped_test(df, terms_cat_df, figsize=(12, 6), include_selftext=True):
    """
    Plot time series for given terms, grouped by category (P, C), with separate lines for each category.
    Args:
        df: DataFrame with posts
        terms_cat_df: DataFrame with terms and their categories (e.g., 'P' or 'C')
        figsize: Tuple of figure dimensions
        include_selftext: Boolean, whether to include 'post_body' in the analysis
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    # Extract terms and their categories
    terms = terms_cat_df['term'].tolist()
    categories = terms_cat_df.set_index('term')['category'].to_dict()  # Dictionary mapping term -> category
    
    # Prepare date column and initialize daily counts
    df['date'] = pd.to_datetime(df['post_datetime'], unit='s')
    dates = sorted(df['date'].unique())
    
    # Fill missing post_body values with an empty string
    df['post_body'] = df['post_body'].fillna('')
    
    # Get vocabulary from all posts
    if include_selftext:
        all_text = ' '.join(df['post_title'] + ' ' + df['post_body'])
    else:
        all_text = ' '.join(df['post_title'])
    vocab = set(preprocess_text(all_text).split())
    
    # Validate terms
    invalid_terms = [term for term in terms if term not in vocab]
    if invalid_terms:
        raise ValueError(f"Terms not in vocabulary: {invalid_terms}")
    
    # Count terms per day and group by category
    p_counts = {term: [] for term in terms if categories[term] == 'P'}
    c_counts = {term: [] for term in terms if categories[term] == 'C'}
    daily_word_counts = []

    for date in dates:
        day_posts = df[df['date'] == date]
        day_text = ' '.join(day_posts['post_title'] + ' ' + day_posts['post_body'])
        words = preprocess_text(day_text).split()
        word_counts = Counter(words)
        daily_word_counts.append(word_counts)
        
        for term, category in categories.items():
            if category == 'P':
                p_counts[term].append(word_counts.get(term, 0))
            elif category == 'C':
                c_counts[term].append(word_counts.get(term, 0))
    
    # Sum counts for each category per day
    p_daily_counts = [sum(p_counts[term][i] for term in p_counts) for i in range(len(dates))]
    c_daily_counts = [sum(c_counts[term][i] for term in c_counts) for i in range(len(dates))]

    # Create Plotly figure
    fig = go.Figure()

    # Plot the main line plots for political and cultural terms
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(dates),
        y=p_daily_counts,
        mode='lines',
        name='Political Terms',
        line=dict(color='#3F8D42', width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(dates),
        y=c_daily_counts,
        mode='lines',
        name='Cultural Terms',
        line=dict(color='#F9A34E', width=1.5)    ))
        # Set plot layout
    fig.update_layout(
        title={
            'text': "Term Frequency Over Time Grouped by <span style='color:green;'><b>Political</b></span> and <span style='color:orange;'><b>Cultural</b></span> Categories",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Date",
        yaxis_title="Frequency",
        xaxis=dict(
            tickangle=45,
            tickformat='%Y-%m-%d',  # Set date format
            dtick="D1"  # Tick every day
        ),
        legend=dict(
            title='Categories',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02  # Position outside plot area
        ),
        template="plotly_white",
        width=1000
    )
    fig.show()


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


def plot_similarities(tfidf_matrix, labels, 
                      title="term document plot", 
                        method='tsne', is_documents=True, label_color=False,
                      top_terms=None, figsize=(12, 8)):
    """
    Create projection visualization of document or term similarities
    
    Parameters:
    - tfidf_matrix: scipy sparse matrix
    - labels: list of labels (document texts or terms)
    - title: plot title
    - method: 'tsne' or 'mds' for dimensionality reduction
    - top_terms: if int, only annotate top n terms
    - is_documents: if True, plot documents, else plot terms
    - figsize: tuple for figure size
    """

    # Convert to dense array and transpose if visualizing terms
    matrix = tfidf_matrix.toarray()
    if not is_documents:
        matrix = matrix.T
    
    # Dimensionality reduction method
    if method == 'tsne':
        tsne = TSNE(n_components=2, 
                    perplexity=min(30, len(labels)-1),
                    random_state=42)
        coords = tsne.fit_transform(matrix)
    elif method == 'mds':
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        distances = 1 - cosine_similarity(matrix)
        coords = mds.fit_transform(distances)
    else:
        raise ValueError("Method must be 'tsne' or 'mds'") 
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
    
    # Add labels
    if top_terms and isinstance(top_terms, int):
        mean_tfidf = tfidf_matrix.mean(axis=0).A1 if is_documents else tfidf_matrix.mean(axis=1).A1
        top_indices = mean_tfidf.argsort()[-top_terms:][::-1]
        labels_to_annotate = [labels[i] for i in top_indices]
        coords_to_annotate = coords[top_indices]
    else:
        labels_to_annotate = labels
        coords_to_annotate = coords

    if label_color:
        unique_labels = list(set(labels_to_annotate))
        color_map = {label: color for label, color in zip(unique_labels, plt.cm.rainbow(np.linspace(0, 1, len(unique_labels))))}
        colors = [color_map[label] for label in labels_to_annotate]
    else:
        colors = ['black'] * len(labels_to_annotate)
    
    for i, (label, color) in enumerate(zip(labels_to_annotate, colors)):
        # Split long labels for documents
        if is_documents:
            label = split_label(label, 20)
            
        ax.annotate(label, (coords_to_annotate[i, 0], coords_to_annotate[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8 if is_documents else 12, alpha=0.7, color=color)
    
    
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig, ax

def plot_subreddit_term_space(vectors, term1, term2, title=None):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Plot vectors from origin
    colors = ['blue', 'green', 'red']
    for (name, vec), color in zip(vectors.items(), colors):
        plt.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1,
                  color=color, label=name, width=0.008)
    

    # Prepare all vectors data
    all_values = np.concatenate([v for v in vectors.values()])
    max_val = np.max(all_values) * 1.2

    # Create the plotly figure
    fig = go.Figure()

    # Plot each vector with an arrow (scatter plot with annotations for arrowheads)
    for label, vector in vectors.items():
        fig.add_trace(go.Scatter(
            x=[0, vector[0]],
            y=[0, vector[1]],
            mode='lines+markers+text',
            marker=dict(size=10,symbol='triangle-right'),
            line=dict(width=4),
            name=label,
            textposition="top center"
        ))

    # Set x and y limits
    fig.update_xaxes(
        range=[-0.1, max_val],
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='black',
        showgrid=True,
        gridcolor="gray",
        gridwidth=0.5,
        tickformat=".2f"
    )

    fig.update_yaxes(
        range=[-0.1, max_val],
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='black',
        showgrid=True,
        gridcolor="gray",
        gridwidth=0.5,
        tickformat=".2f"
    )

    # Set plot aspect ratio and layout
    fig.update_layout(
        title=title or f"Vectors in {term1}-<span style='color:orange;'>{term2}</span> Space (Normalized)",
        xaxis_title=f"'{term1}' Score",
        yaxis_title=f"'<span style='color:orange;'>{term2}</span>' Score",
        showlegend=True,
        legend_title="Vectors",
        template="plotly_white",
        autosize=False,
        width=600,
        height=600
    )

    fig.show()
    
def report_distances(vectors):
    """
    Report the distances between subreddit vectors.
    
    Parameters:
    - vectors: dict with subreddit names as keys and np.arrays as values
    """
    for name1, vec1 in vectors.items():
        for name2, vec2 in vectors.items():
            if name1 < name2:
                dist = np.linalg.norm(vec1 - vec2)
                print(f"Distance between {name1} and {name2}: {dist:.2f}")
                
    # Print angles between vectors
    print("\nAngles between subreddit vectors:")
    for name1, vec1 in vectors.items():
        for name2, vec2 in vectors.items():
            if name1 < name2:  # avoid duplicate comparisons
                cos_sim = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                angle = np.degrees(np.arccos(cos_sim))
                print(f"{name1} vs {name2}: {angle:.1f}°")

                
