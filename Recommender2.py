#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
# from scipy.sparse.linalg import svds
# from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# In[4]:


originalData = pd.read_csv("clothingData.csv")


# ## setting up the product data

# In[5]:


df = originalData.dropna()


# In[6]:


df= df.drop(['gender','year','usage','season'],axis=1)


# In[7]:


df.head()


# In[8]:


duplicate = df[df['productName'].duplicated()]
duplicate


# In[9]:


df.shape


# In[10]:


# finding duplicate products
duplicates = df['productName'].duplicated(keep='first')
duplicates.sum()


# In[11]:


# selecting on those rows which has no duplicate productNames
result = df[~duplicates]
result.shape


# In[12]:


df = result.head(10000)


# ## setting up the image data

# In[13]:


imgData = pd.read_csv("images.csv")
imgData.head(2)


# In[14]:


imgData = imgData.drop_duplicates(subset='productName')


# In[15]:


imgData = imgData[['link', 'productName']]
result = result[['productName']]


# In[16]:


finalImageData = imgData.merge(result, on='productName', how='inner')
finalImageData


# In[17]:


# Create a text feature by combining multiple columns
df["text"] = ' ' + df['baseColour'] + ' ' + df['subCategory'] + ' ' + df['articleType']


# In[18]:


# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[19]:


cosine_sim[1]


# ## content based filtering

# In[20]:


def get_recommendations(product_name, article_type, base_color, cosine_sim=cosine_sim):
    # Create a text combining product name, article type, and base color
    text = f'{base_color} {article_type} {product_name}'

    # Get the indices for items matching the text
    matching_indices = df[df["text"] == text].index

    # Initialize an empty DataFrame to store similar items
    similar_items = pd.DataFrame(columns=['productName'])

    # Check for similar items based on article type
    for index in matching_indices:
        similar_items = similar_items.append(
            df[df['articleType'] == df.at[index, 'articleType']][['productName']], ignore_index=True)

    # Check for similar items based on base color
    for index in matching_indices:
        similar_items = similar_items.append(
            df[df['baseColour'] == df.at[index, 'baseColour']][['productName']], ignore_index=True)

    # If no matches are found based on article type or base color, search for similar product names
    if similar_items.empty:
        # Use text similarity or other matching techniques to find similar items
        product_tokens = product_name.lower().split()

        # Initialize a list to store matching product names
        matching_product_names = []

        # Loop through each product name in the dataset
        for index, row in df.iterrows():
            name_tokens = row['productName'].lower().split()

            # Calculate Jaccard similarity or other similarity metrics
            intersection = len(set(product_tokens) & set(name_tokens))
            union = len(set(product_tokens) | set(name_tokens))

            # Adjust the similarity threshold as needed
            similarity = intersection / union

            # If the similarity is above a threshold and it's not the input product, consider it a match
            if similarity > 0.5 and row['productName'] != product_name:  # Adjust the threshold as needed
                matching_product_names.append(row['productName'])

        # If matching products are found, return them as recommendations
        if matching_product_names:
            return matching_product_names

        # If no similar items are found, return a message indicating no recommendations
        return ["No recommendations found."]

    # Remove the input product name from the list of similar items
    similar_items = similar_items[similar_items['productName'] != product_name]

    # Return the top 10 unique product names as recommendations
    return similar_items['productName'].unique()[:10]


# In[40]:



# ## KNN based filtering

# In[41]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Function to get k-NN recommendations
def knn_recommender(user_product_name, article_type, base_colour, k=10):
    # Create a feature vector for the input product
    input_product = f'{base_colour} {article_type} {user_product_name}'
    
    # Fit a k-NN model on the TF-IDF matrix
    knn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    knn.fit(tfidf_matrix)
    
    # Transform the input product to TF-IDF representation
    input_tfidf = tfidf.transform([input_product])
    
    # Find the k-NN for the input product
    distances, indices = knn.kneighbors(input_tfidf, n_neighbors=k)
    
    # Get the recommended product indices
    recommended_indices = indices[0]
    
    # Get the recommended product names
    recommended_products = df['productName'].iloc[recommended_indices]
    
    return recommended_products




# In[46]:


def get_combined_recommendations(user_product_name, article_type, base_color, k=10):
    # Get recommendations using content-based filtering
    content_recommendations = get_recommendations(user_product_name, article_type, base_color, cosine_sim=cosine_sim)
    
    # Calculate how many recommendations are needed from k-NN to reach the total of 10
    k_nn_needed = max(k - len(content_recommendations), 1)  # Ensure k_nn_needed is at least 1
    
    # Get recommendations using k-NN
    knn_recommendations = knn_recommender(user_product_name, article_type, base_color, k=k_nn_needed)
    
    # Concatenate the results while ensuring a total of 10 recommendations
    combined_recommendations = list(content_recommendations) + list(knn_recommendations)
    
    return combined_recommendations



