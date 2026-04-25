# K_Means_Jaccard_HealthCluster_ML
An implementation of the K-means unsupervised learning algorithm designed to cluster redundant and related health news tweets. By utilizing the Jaccard Distance metric, this project organizes raw text data into concise clusters to facilitate trend analysis and discovery within the healthcare domain.

# CS 4375 - ASSIGNMENT 3 (Part II)

### Project Overview
This project implements an unsupervised machine learning pipeline to cluster health-related news tweets from the **NYTimes Health** dataset. It utilizes a custom-built **K-means** algorithm and the **Jaccard Distance** metric to group similar text data based on word sets.

### Prerequisites & Libraries
The following standard Python libraries were used for data loading and pre-processing:
* `re`: For regular expression-based text cleaning.
* `random`: For initial centroid selection (seeding).

### How to Run the Code
1. Ensure the `nytimeshealth.txt` dataset is in the same directory as the script. 
2. Open your terminal or VS Code command prompt.
3. Run the script using Python:
   ```bash
   python tweets_clustering.py
   ```
4. The script will output the results for five different values of **K** (5, 10, 15, 20, and 25) directly to the console. 

### Data Pre-processing Steps
To ensure high-quality clusters, the following cleaning steps were performed on each tweet: [cite: 104]
* **Metadata Removal**: Stripped tweet IDs and timestamps.
* **Mentions**: Removed all words starting with the `@` symbol. 
* **Hashtags**: Removed the `#` symbol while preserving the keyword.
* **URLs**: Removed all web links and URLs.  
* **Normalization**: Converted all text to lowercase. 
* **Tokenization**: Converted tweets into unordered sets of words to facilitate Jaccard calculations.  

### Evaluation Metric
Performance is evaluated using the **Sum of Squared Error (SSE)**, defined as:  
$$SSE = \sum_{i=1}^{K} \sum_{x \in C_i} dist^2(m_i, x)$$
where $m_i$ is the centroid (medoid) of the $i^{th}$ cluster.  
