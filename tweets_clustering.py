import re
import random

def tweet_clean(tweet_text):
    # - Remove the tweet id and timestamp
    parts = tweet_text.split('|')
    if len(parts) > 2:
        text = " ".join(parts[2:])
    else:
        text = tweet_text
    # - Remove any word that starts with the symbol @ e.g. @AnnaMedaris
    text = re.sub(r'@\w+', '', text)
    # - Remove any hashtag symbols e.g. convert #depression to depression
    text = text.replace('#','')
    # - Remove any URL
    text = re.sub(r'http\S+', '', text)
    # - Convert every word to lowercase
    text = text.lower()
    # Tokenize into a set of words (unordered set)
    words = set(re.findall(r'\w+', text))
    
    return words

def load_dataset(file_path):
    tweets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned = tweet_clean(line)
            if cleaned:
                tweets.append(cleaned)
    return tweets

def jaccard_dist(setA, setB):
    # fulmula -> 1 - (|A intersection B| / |A union B|)
    intersection = len(setA.intersection(setB))
    union = len(setA.union(setB))
    
    if union == 0:
        return 1
    
    return 1 - (intersection / union)


def kmeans(tweets, k, max_iters = 100):
    # 1. Randomly choose k initial centroids
    centroids = random.sample(tweets, k)
    
    for i in range(max_iters):
        # 2. Assignment Step: Create empty clusters
        clusters = [[] for _ in range(k)]
        
        for tweet in tweets:
            # Find the distance from this tweet to every centroid
            distances = [jaccard_dist(tweet, c) for c in centroids]
            # Assign tweet to the index of the closest centroid
            closest_index = distances.index(min(distances))
            clusters[closest_index].append(tweet)
            
        # 3. Update Step: Save old centroids to check for convergence
        prev_centroids = centroids.copy()
        
        # Logic for updating centroids: 
        # Find the tweet in the cluster that is "most central"
        # (Minimizes distance to all other tweets in the same cluster)
        # 3. Update Step: Find the "Medoid" for each cluster
        new_centroids = []
        for cluster in clusters:
            if not cluster:
                # If a cluster is empty, keep the old centroid or pick a new random one
                new_centroids.append(random.choice(tweets))
                continue
            
            # Find the tweet in the cluster that has the MINIMUM total distance 
            # to every other tweet in that same cluster
            min_total_dist = float('inf')
            best_tweet = cluster[0]
            
            for candidate in cluster:
                total_dist = sum(jaccard_dist(candidate, other) for other in cluster)
                if total_dist < min_total_dist:
                    min_total_dist = total_dist
                    best_tweet = candidate
            
            new_centroids.append(best_tweet)

        # Check for convergence (stop if centroids don't change)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return clusters, centroids
        

def calculate_sse(clusters, centroids):
    sse = 0
    for i in range(len(clusters)):
        for tweet in clusters[i]:
            # SSE = sum of squared distances between tweets and their centroid
            dist = jaccard_dist(tweet, centroids[i])
            sse += (dist ** 2)
    return sse
    

data = load_dataset('nytimeshealth.txt')
# print(f"Loaded {len(data)} tweets.")

k_values = [5, 10, 15, 20, 25]
for k in k_values:
    final_clusters, final_centroids = kmeans(data, k)
    sse = calculate_sse(final_clusters, final_centroids)
    
    print(f"K: {k} | SSE: {sse}")
    for idx, cluster in enumerate(final_clusters):
        print(f"  Cluster {idx+1}: {len(cluster)} tweets")