import os
import math
from collections import defaultdict, Counter

def read_documents(folder_path):
    
    return {
        filename: open(os.path.join(folder_path, filename), 'r', encoding='utf-8').read()
        for filename in os.listdir(folder_path) if filename.endswith('.txt')
    }

def build_index(docs):
    
    postings = defaultdict(list)
    doc_lengths = {}
    doc_freq = defaultdict(int)
    
    for doc_id, text in docs.items():
        terms = text.lower().split()
        term_counts = Counter(terms)
        
        for term, freq in term_counts.items():
            postings[term].append((doc_id, freq))
            doc_freq[term] += 1
        
        length = math.sqrt(sum((1 + math.log(freq))**2 for freq in term_counts.values()))
        doc_lengths[doc_id] = length

    return postings, doc_freq, doc_lengths

def process_query(query, doc_freq, num_docs):
    
    terms = query.lower().split()
    term_counts = Counter(terms)
    
    query_vector = {
        term: (1 + math.log(count)) * math.log(num_docs / doc_freq[term])
        for term, count in term_counts.items() if term in doc_freq
    }
    
    length = math.sqrt(sum(tf_idf ** 2 for tf_idf in query_vector.values()))
    return {term: tf_idf / length for term, tf_idf in query_vector.items()}

def search(query_vector, postings, doc_lengths):
    
    scores = defaultdict(float)
    
    for term, weight in query_vector.items():
        for doc_id, tf in postings[term]:
            scores[doc_id] += weight * (1 + math.log(tf))
    
    return sorted(
        ((doc_id, score / doc_lengths[doc_id]) for doc_id, score in scores.items()),
        key=lambda x: (-x[1], x[0])
    )[:10]

def main():
    
    corpus_folder = '/Users/adityadoneria/Desktop/IR ASST/Corpus'
    
    docs = read_documents(corpus_folder)
    postings, doc_freq, doc_lengths = build_index(docs)
    
    query = "Warwickshire, came from an ancient family and was the heiress to some land"
    query_vector = process_query(query, doc_freq, len(docs))
    print(query)
    
    ranked_docs = search(query_vector, postings, doc_lengths)
    print("Ranked Document IDs:")
    for doc_id, score in ranked_docs:
        print(f"{doc_id}: {round(score, 17)}")

if __name__ == "__main__":
    main()
