import numpy as np

def sample_topic_assignment(topic_assignment,
                            topic_counts,
                            doc_counts,
                            topic_N,
                            doc_N,
                            alpha,
                            gamma,
                            words,
                            document_assignment):
    """
    Sample the topic assignment for each word in the corpus, one at a time.
    
    Args:
        topic_assignment: size n array of topic assignments
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words        
        doc_counts: n_docs x n_topics array of counts per document of unique topics

        topic_N: array of size n_topics count of total words assigned to each topic
        doc_N: array of size n_docs count of total words in each document, minus 1
        
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.

        words: size n array of words
        document_assignment: size n array of assignments of words to documents
    Returns:
        topic_assignment: updated topic_assignment array
        topic_counts: updated topic counts array
        doc_counts: updated doc_counts array
        topic_N: updated count of words assigned to each topic
    """
    N = len(words)
    K = len(topic_N)
    W = topic_counts.shape[1]  

    for i in range(N):
        
        word = words[i]
        topic = topic_assignment[i]
        document = document_assignment[i]

        # decrement counts
        doc_counts[document, topic] -= 1
        topic_counts[topic, word] -= 1
        topic_N[topic] -= 1

        # compute parameter vector for conditional distribution
        n_dk_etc = doc_counts[document, :] + alpha
        n_kw_etc = topic_counts[:, word] + gamma
        n_k_etc = topic_N + gamma*W
        p = n_dk_etc*n_kw_etc/n_k_etc
        p /= np.sum(p)

        """
        # alternative computation... returns the same p
        numerator = (doc_counts[document, :] + alpha) * (topic_counts[:, word] + gamma)
        denominator = (K*alpha + doc_N[document]) * (topic_N + gamma*W)
        p2 = numerator/denominator
        p2 /= np.sum(p2)

        if not np.allclose(p, p2):
            print(p)
            print(p2)
        """
        
        # sample from distribution
        topic = np.argmax(np.random.multinomial(1, p))

        # increment counts and update topic assignment
        topic_assignment[i] = topic 
        doc_counts[document, topic] += 1
        topic_counts[topic, word] += 1
        topic_N[topic] += 1
        

    return topic_assignment, topic_counts, doc_counts, topic_N





