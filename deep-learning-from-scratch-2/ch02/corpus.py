import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)
print(id_to_word)

def create_to_matrix(corpus, vocab_size, window_size=1):
  corpus_size = len(corpus)
  co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
  for idx, word_id in enumerate(corpus):
    for i in range(1, window_size + 1):
      left_idx = idx - i
      right_idx = idx + i

      if left_idx >= 0:
        left_word_id = corpus[left_idx]
        co_matrix[word_id, left_word_id] += 1
      
      if right_idx < corpus_size:
        right_word_id = corpus[right_idx]
        co_matrix[word_id, right_word_id] += 1
  return co_matrix

def cos_similarity(x, y, eps=1e-8):
  nx = x / np.sqrt(np.sum(x ** 2) + eps)
  ny = y / np.sqrt(np.sum(y ** 2) + eps)
  return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
  if query not in word_to_id:
    print('%s is not found' % query)
    return 
  
  print('\n[query] ' + query)
  query_id = word_to_id[query]
  query_vec = word_matrix[query_id]

  vocab_size = len(id_to_word)
  similarity = np.zeros(vocab_size)
  for i in range(vocab_size):
    similarity[i] = cos_similarity(word_matrix[i], query_vec)
  
  count = 0
  for i in (-1 * similarity).argsort():
    if id_to_word[i] == query:
      continue
    print(' %s:%s' % (id_to_word[i], similarity[i]))

    count += 1
    if count >= top:
      return

def ppmi(C, verbose=False, eps=1e-8):
  M = np.zeros_like(C, dtype=np.float32)
  N = np.sum(C)
  S = np.sum(C, axis=0)
  total = C.shape[0] * C.shape[1]
  cnt = 0

  for i in range(C.shape[0]):
    for j in range(C.shape[1]):
      pmi = np.log2(C[i, j] * N  / (S[i] * S[j]) + eps)
      M[i, j] = max(0, pmi)

      if verbose:
        cnt += 1
        if cnt % (total // 100 + 1) == 0:
          print('%.1f%% done' % (100 * cnt/total))
  return M

co_matrix = create_to_matrix(corpus, len(word_to_id))
print(co_matrix)
cos_similarities = np.zeros_like(co_matrix, dtype='f')
for i in range(len(co_matrix)):
  for j in range(i + 1, len(co_matrix)):
    cos_similarities[i, j] = cos_similarity(co_matrix[i], co_matrix[j])
    cos_similarities[j, i] = cos_similarity(co_matrix[i], co_matrix[j])
print(cos_similarities)
most_similar('you', word_to_id, id_to_word, co_matrix)
W = ppmi(co_matrix, verbose=True)
# print(W)

U, S, V = np.linalg.svd(W)
print(co_matrix[0])
print(W[0])
print(U[0])
print(U[0, :2])
print(S)
print(V)