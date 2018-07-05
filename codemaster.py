import numpy as np
import random
import gensim
from nltk.corpus import words
from functools import reduce
import heapq

wrds = np.genfromtxt('wordlist.csv', delimiter=',', dtype=str).tolist()

# randomly select 25 cards, then randomly select a side for each
board = [x[random.random() > 0.5].lower() for x in random.sample(wrds, 25)]

# set up game by assigning each card (agent) a role
player1 = board[:9]
player2 = board[9:17]
neutral = board[17:24]
assassin = board[24]

# load model pre-trained on Google News corpus (downloaded from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# choose candidate clues from nltk's words corpus, which itself is drawn from the UNIX words file
dct = words.words('en')

# returns potential clues which are closer to all members of targets than all members of avoids, sorted in order of relevance to targets
def clue (targets, avoids, n=1):
	topclues = []
	for word in dct:
		if word not in model.vocab:
			continue
		if avoids:
			target_distances = [model.similarity(word, target) for target in targets]
			avoid_distances = [model.similarity(word, avoid) for avoid in avoids]
			if max(avoid_distances) > min(target_distances):
				continue
		score = sum([model.similarity(word, target) for target in targets])
		if reduce((lambda x, y: y not in word and word not in y and x), targets, True):
			if len(topclues) < n:
				heapq.heappush(topclues, (score, word))
			elif score > topclues[0][0]:
				heapq.heapreplace(topclues, (score, word))
	return sorted(topclues, key=lambda x: x[0], reverse=True)


candidates = clue(board[:2], [], n=5)
print ('Top clues for pair', board[:2], ':', candidates)
