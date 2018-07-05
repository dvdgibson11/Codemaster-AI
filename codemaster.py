import numpy as np
import random
import gensim

words = np.genfromtxt('wordlist.csv', delimiter=',', dtype=str).tolist()

# randomly select 25 cards, then randomly select a side for each
board = [x[random.random() > 0.5] for x in random.sample(words, 25)]

player1 = board[:9]
player2 = board[9:17]
neutral = board[17:24]
assassin = board[24]

