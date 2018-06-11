import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('~/atire/GoogleNews-vectors-negative300.bin', binary=True)


model.most_similar(positive=['woman', 'king'], negative=['man'])