import gensim
import atire



atire.init("atire")


input = []
COLLECTION_SIZE = 173251



for i in range(0, COLLECTION_SIZE):
    input.append(atire.get_ordered_tokens(i))

model = gensim.models.Word2Vec(sentences=input, size=300,window=5,workers=4,min_count=1,iter=30)
model.save('wsj-collection-vectors')

# TODO: Re-train Word2Vec model, it's not getting numbers in at the moment.Do it Friday night. Train for 30 iterations
print(atire.get_ordered_tokens(63615))


print ("done")


loaded_models = gensim.models.Word2Vec.load("wsj-collection-vectors")
#
#
print (loaded_models.most_similar('money'))