"""
@author: 
    David J. Cox, PhD, MSB, BCBA-D
    dcox33@jhmi.edu
    https://www.researchgate.net/profile/David_Cox26
"""

# Packages!! 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk, re, pprint
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import string

#Set current working directory to the folder that contains your data.
# Home PC
os.chdir('C:/Users/coxda/Dropbox/Projects/Current Project Manuscripts/Empirical/NLP for Skinner\'s VB/Text Files')
# Work Mac
os.chdir('/Users/dcox/Dropbox/Projects/Current Project Manuscripts/Empirical/NLP for Skinner\'s VB/Text Files')
# Download from GitHub repository. 
read.txt(text = GET("https://github.com/davidjcox333/NLP_for_VerbalBehavior/blob/master/(1957)%20Verbal%20Behavior%2C%20Skinner.txt"))

# Show current working directory. 
dirpath = os.getcwd()
print(dirpath)

# Import Skinner's Verbal Behavior.
VB = open('(1957) Verbal Behavior, Skinner.txt', 'r', encoding='latin-1')
VB_raw = VB.read()
len(VB_raw) # Length of VB. 
VB_raw[:100] # First 100 characters of the book. 

# Tokenize: break up the text into words and punctuation. 
nltk.download() # We need to download the 'punkt' model from nltk. 
tokens_word = nltk.word_tokenize(VB_raw)
tokens_word[:50] # Compare with the above! Now we're working with words and punctuation as the unit. 
len(tokens_word) # How many words and punctuation characters do we have?


# Let's find the slice that contains just the text itself, and not the foreword or notes at the end. 
    # We know the first part begins with "Part I A Program"
VB_raw.find("Part I\n A PROGRAM") # \n is the symbol for the return sign to start a new line. 
VB_raw[47046:47080] # Print off some of the text that follows to make sure we've identified it correctly. 

    # We also know the book ends with a sentence that comprises a paragraph. That final sentence is, "The study of the verbal 
    # behavior of speaker and listener, as well as of the practices of the verbal environment which generates such behavior, 
    # may not contribute directly to historical or descriptive linguistics, but it is enough for our present purposes 
    # to be able to say that a verbal environment could have arisen from nonverbal sources and, in its transmission 
    # from generation to generation, would have been subject to influences which might account for the multiplication of 
    # forms and controlling relations and the increasing effectiveness of verbal behavior as a whole."
VB_raw.find("the increasing effectiveness of verbal behavior as a whole.") # Where does this phrase start?
VB_raw[1167958:1168017] # SLice to the end to catch this phrase. 
VB_raw[1167436:1168017] # And let's slice in the rest of the paragraph just to be sure there weren't a few sentnece with this phrase in the book. 

    # Great! Let's slice in the text of the book using the identified start and end points. 
VB_text = VB_raw[47046:1168017]
print(VB_text[:50])

# And let's clean this up for analysis by removing punctuation and making all the words lowercase. 
VB_nopunct = re.sub(r'[^\w\s]', '', VB_text)
VB_nopunct[:75]
VB_clean = [w.lower() for w in VB_nopunct]
VB_clean[:50]
VB_vocab = sorted(set(VB_clean))

# Tokenize this. 
tokens = nltk.word_tokenize(VB_text)
text = nltk.Text(tokens)
text[:50]
# Make all of the words lowercase. 
VB_clean=[w.lower() for w in text]
VB_clean[:50]
# Remove stop words. 
VB_stop_elim = [w for w in VB_clean if not w in stop_words]
VB_stop_elim[:50]
# Change punctuation to empty string. 
VB_stop_elim = [''.join(c for c in s if c not in string.punctuation) for s in VB_stop_elim]
VB_stop_elim[:50]
# Remove empty string. 
VB_filtered = [s for s in VB_stop_elim if s]
VB_filtered[:50]

''''''''''''''''''''''''''''''''''''''''''
# Now we can play! 
''''''''''''''''''''''''''''''''''''''''''
# How many times does Skinner mention different verbal operants. 
mand_count = VB_filtered.count('mand') + VB_filtered.count('mands') + VB_filtered.count('manded')
tact_count = VB_filtered.count('tact') + VB_filtered.count('tacts') + VB_filtered.count('tacted')
echoic_count = VB_filtered.count('echoic') + VB_filtered.count('echoics')
intraverbal_count = text.count('intraverbal')+ text.count('intraverbals')
textual_count = VB_filtered.count('textual') + VB_filtered.count('textuals')
transcription_count = VB_filtered.count('transcription') + VB_filtered.count('transcriptions')

# What are the context surrounding there use?
text.concordance('mand' or 'mands' or 'manded')
text.concordance('tact'or 'tacts' or 'tacted') # Same questions for 'tact'?

# Get the data ready for plotting. 
vb_operant_data = [mand_count, tact_count, echoic_count, intraverbal_count, textual_count, transcription_count]
bars = ('Mand', 'Tact', 'Echoic', 'Intraverbal', 'Textual', 'Transcription')
y_pos = np.arange(len(bars))

# PLot it. 
plt.bar(y_pos, vb_operant_data, color='black')
plt.xticks(y_pos, bars)
plt.ylabel('Count in Verbal Behavior (Skinner, 1957)')
plt.show()

# How about the same for speaker and listener?
speaker_count= VB_filtered.count('speaker') + VB_filtered.count('speakers')
listener_count = VB_filtered.count('listener') + VB_filtered.count('listeners')
sp_li_data = [speaker_count, listener_count]
bars = ('Speaker', 'Listener')
y_pos = np.arange(len(bars))
plt.bar(y_pos, sp_li_data, color='black', width = 0.6, align='center')
plt.xticks(y_pos, bars)
plt.ylabel('Count in Verbal Behavior (Skinner, 1957)')
plt.show()

# Can we do a wordcloud of the entire book?
from PIL import Image
from wordcloud import WordCloud

# Load a picture of Skinner's face that will be used for wordcloud shape. 
cloud_mask = np.array(Image.open("Skinner.jpg"))

# Create the wordcloud. 
VB_words = [w.lower() for w in VB_filtered]
VB_word_string = ' '.join(VB_words)
wordcloud = WordCloud(mask=cloud_mask).generate(VB_word_string) 

# Show wordcloud
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
    

# How about a frequency distribution of the top 50 words used in VB?
fdist = nltk.FreqDist(VB_filtered)
top_50 = fdist.keys()
fdist.plot(39, cumulative = True)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Let's do some NLP. Officially, a scatter plot of PCA projection of Word2Vec Model. 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
from nltk.corpus import stopwords

sentences = nltk.sent_tokenize(VB_text) # Break the text into sentences. 
sentences[:5] # Take a peek at the first 5 sentences. 
sent =[] # Create an empty list we'll put each of our word tokenized sentences into. 
stop_words = set(stopwords.words('english')) # Identify the stopwords to clean from text. 
for i in sentences: # Break each sentence into words, and then append the empty list we just created. 
    words = nltk.word_tokenize(i)
    stop_elim_sentences = [w for w in words if not w in stop_words]
    stop_elim_sentences_2 = [''.join(c for c in s if c not in string.punctuation) for s in stop_elim_sentences]
    filtered_sentences = [s for s in stop_elim_sentences_2 if s]
    sent.append(filtered_sentences)

sent[:10] # Take a peek at the first to make sure everything looks good. 

# Train model.
model = Word2Vec(sent, min_count=1)
# Summarize loaded model
print(model)
# Summarize vocabulary. 
NLP_words = list(model.wv.vocab)
print(NLP_words)
len(NLP_words)
# Access vector for one word. 
print(model['mand'])
# Save model
model.save('model.bin')
# Load model. 
new_model = Word2Vec.load('model.bin')
print(new_model)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Plot it up using PCA to create the 2-dimensions for plotting. 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from sklearn.decomposition import PCA

# Train model. 
model = Word2Vec(sent, min_count = 1)
# Fit a 2D PCA model to the vectors. 
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# Create scatter plot of the projection. 
plt.scatter(result[:, 0], result[:, 1], marker='')
plt.ylim(-.15, .065)
plt.xlim(-.5, 9)
plt.title('Scatter of PCA Projection of Word2Vec Model')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
words = list(model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Plot it up using PCA to create the 3-dimensions for plotting. 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from mpl_toolkits.mplot3d import axes3d, Axes3D
# Train model. 
model = Word2Vec(sent, min_count = 1)
# Fit a 3D PCA model to the vectors. 
X = model[model.wv.vocab]
pca = PCA(n_components=3)
pca.fit(X)
result = pd.DataFrame(pca.transform(X), columns=['PCA%i' % i for i in range(3)])

# Plot initialization. 
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], c='black', cmap="Set2_r", s=60)
for i, word in enumerate(words):
    ax.annotate(word, (result['PCA1'][i], result['PCA0'][i]))

# Simple bare axis lines through space:
xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0,0), (0, 0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
yAxisLine = ((0,0), (min(result['PCA1']), max(result['PCA1'])), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
zAxisLine = ((0,0), (0,0), (min(result['PCA2']), max(result['PCA2'])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

#label the axes
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.xaxis(-1, 10)
ax.yaxis(-5, 5)
ax.zaxis(-10, 10)
ax.set_title("3D Scatter of PCA of Word2Vec Model")
words = list(model.wv.vocab)
fig


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
That's messy. So, let's use Latent Dirichlet Allocation (LDA) 
to plot all of the words grouped into 20 topics using machine learning. 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_features=10000)
X = vect.fit_transform(words)

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_topics=20, learning_method="batch", max_iter=50, random_state=3)
document_topics = lda.fit_transform(X)

print("lda.componenets_.shape: {}".format(lda.components_.shape))

sorting=np.argsort(lda.components_, axis=1)[:, ::-1]
feature_names=np.array(vect.get_feature_names())

import mglearn
mglearn.tools.print_topics(topics=range(20), feature_names=feature_names, sorting=sorting, topics_per_chunk=5, n_words=10)

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
topic_names = ["{:>2} ".format(i) + " ".join(words) \
               for i, words in enumerate(feature_names[sorting[:, :2]])]
for col in [0, 1]:
    start = col * 10
    end = (col +1) * 10
    ax[col].barh(np.arange(10), np.sum(document_topics, axis=0)[start:end], color='black')
    ax[col].set_yticks(np.arange(10))
    ax[col].set_yticklabels(topic_names[start:end], ha="left", va="top")
    ax[col].invert_yaxis()
    ax[col].set_xlim(0, 700)
    yax=ax[col].get_yaxis()
    yax.set_tick_params(pad=130)
plt.tight_layout()
plt.show()



# Use PCA to plot the resulting topics. 
# Train model. 
model = Word2Vec(feature_names, min_count = 1)
# Fit a 2D PCA model to the vectors. 
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# Create scatter plot of the projection. 
plt.scatter(result[:, 0], result[:, 1], marker='o', color='black')
plt.ylim(-1, 1)
plt.xlim(-1, 1)
plt.title('Scatter of PCA Projection of Word2Vec Model')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
words = list(topic_names)
for i, words in enumerate():
    plt.annotate(words, xy=(result[i, 0], result[i, 1]))
plt.show()






















    