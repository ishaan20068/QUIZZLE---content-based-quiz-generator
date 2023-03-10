import re
import string
import time
from similarity.normalized_levenshtein import NormalizedLevenshtein
from collections import OrderedDict
from sense2vec import Sense2Vec

def edits(token):
    '''
    takes in a string and returns all the possible strings with an edit distance of one 
    '''
    alphabets_and_punctuations = string.ascii_lowercase + " " + string.punctuation
    
    final_tokens = []
    for i in range(len(token)+1):    # splits token into all possible pairs of left and right substrings
        left = token[:i]
        right = token[i:]
        
        if len(right) != 0:     # list of words by deleting each non-empty right substring of a given input word
            fin_token = left + right[1:]
            final_tokens.append(fin_token)

        if len(right) > 1:      # list of words by swapping the adjacent character in every non-empty right substring of a given word
            fin_token = left + right[1] + right[0] + right[2:]
            final_tokens.append(fin_token)
        
        if len(right) != 0:     # list of words by replacing each character in every non-empty right substring of a given word
            for alpha_punct in alphabets_and_punctuations:
                fin_token = left + alpha_punct + right[1:]
                final_tokens.append(fin_token)
        
        for alphabet in alphabets_and_punctuations:     # list of words by inserting each character in every possible position of every non-empty right substring of a given word
            fin_token = left + alphabet + right
            final_tokens.append(fin_token)
    
    return set(final_tokens)


def sense2vec_get_words(token, sense):
    '''
    Takes a word and returns all the words in the database that are similar to the given word
    with the same sense. In case of multiple words it returns at most 15 words ordered in 
    descending order by their closeness. The best match word comes before.
    '''
    output = []
    token_preprocessed =  re.sub(r'[^\w\s]','', token).lower()
    token_edits = edits(token_preprocessed)
    token = token.replace(" ", "_")
    sense = sense.get_best_sense(token)

    if sense == None:
        return None, "None"
    
    most_similar = sense.most_similar(sense, n = 15)
    compare_list = [token_preprocessed]
    for word in most_similar:
        temp = word[0].split("|")[0].replace("_", " ")
        temp = temp.strip()
        temp = temp.lower()
        temp = re.sub(r'[^\w\s]','', temp)
        if temp not in compare_list and token_preprocessed not in temp and temp not in token_edits:
            output.append(temp.title())
            compare_list.append(temp)
    out = list(OrderedDict.fromkeys(output))
    print(out)

    return out, "sense2vec"


def sense(word):
    start = time.time()
    word_preprocessed =  re.sub(r'[^\w\s]','', word)
    print(word_preprocessed)
    end = time.time()
    print("The time of execution of above program is :", (end-start), "ms")


# sense("vwsvcsdj$vf, vervw,! fssd#@! vfvr; cvwc; ' edced")
# sense2vec_get_words("vwsvcsdj$vf, vervw,! fssd#@! vfvr; cvwc; ' edced")
string_token = "vwsvcsdj$vf, vervw,! fssd#@! vfvr; cvwc; ' edced"
s2v=Sense2Vec().from_disk("QUIZZLE---content-based-quiz-generator\s2v_old")
temp = sense2vec_get_words(string_token, s2v)
print(temp)
