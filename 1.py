# coding: utf-8

from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/8oehplrobcgi9cq/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.
    """
    ###TODO
    if (keep_internal_punct):
        return np.array(re.findall('[\w_][^\s]*[\w_]|[\w_]', doc.lower()))
    else:
        return np.array(re.findall('[\w_]+', doc.lower()))


def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
    """
    ###TODO
    for token in tokens:
        feats["token=" + token] += 1


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.
    """
    ###TODO
    combination = []
    for i in range(len(tokens)-k+1): 
        combination += [c for c in combinations(tokens[i:i+k],2)]
    for comb in combination:
        feats['token_pair='+comb[0]+'__'+comb[1]] +=1

neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    """
    ###TODO
    feats['neg_words'] = 0
    feats['pos_words'] = 0
    for token in tokens:
        token = token.lower()
        if token in pos_words:
                feats['pos_words'] += 1
        elif token in neg_words:
                feats['neg_words'] += 1 


def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.
    """
    ###TODO
    feats = defaultdict(lambda: 0)
    for i in feature_fns:
        i(tokens,feats)
    return sorted(feats.items())


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    When vocab is None, we build a new vocabulary from the given data.
    when vocab is not None, we do not build a new vocab, and we do not
    add any new terms to the vocabulary. This setting is to be used
    at test time.
    """
    ###TODO
    row = []
    col = []
    data = []
    feats = list(defaultdict(lambda: 0))
    c = Counter()
    for tokens in tokens_list:
        feat = dict(featurize(tokens,feature_fns))
        feats.append(feat)
        c.update(feat.keys())
    tmp = []
    for key,value in c.items():
        if value >=min_freq:              
            tmp.append(key)
    if vocab == None:
        vocab= dict(zip(list(sorted(tmp)), list(range(len(tmp)))))
    
    for key in vocab.keys():
        for Row in range(len(feats)):
            while key in feats[Row].keys():
                row.append(Row)
                col.append(vocab[key])
                data.append(feats[Row][key])
                break
    return csr_matrix((data, (row, col)), shape=(len(feats), len(vocab))) , vocab
   

def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO
    cv = KFold(n_splits=k)
    accuracies = []
    for train_ind, test_ind in cv.split(X):
        clf.fit(X[train_ind], labels[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(labels[test_ind], predictions))
    mean = np.mean(accuracies)
    return mean


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    ###TODO
    dict_list =[]
    combi = []
    for i in range(1, len(feature_fns)+1):
        for x in combinations(feature_fns, i):
            combi.append([*x])
            
    for pun in punct_vals:
        tokens_list = []
        for doc in docs:
            tokens_list.append(tokenize(doc,pun))
        for min_f in min_freqs:
          for com in combi:
            X, voc = vectorize(tokens_list, com, min_f)
            clf = LogisticRegression()
            accuracy = cross_validation_accuracy(clf, X, labels, 5)
            temp = {} 
            temp['punct'] = pun
            temp['features'] = com
            temp['min_freq'] = min_f
            temp['accuracy'] = accuracy
            dict_list.append(temp)
    result = sorted(dict_list,key=lambda x:(-x['accuracy'],-x['min_freq']))
    return result

def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    ###TODO
    result = sorted(results, key = lambda x : x['accuracy'])
    Y = []
    X = range(len(result))
    for i in result:
        y = i['accuracy']
        Y.append(y)
    plt.plot(X, Y, 'b-')   
    plt.xlabel('setting')
    plt.ylabel('accuracy')
    plt.savefig('accuracies.png')


def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    ###TODO
    dic = defaultdict(lambda: (.0,0))
    setting_accuracy = []
    for r in results:
        dic["min_freq="+str(r["min_freq"])] = (dic["min_freq="+str(r["min_freq"])][0]+r["accuracy"]\
                                ,dic["min_freq="+str(r["min_freq"])][1]+1)
        fea = ''
        for f in r["features"]:
          fea += " "+f.__name__
          # fea += "".join(f.__name__)
        dic["features="+fea.strip()] = (dic["features="+fea.strip()][0]+r["accuracy"],\
                                              dic["features="+fea.strip()][1]+1)
        dic["punct="+ str(r["punct"])] = (dic["punct="+ str(r["punct"])][0] + \
                                        r["accuracy"], dic["punct="+ str(r["punct"])][1] + 1)
    for key in dic.keys():
        setting_accuracy.append(((dic[key][0]/dic[key][1]),key))
    result = sorted(setting_accuracy,key=lambda x:-x[0])

    return  result  

def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    ###TODO
    token_list = []
    for doc in docs:
        token_list.append(tokenize(doc, best_result['punct']))
    X, vocab = vectorize(token_list, best_result['features'], best_result['min_freq'])
    clf = LogisticRegression()
    clf.fit(X, labels)
    return clf, vocab


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    ###TODO
    coef = clf.coef_[0]
    top_coef = []
    if label == 0:
        for fea in vocab:
            the_list = (fea,coef[vocab[fea]])
            top_coef.append(the_list)
        result = sorted(top_coef, key=lambda x: x[1])
        temp = []
        for i in result[:n]:
            temp.append((i[0], -1 * i[1]))

        return temp
        
    elif label == 1:
        for fea in vocab:
            the_list = (fea,coef[vocab[fea]])
            top_coef.append(the_list)
        result = sorted(top_coef,key=lambda x:-x[1])[:n]
        return result

def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO
    tokens_list = []
    docs_test, labels_test = read_data(os.path.join('data', 'test'))
    pun = best_result["punct"]
    fea = best_result["features"]
    minf = best_result["min_freq"]
    
    for doc in docs_test:
        tokens_list.append(tokenize(doc, pun))
    X_test, vocab_test = vectorize(tokens_list, fea, minf, vocab)
    return docs_test, labels_test, X_test


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    ###TODO
    predictions = clf.predict(X_test)
    predicted_prob = clf.predict_proba(X_test)
    incorrect_labels = np.where(predictions != test_labels)[0]
    incorrect_pred = predicted_prob[incorrect_labels]
    first_maximum = np.argsort(np.amax(incorrect_pred, axis = 1))[::-1][:n]
    for i in range(n):
        doc = incorrect_labels[first_maximum[i]]
        print("\n"+'truth=' + str(test_labels[doc]) + ' predicted=' + str(predictions[doc]) + \
              ' proba=' + str(np.max(predicted_prob[doc])))
        print(test_docs[doc])

def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()
