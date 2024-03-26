#!/usr/bin/env/python

import subprocess
import benchUtil
import constants

# Measure vector search recall and latency while exploring hyperparameters

# SETUP:
### Download and extract data files: Wikipedia line docs + GloVe
# python src/python/setup.py -download
# cd ../data
# unzip glove.6B.zip
# unlzma enwiki-20120502-lines-1k.txt.lzma
### Create document and task vectors
# ant vectors100
#
# then run this file: python src/python/knnPerfTest.py
#
# you may want to modify the following settings:


# Where the version of Lucene is that will be tested. Expected to be in the base dir above luceneutil.
#LUCENE_CHECKOUT = 'baseline'
LUCENE_CHECKOUT = 'candidate'

# test parameters. This script will run KnnGraphTester on every combination of these parameters
VALUES = {
    #'ndoc': (10000, 100000, 1000000),
    #'ndoc': (10000, 100000, 200000, 500000),
    'ndoc': (500_000, ),
    #'ndoc': (100000,),
    #'maxConn': (32, 64, 96),
    'maxConn': (16, ),
    #'beamWidthIndex': (250, 500),
    'beamWidthIndex': (200, ),
    #'fanout': (20, 100, 250)
    #'addTopK': (0,5,10,50,100),
    #'bits': (7,),
    #'filterSelectivity': (0.006, 0.0059, 0.0058, 0.0057, 0.0056, 0.0055, 0.0054, 0.0053),
    #'topK': (500,),
    #'niter': (1000,),
}

def advance(ix, values):
    for i in reversed(range(len(ix))):
        param = list(values.keys())[i]
        #print("advance " + param)
        if ix[i] == len(values[param]) - 1:
            ix[i] = 0
        else:
            ix[i] += 1
            return True
    return False

def run_knn_benchmark(checkout, values):
    indexes = [0] * len(values.keys())
    indexes[-1] = -1
    args = []
    #dim = 100
    #doc_vectors = constants.GLOVE_VECTOR_DOCS_FILE
    #query_vectors = '%s/luceneutil/tasks/vector-task-100d.vec' % constants.BASE_DIR
    #dim = 768
    #doc_vectors = '%s/data/enwiki-20120502-lines-1k-mpnet.vec' % constants.BASE_DIR
    #query_vectors = '%s/luceneutil/tasks/vector-task-mpnet.vec' % constants.BASE_DIR
    dim = 768
    doc_vectors = '%s/util/wiki768.train' % constants.BASE_DIR
    query_vectors = '%s/util/wiki768.test' % constants.BASE_DIR
    #doc_vectors = '%s/data/enwiki-20120502-lines-1k-300d.vec' % constants.BASE_DIR
    #query_vectors = '%s/luceneutil/tasks/vector-task-300d.vec' % constants.BASE_DIR
    #dim = 256
    #doc_vectors = '/d/electronics_asin_emb.bin'
    #query_vectors = '/d/electronics_query_vectors.bin'
    cp = benchUtil.classPathToString(benchUtil.getClassPath(checkout))
    cmd = [constants.JAVA_EXE, '-cp', cp,
           '--add-modules', 'jdk.incubator.vector',
           '-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false',
           'KnnGraphTester']
    print("recall\tlatency\tnDoc\tfanout\tmaxConn\tbeamWidth\tvisited\tindex ms")
    while advance(indexes, values):
        pv = {}
        args = []
        for (i, p) in enumerate(list(values.keys())):
            #print(f"i={i}, p={p}")
            if p in values:
                if values[p]:
                    value = values[p][indexes[i]]
                    pv[p] = value
                    #print(values[p])
                    #print(indexes)
                    #print(p)
                else:
                    args += ['-' + p]
        args += [a for (k, v) in pv.items() for a in ('-' + k, str(v)) if a]

        this_cmd = cmd + args + [
            '-dim', str(dim),
            '-docs', doc_vectors,
            '-reindex',
            '-search', query_vectors,
            #'-quantize',
            #'-confidenceInterval', '0.9990244',
            #'-randomCommits',
            '-metric', 'mip',
            #'-numMergeWorker', '8',
            '-numMergeWorker', '8', '-numMergeThread', '8',
            '-forceMerge',
            '-quiet',
            ]
        #print(this_cmd)
        subprocess.run(this_cmd)


run_knn_benchmark(LUCENE_CHECKOUT, VALUES)
