#!/usr/bin/env/python

import os
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
LUCENE_CHECKOUT = 'lucene_candidate'

# test parameters. This script will run KnnGraphTester on every combination of these parameters
VALUES = {
    #'ndoc': (10000, 100000, 1000000),
    #'ndoc': (10000, 100000, 200000, 500000),
    'ndoc': (100000,),# 200000),
    #'ndoc': (100000,),
    'maxConn': (16,),
    'beamWidthIndex': (100,),
    #'beamWidthIndex': (200,),
    'fanout': (10,),#, 250),
    'topK': (10,),
    'topKAdd': (10, 20, 30, 40, 50),
    #'niter': (10,),
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

def run_knn_benchmark(checkout, values, training_file, testing_file, dims, metric):
    indexes = [0] * len(values.keys())
    indexes[-1] = -1
    args = []
    print(f"\n\n\nNow running{training_file}\n\n\n")
    dim = dims #768
    doc_vectors = training_file # '%s/util/wiki768ja.random.train' % constants.BASE_DIR #constants.GLOVE_VECTOR_DOCS_FILE
    query_vectors = testing_file # '%s/util/wiki768ja.test' % constants.BASE_DIR #'%s/util/tasks/vector-task-100d.vec' % constants.BASE_DIR
    #dim = 768
    #doc_vectors = '%s/data/enwiki-20120502-lines-1k-mpnet.vec' % constants.BASE_DIR
    #query_vectors = '%s/luceneutil/tasks/vector-task-mpnet.vec' % constants.BASE_DIR
    #dim = 384
    #doc_vectors = '%s/data/enwiki-20120502-lines-1k-minilm.vec' % constants.BASE_DIR
    #query_vectors = '%s/luceneutil/tasks/vector-task-minilm.vec' % constants.BASE_DIR
    #dim = 300
    #doc_vectors = '%s/data/enwiki-20120502-lines-1k-300d.vec' % constants.BASE_DIR
    #query_vectors = '%s/util/tasks/vector-task-300d.vec' % constants.BASE_DIR
    #dim = 256
    #doc_vectors = '/d/electronics_asin_emb.bin'
    #query_vectors = '/d/electronics_query_vectors.bin'
    JAVA_EXE = '/Users/benjamintrent/Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home/bin/java'

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
        for co in checkout:
            print(co)
            #jfr = f"-XX:StartFlightRecording=settings=profile,stackdepth=1028,maxsize=500M,dumponexit=true,filename={co}-768-100000.jfr"
            jfr = f"-agentpath:/Users/benjamintrent/Downloads/async-profiler-2.9-macos/build/libasyncProfiler.so=start,event=wall,file={co}-768-100000-wall.jfr"
            cp = benchUtil.classPathToString(benchUtil.getClassPath(co))
            cmd = [JAVA_EXE,
                   #jfr,
                   '-cp', cp,
                   '--add-modules', 'jdk.incubator.vector',
                   '-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false',
                   'KnnGraphTester']
            this_cmd = cmd + args + [
                '-dim', str(dim),
                '-docs', doc_vectors,
                #'-stats',
                #'-reindex',
                '-metric', metric,
                '-search', query_vectors,
                #'-forceMerge',
                #'-nested', str(128),
                #'-niter', str(3000),
                #'-quantile', "1.0",
                '-quantize',
                '-quiet',
            ]
            #print(this_cmd)
            subprocess.run(this_cmd)


run_knn_benchmark(['lucene_candidate'], VALUES, 'wiki768.train.norm', 'wiki768.test.norm', 768, "angular")
