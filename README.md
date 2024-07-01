## Requirements
- [Tensorflow=1.2.0](https://github.com/tensorflow/tensorflow)
- [jieba=0.37](https://github.com/fxsjy/jieba)

## Basic Usage

### Default parameters:
- batch size: 20
- gradient clip: 5
- embedding size: 100
- optimizer: Adam
- dropout rate: 0.5
- learning rate: 0.001

### Train the model with default parameters:
```shell
$ python3 main.py --train=True --clean=True
```

### Online evaluate:
```shell
$ python3 main.py
$ python3 muba_KG_new_new. py
```

### Explain

The project consists of two parts, one is named entity recognition (execute main. py), which includes the basic crf+BILSTM, and the addition of the attention mechanism network and the highway network on this basis, and the other is knowledge graph construction (install Neo4j in advance, create an empty database, use python to construct the knowledge graph, and execute muba_KG_new_new. py).

The knowledge graph construction process mainly includes four steps: 1) the design of the concept layer. 2) the development of the data layer, which is composed of data annotation, feature extraction, and knowledge fusion. 3) knowledge graph construction. 4) knowledge graph applications. The details of each step are described in the paper: DOI10.1186/s12911-023-02322-0.



Because of the medical electronic medical record, no data set is provided in the code.



