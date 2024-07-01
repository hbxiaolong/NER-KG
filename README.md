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
![FIG 1](https://github.com/hbxiaolong/NER-KG/assets/116270284/c4a5930c-67b6-4a0d-b278-0358571eac59)



**In the design of the concept layer**, we leveraged the Breast Imaging Reporting and Data System (BI-RADS) lexicon of the American College of Radiology , and the Breast Cancer Diagnosis and Treatment Guidelines issued by the Chinese Anti-Cancer Association. Additionally, we engaged four mammography radiologists from different hospitals to participate in ontology definition and concept layer framework design. Moreover, we extensively referenced domestic and international literature, as well as mammography examination norms and standards, to revise the concept layer .
The concept layer is designed as a 3-level hierarchical structure with 15 types of mammography features primarily (Calcification, Density, Distribute, Location, Mass, Lymph Node, Margin, Merge, Number, Shape, Size, Special, Structure, Category descriptions, Negation). There are 4 entities such as "common signs" at the first level, there are 9 entities such as "Mass" at the second level, and related entity attributes are at the third level (Table 1). According to the three-element principle of knowledge graph construction , it is necessary to clarify the three elements of entity-relationship-entity attribute or entity-relationship-entity. Guided by clinicians, we established relationships among entities and attributes. This study not only defined a set of entities and their attributes, but also established their hierarchical relationships. Table 2 shows the three elements of subject, predicate and object among different entities and attributes in the concept layer. Once entities, entity attributes, and relationships were specified, the design of the concept layer of the knowledge graph for breast cancer diagnosis was completed. 

![image](https://github.com/hbxiaolong/NER-KG/assets/116270284/4667e984-1aa5-48a8-9df5-37a792a3a71d)

![image](https://github.com/hbxiaolong/NER-KG/assets/116270284/038324d2-1dd1-4361-84d6-88bcfd418d28)

**Development of the data layer **, we used the BILSTM-CRF deep learning model as the basic model, and improved the performance of the model by adding attention mechanisms and highway networks. This mainly includes steps such as data annotation, data extraction, paper knowledge fusion, and knowledge processing.

 The approval was obtained from the hospital's ethics review committee for this study(Serial No. 2022-082-01). There are 2989 mammography examination reports collected from the Radiology department of a Three-A hospital, spanning from December 2018 to July 2019. The data was then subjected to de-identification processing. Each report was prepared by one doctor and verified by another doctor. A total of nine doctors participated in this work. After duplicated reports were removed and reports with incomplete or incorrect data were deleted, the final dataset consists of 1171 reports used to construct the knowledge graph for breast cancer diagnosis.
Because of the medical electronic medical record, no data set is provided in the code. We have provided some sample data in the "DATA" file. The data can be used for scientific research. If needed, please contact: hbxiaolong@ctgu.edu.cn.



