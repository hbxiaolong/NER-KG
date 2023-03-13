# -*- coding:utf-8 -*-
import re

result = {'string': '59岁 DR00919681。双乳呈混合型,腺体丰富呈片絮状影,右乳外上象限见一枚小结节状钙化灶,径约4x5mm,左乳上象限见一枚颗粒状钙化灶,结构清楚,双乳未见确切肿块、异常钙化及增粗血管。双侧皮肤乳头及皮下脂肪层结构清晰,未见明显异常征象。双腋下未见肿大淋巴结。',
          'entities': [{'word': '59岁 ', 'start': 0, 'end': 4, 'type': 'Age'},
                       {'word': 'DR00919681', 'start': 4, 'end': 14, 'type': 'DRnum'},
                       {'word': '双乳', 'start': 15, 'end': 17, 'type': 'Location'},
                       {'word': '混合型', 'start': 18, 'end': 21, 'type': 'Typ'},
                       {'word': '右乳外上象限', 'start': 32, 'end': 38, 'type': 'Location'},
                       {'word': '一枚', 'start': 39, 'end': 41, 'type': 'Number'},
                       {'word': '小结节状', 'start': 41, 'end': 45, 'type': 'Shape'},
                       {'word': '钙化灶', 'start': 45, 'end': 48, 'type': 'Calcifications'},
                       {'word': '约4x5mm', 'start': 50, 'end': 56, 'type': 'Size'},
                       {'word': '左乳上象限', 'start': 57, 'end': 62, 'type': 'Location'},
                       {'word': '一枚', 'start': 63, 'end': 65, 'type': 'Number'},
                       {'word': '颗粒状', 'start': 65, 'end': 68, 'type': 'Shape'},
                       {'word': '钙化灶', 'start': 68, 'end': 71, 'type': 'Calcifications'},
                       {'word': '结构清楚', 'start': 72, 'end': 76, 'type': 'Structure'},
                       {'word': '双乳', 'start': 77, 'end': 79, 'type': 'Location'},
                       {'word': '未见', 'start': 79, 'end': 81, 'type': 'Negation'},
                       {'word': '肿块', 'start': 83, 'end': 85, 'type': 'Lump'},
                       {'word': '异常钙化', 'start': 86, 'end': 90, 'type': 'Calcifications'},
                       {'word': '增粗血管', 'start': 91, 'end': 95, 'type': 'Special'},
                       {'word': '双侧皮肤乳头及皮下脂肪层', 'start': 96, 'end': 108, 'type': 'Location'},
                       {'word': '结构清晰', 'start': 108, 'end': 112, 'type': 'Structure'},
                       {'word': '未见', 'start': 113, 'end': 115, 'type': 'Negation'},
                       {'word': '异常征象', 'start': 117, 'end': 121, 'type': 'Merge'},
                       {'word': '双腋下', 'start': 122, 'end': 125, 'type': 'Location'},
                       {'word': '未见', 'start': 125, 'end': 127, 'type': 'Negation'},
                       {'word': '肿大淋巴结', 'start': 127, 'end': 132, 'type': 'LymphNode'}
                       ]
          }


# print(result.keys(), result.values())
entities = result['entities']
sentence = ""
sentences = []

tag4n = 1
a = 1

sub_entity = []
sub_entities = []
sub_word = ""
sub_sentence = []
sub_tag = ""
sub_tags = []

age = []
drnum = []
typ = []
shape4Cal = []
margin4Cal = []
margin4Lump = []
structure = []
structure4Cal = []
shape4Cal = []
shape4Lump = []
calcifications = []
density = []
lymph = []
merge = ["未见"]
special = ["未见"]
# for i, entity in enumerate(entities):
#     print(entity)
#     if entity['type'] == 'Age':
#         age = entity['word']
#         continue
#     if entity['type'] == 'DRnum':
#         drnum = entity['word']
#         continue
#     if entity['type'] == 'Location':
#         tag4n = 1
#         if sentence is not "":
#             sentences.append(sentence)
#             sentence = entity['word']
#         else:
#             sentence += entity['word']
#     elif entity['type'] == 'Negation':
#         tag4n = 0
#     else:
#         if tag4n:
#             sentence += entity["word"]
# if sentence is not "":
#     sentences.append(sentence)
# print(age, drnum, sentences)

for i, entity in enumerate(entities):
    # 年龄和检查号信息
    if entity['type'] == 'Age':
        age = entity['word']
        continue
    if entity['type'] == 'DRnum':
        drnum = entity['word']
        continue
    if entity['type'] == 'Location':
        a = 1
        if sub_entity is not "":
            sub_entities.append(sub_entity)
            sub_entity = []
            sub_entity.append(str(entity['word'] + '$' + entity['type']))
        else:
            sub_entity.append(str(entity['word'] + '$' + entity['type']))
    elif entity['type'] == 'Negation':
        a = 0
    else:
        if a:
            sub_entity.append(str(entity['word'] + '$' + entity['type']))
if sub_entity is not "":
    sub_entities.append(sub_entity)
sub_entities.pop(0)
# print(sub_entities)
for i in range(len(sub_entities)):
    # 保存句子信息
    for x in range(len(sub_entities[i])):
        sub_word += sub_entities[i][x].split('$')[0]
        sub_tag += sub_entities[i][x].split('$')[1]
        if x == int(len(sub_entities[i])) - 1:
            print(sub_entities[i][x].split('$')[0])
        else:
            print(sub_entities[i][x].split('$')[0], end='')
    sub_sentence.append(sub_word)
    sub_tags.append(sub_tag)
    sub_word = ""
    sub_tag = ""

print(sub_tags)
for i in range(len(sub_entities)):
    for x in range(len(sub_entities[i])):
        # 分型特征
        if sub_entities[i][x].split('$')[1] == 'Typ':
            typ.append(sub_sentence[i])
        # if sub_entities[i][x].split('$')[1] == 'Calcifications':
        #     if sub_entities[i][x].split('$')[1] == 'Structure':
        #         structure.append(sub_sentence[i])
        #     pass
        # 合并征象
        if sub_entities[i][x].split('$')[1] == 'Merge':
            merge.append(sub_entities[i][x].split('$')[0])
        else:
            merge.append("未见")
        # 特殊征象
        if sub_entities[i][x].split('$')[1] == 'Special':
            special.append(sub_entities[i][x].split('$')[0])
        else:
            special.append("未见")
        # 淋巴结
        if sub_entities[i][x].split('$')[1] == 'LymphNode':
            lymph.append(sub_sentence[i])
        else:
            lymph.append("未见")

print('sub_entities', sub_entities)

for i in range(len(sub_entities)):
    if re.findall('Calcifications', ''.join(sub_entities[i])):
        calcifications.append(sub_sentence[i])
        # 钙化边缘信息
        if re.findall('Margin', ''.join(sub_entities[i])):
            for x in sub_entities[i]:
                if x.split('$')[1] == 'Margin':
                    margin4Cal.append(x.split('$')[0])
                    # 钙化形状信息
        if re.findall('Shape', ''.join(sub_entities[i])):
            for x in sub_entities[i]:
                if x.split('$')[1] == 'Shape':
                    shape4Cal.append(x.split('$')[0])
                    # 钙化结构信息
        if re.findall('Structure', ''.join(sub_entities[i])):
            for x in sub_entities[i]:
                if x.split('$')[1] == 'Structure':
                    structure4Cal.append(x.split('$')[0])
    if re.findall('Density', ''.join(sub_entities[i])):
        density.append(sub_sentence[i])
        # 肿块边缘信息
        if re.findall('Margin', ''.join(sub_entities[i])):
            for x in sub_entities[i]:
                if x.split('$')[1] == 'Margin':
                    margin4Lump.append(x.split('$')[0])

                    
print(age, drnum, structure, calcifications, '、'.join(typ), merge, special, lymph)
print(shape4Cal, structure4Cal, margin4Cal)


def proprecessing(result):
    """

    :param result:
    :return:
    """
