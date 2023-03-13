import py2neo
from py2neo import Graph, Node, Relationship
import pandas as pd
import csv
import numpy as np
import datetime

start = datetime.datetime.now()
with open('data_align3-lee.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    data = np.array(data)
    columns , index =data.shape
# print(data)
#
# 连接neo4j数据库，输入地址、用户名、密码
graph = Graph("http://localhost:7474", auth=('neo4j','123456'))
graph.delete_all()

node_list = ['姓名', '性别', '年龄', '地址', '电话', '病人号', '检查号', '病人类型', '检查日期', '申请科室', '申请医生', '主诉情况', '其他',
             '检查部位', '分型', '类型', '位置', '否定', '数量', '形状', '边缘', '密度', '大小',
             '类型', '位置', '否定', '数量', '形状', '大小', '描述', '分布', '乳腺结构', '导管',
             '位置', '否定', '数量', '形状', '描述', '不对称', '位置', '否定', '数量', '征象', '位置', '否定', '数量', '形状', '描述',
             '位置', '否定', '描述', 'Bi_rads']
node_list_sort = ['姓名类', '性别类', '年龄类', '地址类', '电话类', '病人号类', '检查号类', '病人类型类', '检查日期类',
             '申请科室类', '申请医生类', '主诉情况类', '其他类',
             '检查部位类', '分型类', '肿块类型类', '肿块位置类', '肿块否定类', '肿块数量类', '肿块形状类', '肿块边缘类', '肿块密度类', '肿块大小类',
             '钙化类型类', '钙化位置类', '钙化否定类', '钙化数量类', '钙化形状类', '钙化大小类', '钙化描述类', '钙化分布类', '乳腺结构类', '导管类',
             '房内淋巴位置类', '房内淋巴否定类', '房内淋巴数量类', '房内淋巴形状类', '房内淋巴描述类', '不对称类', '关联异常位置类', '关联异常否定类',
                  '关联异常数量类', '关联异常征象类', '腋下淋巴位置类', '腋下淋巴否定类', '腋下淋巴数量类', '腋下淋巴形状类', '腋下淋巴描述类',
                  '血管位置类', '血管否定类', '血管描述类', 'Bi_rads类']
node_num = len(node_list) #code_num :52

if index-node_num != 0:
    print("概念层节点数与数据节点数不一致！！！")

'''以下部分是为了设置统计节点，即将每一类别节点进行统计，方便推理（含信息为空节点）'''

data_list=[]  #统计每列数据
for i in range(index):
    data_list.append([])
    for j in range(1,columns):
        if data[j][i]:
            data_list[i].append(data[j][i])

data_list_sort = [] #去重每列数据
for i in range(index):
    data_list_sort.append(list(set(data_list[i])))

node_sort = [] #归类节点
for i in range(index):
    node_sort.append(Node(node_list_sort[i], lable=node_list_sort[i]))
    graph.create(node_sort[i])

nodes_list = []  #创建统计节点
rela_list = [] #创建统计节点间关系
rela_connect = [] #创建数据节点和统计节点间关系

for i in range(index):
    nodes_list.append([])
    rela_list.append([])
    rela_connect.append([])
    for j in range(len(data_list_sort[i])):
        nodes_list[i].append(Node(node_list[i], lable=data_list_sort[i][j]))
        graph.create(nodes_list[i][j])
        rela_list[i].append(Relationship(node_sort[i], 'select', nodes_list[i][j]))
        graph.create(rela_list[i][j])

'''以下部分是为了设置实际数据节点，用于单个病人查询'''
node_dr = []  #设置病历节点
node = []#初始化节点
rela = []#初始化关系
rela_0 = [] #报告到信息关系
rela_1 = [] #钼靶信息到征象关系
rela_2 = [] #征象到实体
rela_3 =[] #实体到具体数据

node_1= [[0 for x in range(3)] for y in range(columns-1)] #每个病人含一级节点3个：基本信息、钼靶信息、bi_rads分类
node_2= [[0 for x in range(4)] for y in range(columns-1)] #每个病人含二级节点4个：乳腺分型、常见征象、特殊征象、合并征象
node_3= [[0 for x in range(9)] for y in range(columns-1)] #每个病人含三级节点9个：肿块、钙化、结构；导管、乳房内淋巴结、不对称；关联异常、腋下淋巴结、血管


for i in range(columns-1):
    '''为每个病人创建概念节点（框架节点）'''
    node_dr.append(i)
    node_dr[i] = Node('报告', lable=data[i+1][6])  #每个病人含零级节点1个：钼靶报告
    graph.create(node_dr[i])
    node_1[i][0] = Node('基本信息', lable='基本信息')
    node_1[i][1] = Node('钼靶信息', lable='钼靶信息')
    node_1[i][2] = Node('Bi_rads等级', lable='Bi_rads等级')
    for x in range(3):
        graph.create(node_1[i][x])
    node_2[i][0] = Node('乳腺分型', lable='乳腺分型')
    node_2[i][1] = Node('常见征象', lable='常见征象')
    node_2[i][2] = Node('特殊征象', lable='特殊征象')
    node_2[i][3] = Node('合并征象', lable='合并征象')
    for x in range(4):
        graph.create(node_2[i][x])
    node_3[i][0] = Node('肿块', lable='肿块')
    node_3[i][1] = Node('钙化', lable='钙化')
    node_3[i][2] = Node('结构', lable='结构')
    node_3[i][3] = Node('导管', lable='导管')
    node_3[i][4] = Node('乳房内淋巴结', lable='乳房内淋巴结')
    node_3[i][5] = Node('不对称', lable='不对称')
    node_3[i][6] = Node('关联异常', lable='关联异常')
    node_3[i][7] = Node('腋下淋巴结', lable='腋下淋巴结')
    node_3[i][8] = Node('血管', lable='血管')
    for x in range(9):
        graph.create(node_3[i][x])
    '''为每个病人创造概念节点间的关系:从报告到基本信息、钼靶信息、birands信息关系'''
    for i_0 in range(3):
        rela_0.append([])
        rela_0[i_0] = Relationship(node_dr[i], 'has_a', node_1[i][i_0])
        graph.create(rela_0[i_0])
    '''为每个病人创造概念节点间的关系:从基本信息到乳腺分型、常见征象、特殊征象、合并征象的关系'''
    for i_1 in range(4):
        rela_1.append([])
        rela_1[i_1] =Relationship(node_1[i][1], 'instance_of', node_2[i][i_1])
        graph.create(rela_1[i_1])
    '''为每个病人创造概念节点间的关系:从常见征象到肿块、钙化、结构的关系'''
    '''为每个病人创造概念节点间的关系:从特殊征象到导管、乳房内淋巴结、团状不对称、局灶性不对称的关系'''
    '''为每个病人创造概念节点间的关系:从合并征象到关联异常、腋下淋巴结、血管的关系'''
    for i_2 in range(9):
        rela_2.append([])
        if i_2 <= 2:
            rela_2[i_2] =Relationship(node_2[i][1], 'part_of', node_3[i][i_2])
        elif i_2 > 2 and i_2 <= 5 :
            rela_2[i_2] = Relationship(node_2[i][2], 'part_of', node_3[i][i_2])
        else:
            rela_2[i_2] = Relationship(node_2[i][3], 'part_of', node_3[i][i_2])
        graph.create(rela_2[i_2])
    '''为每个病人创造数据层节点，具体数据与上面汇总数据想对应'''
    tt=0
    rela_3.append([])
    for i_3 in range(index):
        pa = data[i+1][i_3]
        for i_4 in range(len(data_list_sort[i_3])):
            pb = data_list_sort[i_3][i_4]
            if pa == pb:
                if i_3 <= 12:
                    rela_3[i].append(Relationship(node_1[i][0], 'select', nodes_list[i_3][i_4]))
                elif (i_3>12 and i_3 <= 14) :
                    rela_3[i].append(Relationship(node_2[i][0], 'select', nodes_list[i_3][i_4]))
                elif (i_3>14 and i_3 <= 22) :
                    rela_3[i].append(Relationship(node_3[i][0], 'select', nodes_list[i_3][i_4]))
                elif (i_3>22 and i_3 <= 30) :
                    rela_3[i].append(Relationship(node_3[i][1], 'select', nodes_list[i_3][i_4]))
                elif (i_3>30 and i_3<= 31) :
                    rela_3[i].append(Relationship(node_3[i][2], 'select', nodes_list[i_3][i_4]))
                elif (i_3>31 and i_3<= 32) :
                    rela_3[i].append(Relationship(node_3[i][3], 'select', nodes_list[i_3][i_4]))
                elif (i_3>32 and i_3<= 37) :
                    rela_3[i].append(Relationship(node_3[i][4], 'select', nodes_list[i_3][i_4]))
                elif (i_3>37 and i_3<= 38) :
                    rela_3[i].append(Relationship(node_3[i][5], 'select', nodes_list[i_3][i_4]))
                elif (i_3>38 and i_3<= 42) :
                    rela_3[i].append(Relationship(node_3[i][6], 'select', nodes_list[i_3][i_4]))
                elif (i_3>42 and i_3<= 47) :
                    rela_3[i].append(Relationship(node_3[i][7], 'select', nodes_list[i_3][i_4]))
                elif (i_3>47 and i_3<= 50) :
                    rela_3[i].append(Relationship(node_3[i][8], 'select', nodes_list[i_3][i_4]))
                else:
                    rela_3[i].append(Relationship(node_1[i][2], 'select', nodes_list[i_3][i_4]))

    for t in range(len(rela_3[i])):
        graph.create(rela_3[i][t])

end = datetime.datetime.now()
print('运行时间 : %s 秒'%(end-start))
