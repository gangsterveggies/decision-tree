import math
from decisionNode import *
from decisionData import *

class DecisionTree:
  def __init__(self):
    self.root = None
    self.data = None

  def trainData(self, data):
    self.data = data
    self.root = DecisionNode(range(data.n_attributes), range(data.n_instances))
    self.processNode(self.root)

  def processNode(self, current_node):
    categories_left = len(set([self.data.classification_list[item] for item in current_node.current_set]))

    if categories_left == 0:
      if current_node.parent == None:
        raise IndexError, "Zero data instances provided"

      current_node.leaf = True
      current_node.category = self.calculateMajority(current_node.parent)
    elif categories_left == 1:
      current_node.leaf = True
      current_node.category = self.data.classification_list[current_node.current_set[0]]
    elif len(current_node.current_attributes) == 0:
      current_node.leaf = True
      current_node.category = self.calculateMajority(current_node)
    else:
      score_list = []

      for att in current_node.current_attributes:
        score = self.calculateScore(att, current_node)
        score_list.append((att, score))

      next_att = sorted(score_list, key=lambda x: x[1])[-1][0]
      current_node.chosen = next_att
      dis_sets = [[] for i in range(len(self.data.attribute_pos[next_att]))]

      for item in current_node.current_set:
        dis_sets[self.data.instance_list[item][next_att]].append(item)

      nlist = list(current_node.current_attributes)
      nlist.remove(next_att)
      for i in range(len(dis_sets)):
        nnode = DecisionNode(nlist, dis_sets[i])
        nnode.parent = current_node
        current_node.children.append((nnode, i))
        self.processNode(nnode)

  def calculateScore(self, att, node):
    dis_sets = [[] for i in range(len(self.data.attribute_pos[att]))]

    for item in node.current_set:
      dis_sets[self.data.instance_list[item][att]].append(item)

    return self.calculateEntr(node.current_set) - sum([len(dis_sets[i]) * self.calculateEntr(dis_sets[i]) / len(node.current_set) for i in range(len(dis_sets))])

  def calculateEntr(self, ilist):
    if len(ilist) == 0:
      return 0

    count_array = [0, 0]

    for item in ilist:
      count_array[self.data.classification_list[item]] += 1

    total = float(sum(count_array))
    return -sum([count_array[i] / total * math.log(count_array[i] / total, 2) for i in range(2) if count_array[i] > 0])

  def calculateMajority(self, node):
    count_array = [0, 0]

    for item in node.current_set:
      count_array[self.data.classification_list[item]] += 1
    
    if count_array[0] > count_array[1]:
      return 0
    return 1

  def printTree(self):
    self.printNode(self.root, 0, "")

  def printNode(self, node, indent, prev_att):
    if node.leaf:
      print " " * indent + prev_att + ": " + self.data.classification_names[node.category] + "(" + str(len(node.current_set)) + ")"
    else:
      if len(prev_att) > 0:
        print " " * indent + prev_att + ":"
        indent += 1

      print " " * indent + self.data.attribute_names[node.chosen] + ":"
      for child in node.children:
        self.printNode(child[0], indent + 1, self.data.attribute_pos[node.chosen][child[1]])

  def classify(self, raw_instance):
    instance = [-1 for j in range(self.data.n_attributes)]
    for item in raw_instance:
      instance[self.data.attribute_dict[item[1]]] = self.data.attribute_pos_dict[self.data.attribute_dict[item[1]]][item[0]]

    return self.data.classification_names[self.searchNode(instance, self.root)]

  def searchNode(self, instance, node):
    if node.leaf:
      return node.category
    for ch in node.children:
      if instance[node.chosen] == ch[1]:
        return self.searchNode(instance, ch[0])
