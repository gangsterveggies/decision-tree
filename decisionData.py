class DecisionData:
  def __init__(self, raw_attributes, raw_attributes_spec, raw_data, raw_classification, raw_classification_spec):
    self.n_attributes = len(raw_attributes)
    self.attribute_list = range(self.n_attributes)
    self.attribute_names = raw_attributes
    self.attribute_dict = dict([(self.attribute_names[i], i) for i in range(self.n_attributes)])
    self.attribute_type = [int(DecisionData.isAllNumeric(DecisionData.column(raw_data, att))) for att in self.attribute_names]

    self.attribute_pos = raw_attributes_spec
    self.attribute_pos_dict = [dict([(self.attribute_pos[att][i], i) for i in range(len(self.attribute_pos[att]))]) for att in self.attribute_list]

    self.n_instances = len(raw_data)
    self.instance_list = []
    for i in range(self.n_instances):
      cons = [-1 for j in range(self.n_attributes)]
      for item in raw_data[i]:
        if self.attribute_type[self.attribute_dict[item[1]]] == 0:
          cons[self.attribute_dict[item[1]]] = self.attribute_pos_dict[self.attribute_dict[item[1]]][item[0]]
        else:
          cons[self.attribute_dict[item[1]]] = float(item[0])

      self.instance_list.append(cons)

    self.classification_names = raw_classification_spec
    self.classification_dict = dict([(raw_classification_spec[i], i) for i in range(len(raw_classification_spec))])
    self.classification_list = [self.classification_dict[item] for item in raw_classification]

  @staticmethod
  def column(matrix, att):
    ilist = []
    for line in matrix:
      for j in line:
        if j[1] == att:
          ilist.append(j[0])
    return ilist

  @staticmethod
  def isAllNumeric(ilist):
    for i in ilist:
      try:
        float(i)
      except:
        return False
    return True
