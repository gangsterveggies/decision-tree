class DecisionData:
  def __init__(self, raw_attributes, raw_attributes_spec, raw_data, raw_classification, raw_classification_spec):
    self.n_attributes = len(raw_attributes)
    self.attribute_list = range(self.n_attributes)
    self.attribute_names = raw_attributes
    self.attribute_dict = dict([(self.attribute_names[i], i) for i in range(self.n_attributes)])

    self.attribute_pos = raw_attributes_spec
    self.attribute_pos_dict = [dict([(self.attribute_pos[att][i], i) for i in range(len(self.attribute_pos[att]))]) for att in self.attribute_list]

    self.n_instances = len(raw_data)
    self.instance_list = []
    for i in range(self.n_instances):
      cons = [-1 for j in range(self.n_attributes)]
      for item in raw_data[i]:
        cons[self.attribute_dict[item[1]]] = self.attribute_pos_dict[self.attribute_dict[item[1]]][item[0]]

      self.instance_list.append(cons)

    self.classification_names = raw_classification_spec
    self.classification_dict = dict([(raw_classification_spec[i], i) for i in range(len(raw_classification_spec))])
    self.classification_list = [self.classification_dict[item] for item in raw_classification]
