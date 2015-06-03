from decisionTree import *
from decisionNode import *
from decisionData import *
import pickle

if __name__ == "__main__":
  attributes = [i.strip() for i in raw_input().split(',')][1:-1]
  instances = []
  classes = []

  while True:
    try:
      line = [i.strip() for i in raw_input().split(',')]
      inst = line[1:-1]
      clas = line[-1]
      instances.append([(inst[i], attributes[i]) for i in range(len(inst))])
      classes.append(clas)
    except EOFError:
      break

  att_vals = [[] for i in attributes]

  for inst in instances:
    for vl in inst:
      att_vals[attributes.index(vl[1])].append(vl[0])

  for i in range(len(att_vals)):
    att_vals[i] = list(set(att_vals[i]))

  data = DecisionData(attributes, att_vals, instances, classes, list(set(classes)))
  dt = DecisionTree()
  dt.trainData(data)
  dt.printTree()

  for i in range(len(instances)):
    print dt.classify(instances[i]), classes[i]

"""
  with open('decisionTree.pickle', 'wb') as f:
    pickle.dump(dt, f)
  with open('decisionTree.pickle', 'rb') as f:
    dt = pickle.load(f)

  print attributes
  print ""
  print att_vals
  print ""
  print instances
  print ""
  print classes
  print ""
  print list(set(classes))
  print ""
"""
