from optparse import OptionParser
from decisionTree import *
from decisionNode import *
from decisionData import *
import pickle

def isDefined(var):
  return var in globals()

def readInstanceClass(line, attributes):
  inst = [i.strip() for i in line.split(',')]
  return [(inst[i], attributes[i]) for i in range(len(inst))]

def readInstance(line, attributes):
  line = [i.strip() for i in line.split(',')]
  inst = line[1:-1]
  clas = line[-1]
  return [(inst[i], attributes[i]) for i in range(len(inst))], clas

def readData(filename):
  lines = [line.rstrip('\n') for line in open(filename, 'r')]
  attributes = [i.strip() for i in lines[0].split(',')][1:-1]
  instances = []
  classes = []

  for line in lines[1:]:
    a, b = readInstance(line, attributes)
    instances.append(a)
    classes.append(b)

  return DecisionData(attributes, instances, classes)

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-t", "--train", dest="train",
                    help="train decision tree from FILE", metavar="FILE")
  parser.add_option("-o", "--output", dest="output",
                    help="store decision tree in FILE", metavar="FILE")
  parser.add_option("-l", "--load", dest="load",
                    help="load decision tree from FILE", metavar="FILE")
  parser.add_option("-d", "--draw", dest="draw",
                    help="draw the decision tree", metavar="FILE")
  parser.add_option("-p", "--print", dest="dprint", default=False,
                    help="print the decision tree", action="store_true")
  parser.add_option("-s", "--test", dest="test",
                    help="test instances from FILE", metavar="FILE")
  parser.add_option("-c", "--classify", dest="classify", default=False,
                    help="classify instance from standard input", action="store_true")

  (options, args) = parser.parse_args()

  if options.train:
    data = readData(options.train)
    dt = DecisionTree(advanced_score=True)
    dt.trainData(data)

  if options.load:
    with open(options.load, 'rb') as f:
      dt = pickle.load(f)

  if options.output:
    if not(isDefined('dt')):
      raise Exception, "The decision tree was not built"

    with open(options.output, 'wb') as f:
      pickle.dump(dt, f)

  if options.dprint:
    if not(isDefined('dt')):
      raise Exception, "The decision tree was not built"

    dt.printTree()

  if options.draw:
    if not(isDefined('dt')):
      raise Exception, "The decision tree was not built"

    dt.drawTree(sfile=options.draw)

  if options.test:
    if not(isDefined('dt')):
      raise Exception, "The decision tree was not built"

    lines = [line.rstrip('\n') for line in open(options.test, 'r')]

    correct = 0
    for line in lines:
      inst, res = readInstance(line, dt.data.attribute_names)
      if res == dt.classify(inst):
        correct += 1

    print "Test report:"
    print "Got %d of %d correct" % (correct, len(lines))
    print "That's a %0.2f%% success percentage" % (round(100 * correct / float(len(lines)), 2))

  if options.classify:
    if not(isDefined('dt')):
      raise Exception, "The decision tree was not built"

    print dt.classify(readInstanceClass(raw_input(), dt.data.attribute_names))
