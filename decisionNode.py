class DecisionNode:
  def __init__(self, attributes, data):
    self.current_attributes = attributes
    self.current_set = data
    self.children = []
    self.category = None
    self.chosen = None
    self.parent = None
    self.leaf = False
