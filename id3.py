import copy
import numpy as np
import math

class Node:
  def __init__ (self, feature):
    self.children = []
    self.feature = feature
    self.label = ""
 
class Attribute:
	pass

class Data:
	
	def __init__(self, *, fpath = "", data = None):
		
		if not fpath and data is None:
			raise Exception("Must pass either a path to a data file or a numpy array object")

		self.raw_data, self.attributes, self.index_column_dict, \
		self.column_index_dict, self.header = self._load_data(fpath, data)

	def _load_data(self, fpath = "", data = None):
		
		if data is None:
			data = np.loadtxt(fpath, delimiter=',', dtype = str)

		header = data[0]
		index_column_dict = dict(enumerate(header))

		#Python 2.7.x
		# column_index_dict = {v: k for k, v in index_column_dict.items()}

		#Python 3+
		column_index_dict = {v: k for k, v in index_column_dict.items()}

		data = np.delete(data, 0, 0)

		attributes = self._set_attributes_info(index_column_dict, data)

		return data, attributes, index_column_dict, column_index_dict, header
	
	def _set_attributes_info(self, index_column_dict, data):
		attributes = dict()

		for index in index_column_dict:
			column_name = index_column_dict[index]
			if column_name == 'label':
				continue
			attribute = Attribute()
			attribute.name = column_name
			attribute.index = index - 1
			attribute.possible_vals = np.unique(data[:, index])
			attributes[column_name] = attribute

		return attributes

	def get_attribute_possible_vals(self, attribute_name):
		"""

		Given an attribute name returns the all of the possible values it can take on.
		
		Args:
		    attribute_name (str)
		
		Returns:
		    TYPE: numpy.ndarray
		"""
		return self.attributes[attribute_name].possible_vals

	def get_row_subset(self, attribute_name, attribute_value, data = None):
		"""

		Given an attribute name and attribute value returns a row-wise subset of the data,
		where all of the rows contain the value for the given attribute.
		
		Args:
		    attribute_name (str): 
		    attribute_value (str): 
		    data (numpy.ndarray, optional):
		
		Returns:
		    TYPE: numpy.ndarray
		"""
		if not data:
			data = self.raw_data

		column_index = self.get_column_index(attribute_name)
		new_data = copy.deepcopy(self)
		new_data.raw_data = data[data[:, column_index] == attribute_value]
		return new_data

	def get_column(self, attribute_names, data = None):
		"""

		Given an attribute name returns the corresponding column in the dataset.
		
		Args:
		    attribute_names (str or list)
		    data (numpy.ndarray, optional)
		
		Returns:
		    TYPE: numpy.ndarray
		"""
		if not data:
			data = self.raw_data

		if type(attribute_names) is str:
			column_index = self.get_column_index(attribute_names)
			return data[:, column_index]

		column_indicies = []
		for attribute_name in attribute_names:
			column_indicies.append(self.get_column_index(attribute_name))

		return data[:, column_indicies]


	def get_column_index(self, attribute_name):
		"""

		Given an attribute name returns the integer index that corresponds to it.
		
		Args:
		    attribute_name (str)
		
		Returns:
		    TYPE: int
		"""
		return self.column_index_dict[attribute_name]

	def __len__(self):
		return len(self.raw_data)

def compute_entropy (label):
  val = np.unique (label)
  total = label.shape[0]
  entropy = 0

  if val.size == 1:
    return 0

  entropy_count = np.zeros((val.shape[0], 1))

  for x in range (val.shape[0]):
    entropy_count[x] = (sum (label == val[x]))/(total * 1.0)

  for y in range (entropy_count.shape[0]):
    entropy += -1 * entropy_count[y] * math.log (entropy_count[y], 2)   

  return entropy

def compute_subtable (data_obj, col):
  train_data = data_obj.raw_data
  dict = {}

  pos_val_col = np.unique(train_data[:, col])
  col_name = data_obj.index_column_dict [col]

  for x in range (pos_val_col.shape[0]):
    dict [pos_val_col[x]] = data_obj.get_row_subset (col_name, pos_val_col[x])

  return pos_val_col, dict

def info_gain_cal (data_obj, col):
  train_data = data_obj.raw_data
  total_elements = train_data.shape[0] 

  pos_val_col, dict = compute_subtable (data_obj, col) 

  ext_entropy = np.zeros((pos_val_col.shape[0], 1))

  for x in range(pos_val_col.shape[0]):
    probability = dict[pos_val_col[x]].raw_data.shape[0]/(total_elements * 1.0)
    ext_entropy[x] = probability * compute_entropy (dict[pos_val_col[x]].raw_data[:,0])

  entropy_label = compute_entropy (train_data[:,0])
  info_gain = entropy_label 

  for x in range(ext_entropy.shape[0]):
    info_gain =  info_gain - ext_entropy[x]

  return info_gain

# Function to create nodes of the decision tree
def create_node (data_obj, header):
  train_data = data_obj.raw_data

  if (np.unique (train_data[:, 0])).shape[0] == 1:
    node = Node ("")
    node.label = np.unique (train_data[:, 0])[0]  
    return node

  info_gain = np.zeros((train_data.shape[1] - 1, 1))

  for col in range(1, train_data.shape[1]):
    info_gain[col - 1] = info_gain_cal (data_obj, col)

  # Since the first column is label
  root = np.argmax (info_gain) + 1
  feature_name = header[root]

  node = Node (feature_name)

  pos_val_col, dictionary = compute_subtable (data_obj, root)

  header = np.delete (header, root) 

  for x in range (pos_val_col.shape[0]):
    dictionary[pos_val_col[x]].raw_data = np.delete (dictionary[pos_val_col[x]].raw_data, root, 1)
    dictionary[pos_val_col[x]].index_column_dict = dict(enumerate(header))
    dictionary[pos_val_col[x]].column_index_dict = {v: k for k, v in dictionary[pos_val_col[x]].index_column_dict.items()}

  for x in range (pos_val_col.shape[0]):
    child_node = create_node (dictionary[pos_val_col[x]], header)
    node.children.append ((pos_val_col[x], child_node))

  return node
 
def space (size):
  s = ""
  for x in range(size):
    s += "   "
  return s

def print_tree (node, level):
  if node.label != "":
    print (space(level), node.label)
    return

  print (space(level), node.feature)

  for value, n in node.children:
    print (space(level + 1), value)
    print_tree(n, level + 2)

def match_label (test_row, header_dict, header, root):
  if (root.label != "") and (root.label == test_row[0]):
    return True
  elif (root.label != ""):
    return False

  col = header_dict[root.feature]

  for value, n in root.children:
    if (test_row[col] == value):
      header = np.delete (header, col)
      index_column_dict = dict(enumerate(header))
      header_dict= {v: k for k, v in index_column_dict.items()}
      test_row = np.delete (test_row, col)
      ret = match_label (test_row, header_dict, header, n)
      return ret

def test_data (test_obj, root):
  total_len = test_obj.__len__  ()
  match_count = 0

  for x in range (total_len):
    match = match_label (test_obj.raw_data[x], test_obj.column_index_dict,
                         test_obj.header,root)

    if match is True:
      match_count += 1
  
  print ('Match found = {} out of {} entries.'.format (match_count, total_len))

  accuracy = (match_count/total_len) * 100
  print ('Accuracy in train data = {}%'.format (accuracy))

# Main Function
def main ():
  data_obj = Data (fpath = "train.csv")

  root = create_node (data_obj, data_obj.header)

  print_tree (root, 0)

  test_obj = Data (fpath = "test.csv")

  test_data (test_obj, root)

if __name__=="__main__":
  main()
