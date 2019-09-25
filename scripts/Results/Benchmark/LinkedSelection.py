import os
import sys

class LinkedSelection():
	def __init__(self, selection = [], name=None):
		self.selection = selection
		self.links = []
		self.unique_name = name
	
	def __str__(self):
		return "LinkedSelection: "+str(self.selection)

	def __getitem__(self, index):
		return self.selection[index]

	def __len__(self):
		return len(self.selection)

	def __contains__(self, element):
		return element in self.selection

	def link(self, another, match):
		self.links.append( (another, match) )

	def remove(self, element):
		if element in self.selection:
			index = self.selection.index(element)
		else:
			return

		del self.selection[index]

		for another, match in self.links:
			if not (element in match.keys()):
				continue
			if match[element] is None:
				continue
			
			another_element = match[element]
			del match[element]
			another.remove(another_element)
		
			

def double_link_selections(selection_a, selection_b, match_ab):
	match_ba = {}
	for a_ind in match_ab.keys():
		b_ind = match_ab[a_ind]
		if not (b_ind is None):
			match_ba[b_ind] = a_ind
	selection_a.link(selection_b, match_ab)
	selection_b.link(selection_a, match_ba)

def test_link():
	print('Testing linkage:')
	a = LinkedSelection(["a","b","c","d"])
	b = LinkedSelection(["1","2","3"])
	print("Init:")
	print(a)
	print(b)
	
	ab_match = {"a":"1", "b":"2", "c":None, "d":"3"}
	a.link(b, ab_match)

	print("Remove a:")
	a.remove("a")

	print(a)
	print(b)

def test_double_link():
	print('Testing bidirectional linkage:')
	a = LinkedSelection(["a","b","c","d"])
	b = LinkedSelection(["1","2","3"])

	print("Init:")
	print(a)
	print(b)
	
	
	ab_match = {"a":"1", "b":"2", "c":None, "d":"3"}
	double_link_selections(a, b, ab_match)

	print("Remove 2:")
	b.remove("2")
	print(a)
	print(b)
	
	print("Remove a:")
	a.remove("a")
	print(a)
	print(b)

def test_circle_link():
	print('Testing bidirectional circular linkage:')
	a = LinkedSelection(["a","b","c","d"])
	b = LinkedSelection(["1","2","3"])
	c = LinkedSelection(["l","m","n", "o", "p"])

	print("Init:")
	print(a)
	print(b)
	print(c)
	
	
	ab_match = {"a":"1", "b":"2", "c":None, "d":"3"}
	bc_match = {"1":"l", "3":"o"}
	ca_match = {"l":"a", "n":"d"}
	double_link_selections(a, b, ab_match)
	double_link_selections(b, c, bc_match)
	double_link_selections(c, a, ca_match)

	print("Remove 2:")
	b.remove("2")
	print(a)
	print(b)
	print(c)
	
	print("Remove a:")
	a.remove("a")
	print(a)
	print(b)
	print(c)

if __name__=='__main__':
	test_link()
	test_double_link()
	test_circle_link()