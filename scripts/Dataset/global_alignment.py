import shlex
import subprocess
import os
import time
from subprocess import Popen, PIPE
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm

from Bio.PDB import *
from Bio.PDB.Polypeptide import d3_to_index, dindex_to_1
parser = PDBParser(QUIET=True)

def get_pdb_seq(pdb_path):
	structure = parser.get_structure('X', pdb_path)
	residues = structure.get_residues()
	seq = ''
	for residue in residues:
		if not residue.get_resname() in d3_to_index.keys():
			continue
		index = d3_to_index[residue.get_resname()]
		aa = dindex_to_1[index]
		seq += aa
	return Seq(seq)

def get_complex_pdb_sequences(pdb_list):
	complex_seq = []
	for receptor_path, ligand_path in pdb_list:
		receptor_seq = get_pdb_seq(receptor_path)
		ligand_seq = get_pdb_seq(ligand_path)
		complex_seq.append( (receptor_path, receptor_seq, ligand_path, ligand_seq) )
		
	return complex_seq

def get_alignment( params ):
	from Bio import pairwise2
	from Bio.SubsMat import MatrixInfo as matlist
	seq1, seq2 = params
	matrix = matlist.blosum62
	result = pairwise2.align.globaldx(seq1, seq2, matrix)
	# result = pairwise2.align.localms(seq1, seq2, 2, -1, -.5, -.1)
	
	if len(result)==0:
		return None, 0.0

	seq1_aligned = result[0][0]
	seq2_aligned = result[0][1]
	identical = 0
	mapping = []
	
	idx_seq1 = 0
	idx_seq2 = 0
	for c_seq1, c_seq2 in zip(seq1_aligned, seq2_aligned):
		if c_seq1 == '-' and c_seq2=='-':
			continue
		if c_seq1 == '-':
			idx_seq2+=1
			continue
		if c_seq2 == '-':
			idx_seq1+=1
			continue
		
		if c_seq1 == c_seq2:
			identical += 1
			mapping.append((idx_seq1, idx_seq2))
		
		idx_seq1 += 1
		idx_seq2 += 1
	
	identity_percentage = float(identical)/float(len(seq1_aligned))

	return mapping, identity_percentage, (seq1_aligned, seq2_aligned)

if __name__=='__main__':
	print(get_alignment(('GCCSLPPCALSNPDYCX', 'LPPCARSNPDYC')))