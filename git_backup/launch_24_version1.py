#With the backpropagation in development
import numpy, argparse, os, re, random, math

import numpy as np

from collections import Counter

def sigmoid(x):
	#print "Hello",x
	return 1 / (1 + math.exp(-x))

def sigmoid1(X):
	return 1/(1+np.exp(-X*1.000))

def dsigmoid(X):
	"""
	print "Hello"
	print sigmoid1(X).shape
	print (1-sigmoid1(X)).shape
	"""
	#return sigmoid1(X)*(1-sigmoid1(X))
	save_results = np.multiply(sigmoid1(X),(1-sigmoid1(X)))
	#print save_results
	return save_results
	#return sigmoid1(X).dot(1-sigmoid1(X))
	#return np.einsum('ij,jkl->ikl',sigmoid1(X),(1-sigmoid1(X)))



def slidingWindow(sequence,winSize,step=1):
	sliding_print = 1024
	
	global W
	global W_layer2
	global bias_layer_1
	global bias_layer_2
	
	file_output = open("log_sigmoid.txt", 'wa')
	
	if(sliding_print==1):
		print "\n\nThe calculations begin here"

	# Verify the inputs
	try: it = iter(sequence)
	
	except TypeError:
		raise Exception("**ERROR** sequence must be iterable.")
	
	if not ((type(winSize) == type(0)) and (type(step) == type(0))):
		raise Exception("**ERROR** type(winSize) and type(step) must be int.")
	
	if step > winSize:
		raise Exception("**ERROR** step must not be larger than winSize.")
	
	if winSize > len(sequence):
		raise Exception("**ERROR** winSize must not be larger than sequence length.")

	# Pre-compute number of chunks to emit
	numOfChunks = ((len(sequence)-winSize)/step)+1

	# Do the work
	for i in range(0,numOfChunks*step,step):
		current_window = [x for x in sequence[i:i+winSize]]
		#Pay attention to the fact that the dummy word has been marked as occuring once, though it has not occured before
		
		if(sliding_print==1):
			print "This is the current window :"
			print current_window
		
		abc = [vector_update[x] for x in [x1 for x1 in sequence[i:i+winSize]]]
		abc_temp=abc
		
		if(sliding_print==1):
			print abc
		
		abc = np.asarray(abc)
		abc = np.concatenate(abc)
		abc.resize(len(abc),1)
		
		if(sliding_print==1):
			print "abc is"
			print abc
			print "here"
		
		if(args.window_size%2==0):
			center = args.window_size/2
		else:
			center = (args.window_size-1)/2
		
		
		#
		#
		#
		#The location to shift the loop in case we desire the repetition to go for each middle word
		#rather than through the document
		#
		#
		
		cnt = 1
		repetition = 1
		while (cnt!=0 and repetition<3) :
			print "\n\n\ncnt", cnt, " repetition", repetition
			repetition = repetition + 1
			cnt = 0
			#
			#
			#
			#The location to shift the loop in case we desire the repetition to go through the document,
			#rather than for each middle word
			#
			#
			"""
			First use the middle word to calculate the score
			"""
			print words
			print "\nThe window is", current_window;
			print "\n",current_window[center],"\t: ",
			print vector_update[current_window[center]]
			
			W_temp1 = np.asmatrix(W)
			"""
			print "W_temp1 is "
			print W_temp1
			W_temp123 = W_temp1.T
			print "W_temp123 is "
			print W_temp123
			print "W_temp1 is "
			print W_temp1
			"""
			
			if(args.verbosity):
				print W_temp1
				print "*"
				print abc
				print "="
			
			W_temp1 = W_temp1*abc + bias_layer_1
			
			if(args.verbosity):
				print W_temp1
			
			W_temp2 = np.asmatrix(W_layer2)
			
			if(args.verbosity):
				print "The shape of the layer 1's output is",W_temp1.shape
				print "The shape of the layer 2 is",W_temp2.shape
			
			print "The original score is\t",
			original_score = W_temp2*W_temp1 + bias_layer_2
			original_score = np.squeeze(np.asarray(original_score[0][0]))
			print original_score, "\tfor \"",current_window[center],"\"\n"

			"""
			Then calculate the score for the other words
			"""
		
			for word_tuple in words:
				if(word_tuple[0]!=current_window[center]):
					if(args.verbosity==True):
						print "\n",word_tuple[0],"\t: ",
						print vector_update[word_tuple[0]]
					"""
					else:
						print word_tuple[0],
					"""
					abc_window_temp = abc
					
					if(args.verbosity):
						print "The value of center is", center
					
					for i in xrange ((center)*args.word_size,  (center+1)*args.word_size):
						abc_window_temp[i] = vector_update[word_tuple[0]][i-(center)*args.word_size]
					
					if(args.verbosity):
						print "Updated vector"
						print abc_window_temp
					
					W_temp1 = np.asmatrix(W)
					
					if(args.verbosity):
						print W_temp1.shape
						print abc_window_temp.shape
					
					W_temp1 = W_temp1*abc_window_temp + bias_layer_1
					"""
					print W_temp1.__class__
					print W_temp1
					W_temp1 = sigmoid1(W_temp1)
					print W_temp1.__class__
					print W_temp1
					"""
					W_temp1_sigmoid = sigmoid1(W_temp1)
					
					#backpropagate_1 = W_temp1
					W_temp1_save = W_temp1
					backpropagate_1 = W_temp1_sigmoid
					
					if(args.verbosity):
						print W_temp1
					
					W_temp2 = np.asmatrix(W_layer2)
					
					if(args.verbosity):
						print "The shape of the layer 1's output is",W_temp1.shape
						print "The shape of the layer 2 is",W_temp2.shape
					
					print "The changed score is\t",
					word_score = W_temp2*W_temp1 + bias_layer_2
					word_score = np.squeeze(np.asarray(word_score[0][0]))
					
					backpropagate_2 = word_score
					
					print 1+original_score-word_score, "\tfor \"",word_tuple[0],"\""
					update_parameter = original_score - word_score - 1
					error = max(0, update_parameter)

					if(error):
						print "This error has to be accounted for"
						cnt = cnt + 1
						d_word_score = 1

						"""
						Backpropagation is done here

						dZ2=np.zeros_like(Z2)
						dZ2[margin>0]=1
						dZ2[margin<0]=0
						dZ2[y]-=np.sum(margin>0)
						dW2=dZ2.dot(A1.T)
						dA1=np.dot(W2.T,dZ2)
						dZ1=dsigmoid(Z1)*dA1
						dW1=dZ1.dot(X.T)
						dx=np.dot(W1.T,dZ1)
						dx=dx[1:]
						"""

						#"""
						d_word_score = 1
						d_W_layer2 = d_word_score*W_layer2.T  #dZ2.dot(A1.T)
						d_W_temp1_sigmoid = W_layer2.T*1
						"""
						print W_temp1_save.shape
						print d_W_temp1_sigmoid.shape
						"""
						d_W_temp1_save = np.multiply(dsigmoid(W_temp1_save),d_W_temp1_sigmoid)
						print "d_W_temp1_save's shape is ",d_W_temp1_save.shape
						print "abc.T's shape is ",(abc.T).shape
						d_W = np.multiply(d_W_temp1_save,abc.T)
						#d_W = d_W_temp1_save*abc.T
						#dx=np.dot(W1.T,dZ1)
						d_X = np.multiply(W, d_W_temp1_save)
						#d_X = W.T*d_W_temp1_save
						print "abc's shape is ",abc.shape
						print "d_X's shape is ",d_X.shape
						"""
						dx=dx[1:]
						"""
						#"""

						"""
						W = W + error/original_score;
						W_layer2 = W_layer2 + error/original_score;
						vector_update[word_tuple[0]] = vector_update[word_tuple[0]] + error/original_score;
						"""
						print "W's shape is ",W.shape
						print "d_W's shape is ",d_W.shape
						print "W_layer2's shape is ",W_layer2.shape
						print "d_W_layer2's shape is ",d_W_layer2.shape
						W = W + d_W
						W_layer2 = W_layer2 + d_W_temp1_save
					#break
				else:
					continue;
			"""
			for word_tuple in words:
				if(word_tuple[0]==current_window[center]):
					print "\n",word_tuple[0],"\t: ",
					print vector_update[word_tuple[0]]
					W = np.asmatrix(W)
					abc = np.asmatrix(abc)
					if(sliding_print==1):
						print "abc original is"
						#print abc
						print "here"
					abc_temp = np.asmatrix(abc_temp)
					if(sliding_print==1):
						print "New added vector is"
						print vector_update[word_tuple[0]]
					for i in range(0,len(vector_update[word_tuple[0]])):
						abc[center*args.word_size+i] = vector_update[word_tuple[0]][i]
					if(sliding_print==1):
						print 
						print "abc modified is"
						#print abc
						print "here"
					Z = W*abc
					if(sliding_print==1):
						print "The first multiplication is done"
						print Z
					A = np.tanh(Z)
					if(sliding_print==1):
						print "tanh function has been acted upon"
						print A
					global W_layer2
					if(sliding_print==1):
						print "W_layer2 is of shape :",W_layer2.shape
						print "A is of shape :",A.shape
					Z_2 = (W_layer2)*A
					if(sliding_print==1):
						print "Z_2 is"
						print Z_2
					h_w = sigmoid(Z_2)
					if(sliding_print==1):
						print "h_w(x) is "
						print h_w
					file_output.write(str(h_w)+"'"+' '.join(str(e) for e in current_window)+"'"+os.linesep)
				else:
					continue;
					print "\nFor the actual word this is the computation that is being carried out"
					print word_tuple[0]
					print current_window
					
					if(sliding_print==1):
						print "\n\n",word_tuple[0]
						print "The word's representation is :",vector_update[word_tuple[0]]
					W = np.asmatrix(W)
					abc = np.asmatrix(abc)
					if(sliding_print==1):
						print "abc original is"
						#print abc
						print "here"
					abc_temp = np.asmatrix(abc_temp)
					if(sliding_print==1):
						print "New added vector is"
						print vector_update[word_tuple[0]]
					for i in range(0,len(vector_update[word_tuple[0]])):
						abc[center*args.word_size+i] = vector_update[word_tuple[0]][i]
					if(sliding_print==1):
						print 
						print "abc modified is"
						print abc
					Z = W*abc
					if(sliding_print==1):
						print "The first multiplication is done"
						print Z
					A = np.tanh(Z)
					if(sliding_print==1):
						print "tanh function has been acted upon"
						print A
					#global W_layer2
					if(sliding_print==1):
						print "W_layer2 is of shape :",W_layer2.shape
						print "A is of shape :",A.shape
					Z_2 = (W_layer2)*A
					if(sliding_print==1):
						print "Z_2 is"
						print Z_2
					h_w = sigmoid(Z_2)
					if(sliding_print==1):
						print "h_w(x) is "
						print h_w
					file_output.write(str(h_w)+"'"+' '.join(str(e) for e in current_window)+"'"+os.linesep)
					continue
			break
			"""



			#return
			"""
			Then calculate the score for the other words
			"""
			"""
			for word_tuple in words:
				if(word_tuple[0]!=current_window[center]):
					print "\n",word_tuple[0],"\t: ",
					#print "The word's representation is :",vector_update[word_tuple[0]]
					print vector_update[word_tuple[0]]
					W = np.asmatrix(W)
					abc = np.asmatrix(abc)
					if(sliding_print==1):
						print "abc original is"
						#print abc
						print "here"
					abc_temp = np.asmatrix(abc_temp)
					if(sliding_print==1):
						print "New added vector is"
						print vector_update[word_tuple[0]]
					for i in range(0,len(vector_update[word_tuple[0]])):
						abc[center*args.word_size+i] = vector_update[word_tuple[0]][i]
					if(sliding_print==1):
						print 
						print "abc modified is"
						#print abc
						print "here"
					Z = W*abc
					if(sliding_print==1):
						print "The first multiplication is done"
						print Z
					A = np.tanh(Z)
					if(sliding_print==1):
						print "tanh function has been acted upon"
						print A
					#global W_layer2
					if(sliding_print==1):
						print "W_layer2 is of shape :",W_layer2.shape
						print "A is of shape :",A.shape
					Z_2 = (W_layer2)*A
					if(sliding_print==1):
						print "Z_2 is"
						print Z_2
					h_w = sigmoid(Z_2)
					if(sliding_print==1):
						print "h_w(x) is "
						print h_w
					file_output.write(str(h_w)+"'"+' '.join(str(e) for e in current_window)+"'"+os.linesep)
				else:
					print "\nFor the actual word this is the computation that is being carried out"
					print word_tuple[0]
					print current_window
					
					if(sliding_print==1):
						print "\n\n",word_tuple[0]
						print "The word's representation is :",vector_update[word_tuple[0]]
					W = np.asmatrix(W)
					abc = np.asmatrix(abc)
					if(sliding_print==1):
						print "abc original is"
						#print abc
						print "here"
					abc_temp = np.asmatrix(abc_temp)
					if(sliding_print==1):
						print "New added vector is"
						print vector_update[word_tuple[0]]
					for i in range(0,len(vector_update[word_tuple[0]])):
						abc[center*args.word_size+i] = vector_update[word_tuple[0]][i]
					if(sliding_print==1):
						print 
						print "abc modified is"
						print abc
					Z = W*abc
					if(sliding_print==1):
						print "The first multiplication is done"
						print Z
					A = np.tanh(Z)
					if(sliding_print==1):
						print "tanh function has been acted upon"
						print A
					#global W_layer2
					if(sliding_print==1):
						print "W_layer2 is of shape :",W_layer2.shape
						print "A is of shape :",A.shape
					Z_2 = (W_layer2)*A
					if(sliding_print==1):
						print "Z_2 is"
						print Z_2
					h_w = sigmoid(Z_2)
					if(sliding_print==1):
						print "h_w(x) is "
						print h_w
					file_output.write(str(h_w)+"'"+' '.join(str(e) for e in current_window)+"'"+os.linesep)

					This is the redundancy to be taken care of later on
					continue
			break
			#Comment this out when the sliding window has to be carried out for the entire sequence
			"""
			file_write = open("backpropagation.txt", "a")
			file_write.write("Total errors for the window is " + str(cnt) + " corresponding to the repetition " + str(repetition-1) + "\n")
		#repetition = repetition + 1
		#print "The repetition is", repetition
		print "Now the cnt is", cnt, " and the repetition", repetition-1
		file_write.write("\n\n")
	return

def word_vocab_build():
	global words
	
	lines = [line.strip() for line in open(args.input_file)]
	lines = [line.split() for line in lines]
	words = [item for sublist in lines for item in sublist]
	
	if(not args.uppercase):
		words = [x.lower() for x in words]
	
	words.append("DGDD")
	words = Counter(words).most_common(len(words))
	statinfo = (os.stat(args.input_file).st_size)/(1024**2)
	print "The input file is of size :",statinfo
	
	if(statinfo<10):
		print words
	
	file_output = open(args.vocabulary, 'w')
	file_output_count = open(args.vocabulary_count, 'w')
	
	for word_tuple in words:
		if(word_tuple[1]<args.word_limit):
			#continue
			break
		else:
			file_output.write(word_tuple[0]+os.linesep)
			file_output_count.write(word_tuple[0]+"\t"+str(word_tuple[1])+os.linesep)

def word_vec_init():
	global vec_init
	
	vec_init = [random.uniform(0, 1) for _ in range(0, args.window_size)]
	print vec_init
	file_vec_output = open(args.vector_output, 'w')
	file_vec_output_word = open("word_vector_output_verbose.txt", 'w')
	global vector_update
	vector_update = {}
	vec_dummy = [1]*args.word_size
	vector_update["DGDD"] = vec_dummy
	print "DGDD and : ",
	print vector_update["DGDD"]
	
	for word_tuple in words:
		if(word_tuple[1]<args.word_limit):
			break
		else:
			vec_init = [random.uniform(1, 10) for _ in range(0, args.word_size)]
			vec_init_str=' '.join(str(e) for e in vec_init)
			vector_update[word_tuple[0]] = vec_init
			file_vec_output.write(vec_init_str+os.linesep)
			file_vec_output_word.write(word_tuple[0]+"\t"+vec_init_str+os.linesep)
	
	print
	print

def word_vec_check():
	return
	lines_matrix = [line.strip() for line in open(args.input_file)]
	lines_matrix = [line.split() for line in lines_matrix]
	words_matrix = [item for sublist in lines_matrix for item in sublist]
	
	if(not args.uppercase):
		words_matrix = [x.lower() for x in words_matrix]
	
	for word_matrix in words_matrix:
		print word_matrix + " : " + vector_update[word_matrix]

def word_vec_process():
	if(args.verbosity):
		print "Initial checking : vector_update[\"this : \"]",
		print type(vector_update["this"][0])
	
	word_vec_check()
	lines_matrix = [line.strip() for line in open(args.input_file)]
	lines_matrix = [line.split() for line in lines_matrix]
	
	for lines_matrix_each in lines_matrix:
		lines_matrix_each = [line.split() for line in lines_matrix_each]
		words_matrix = [item for sublist in lines_matrix_each for item in sublist]
		
		if(not args.uppercase):
			words_matrix = [x.lower() for x in words_matrix]
		
		print "We are sending : "
		print words_matrix
		padding = 3-2;
		words_matrix = ["DGDD"]*padding + words_matrix + ["DGDD"]*padding
		slidingWindow(words_matrix,args.window_size)
		return
		#Remove the preceding line for complete functioning
	return
	lines_matrix = [line.split() for line in lines_matrix]
	words_matrix = [item for sublist in lines_matrix for item in sublist]
	
	if(not args.uppercase):
		words_matrix = [x.lower() for x in words_matrix]
	
	print words_matrix
	print "The sliding window"
	slidingWindow(words_matrix, args.window_size)

def weight_init():
	bias_print = 10
	
	if(bias_print==1):
		print "Init"
	
	global W
	global W_layer2
	global bias_layer_1
	global bias_layer_2
	
	W = numpy.random.uniform(low=0, high=1, size=(args.deep_neural,args.word_size*args.window_size))
	W_layer2 = numpy.random.uniform(low=0, high=10, size=(1,args.deep_neural))
	bias_layer_1 = numpy.random.uniform(low=0, high=10, size=(args.deep_neural,1))
	bias_layer_2 = numpy.random.uniform(low=0, high=10, size=(1,1))
	
	if(bias_print==1):
		print "The hidden neural is :",args.deep_neural
		print "The word vector size is :",args.word_size
		print "The window size is :",args.window_size
		print "W = numpy.random.uniform(low=0, high=10, size=(args.deep_neural,args.word_size*args.window_size))"
		print "The weight matrix is :"
		print W.shape
		print "W is"
		print W

	if(bias_print==1):
		print "Bias layer 1"
	
	global bias
	bias = numpy.random.uniform(low=0, high=1, size=(args.deep_neural,1))
	
	if(bias_print==1):
		print "The bias layer_1 matrix is :"
		print bias.shape
		print "bias is"
		print bias

		print
		print "Bias layer 2"
	
	global bias_2
	bias_2 = numpy.random.uniform(low=0, high=1, size=(1,1))
	
	if(bias_print==1):
		print "The bias layer_1 matrix is :"
		print bias_2.shape
		print "bias_2 is"
		print bias_2
		print "\n\n"

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	#parser.add_argument("-square", help="display the square of a given number", type=int, default=2)
	parser.add_argument("-dn", "--deep-neural", help="Deep Neural Network size", type=int, default=50)
	parser.add_argument("-window", "--window-size", help="Window size of words(5)", type=int, default=5)
	parser.add_argument("-word", "--word-size", help="Word vector size(50)", type=int, default=50)
	parser.add_argument("-iter", "--iteration-count", help="iteration count upperbound", type=int, default=5000)
	parser.add_argument("-eps", "--epsilon", help="epsilon value for termination condition", type=int, default=0.001)
	parser.add_argument("-in", "--input-file", help="corpus to read from in order to create the word vector", default="in.txt")
	parser.add_argument("-vocab", "--vocabulary", help="Save the vocabulary file to entered location", default="build_vocab.txt")
	parser.add_argument("-vocabc", "--vocabulary-count", help="Save the vocabulary file with the frequencies to entered location", default="build_vocab_count.txt")
	parser.add_argument("-vec", "--vector-output", help="Save the word vectors to the given file", default="word_vector_output.txt")
	parser.add_argument("-uc", "--uppercase", help="Let words remain in uppercase", action="store_true")
	parser.add_argument("-wlimit", "--word-limit", help="Set a lower bound for word frequencies", type=int, default=0)
	parser.add_argument("-v", "--verbosity", help="increase output verbosity", action="store_true")
	parser.add_argument("-arguments", "--arguments", help="print out the arguments entered", action="store_true")
	
	args = parser.parse_args()
	
	if (args.arguments):
		print "args.deep_neural", args.deep_neural
		print "args.window_size", args.window_size
		print "args.word_size", args.word_size
		print "args.iteration_count", args.iteration_count
		print "args.input_file", args.input_file
		print "args.epsilon", args.epsilon
		print "args.input_file", args.input_file
		print "args.vocabulary", args.vocabulary
		print "args.vector_output", args.vector_output
		print "args.uppercase", args.uppercase
		print "args.word_limit", args.word_limit
		print "args.verbosity", args.verbosity, "\n"
	
	weight_init()
	#Uses deep_neural, args.word_size*args.window_size as the size of the layer 1
	#Uses 1, deep_neural as the size of the layer 2
	word_vocab_build()
	
	for word_tuple in words:
		if(word_tuple[1]<args.word_limit):
			break
		else:
			continue
	
	word_vec_init()
	
	word_vec_process()