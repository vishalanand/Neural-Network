#python launch_score.py -wlimit 0 -dn 4 -word 6 -window 3 -arguments -in in.txt
import numpy, argparse, os, re, random, math, copy
import numpy as np

from collections import Counter

def sigmoid(X):
	#return 1/(1+np.exp(-X*1.000))
	return X

def dsigmoid(X):
	#return np.multiply(sigmoid(X),(1-sigmoid(X)))
	return 1

def scale_magnitude(X):
	print "The magnitude is",
	print int( math.log10(X) )
	return X
	#return ( X / ( 10**( int( math.log10(X) ) ) ) ) * 100

def slidingWindow(sequence,winSize,step=1):
	global W1
	global W2
	global X
	global vector_update
	
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
		
		abc = [vector_update[x] for x in [x1 for x1 in sequence[i:i+winSize]]]
		abc_temp=abc
		abc_store=abc
		abc = np.asarray(abc)
		abc = np.concatenate(abc)
		abc.resize(len(abc),1)
		
		if(args.window_size%2==0):
			center = args.window_size/2
		else:
			center = (args.window_size-1)/2

		cnt = 1
		repetition = 1
		while (cnt!=0 and repetition<3) :
			#print "\n\n\ncnt", cnt, " repetition", repetition
			repetition = repetition + 1
			cnt = 0
			"""
			First use the middle word to calculate the score
			"""
			#print words
			#print "\nThe window is", current_window;
			#print "\n",current_window[center],"\t: ",
			#print vector_update[current_window[center]]
			
			X = abc_store
			X = np.asarray(X)
			input_size_X = args.word_size*args.window_size
			X = X.reshape(input_size_X, 1)
			add_row = np.array([1])
			add_row = add_row.reshape(1, 1)
			X = np.vstack((add_row,X))
			#print X
			#print "This was X"
			Z1=W1.dot(X)
			#print Z1
			#print "This was Z1"
			A1=np.zeros_like(Z1)
			#print "The value of sigmoid thingy is "
			#print sigmoid(Z1)
			A1=sigmoid(Z1)
			A1=np.concatenate((np.array([[1]]),A1),0)
			#print A1
			#print "This was A1"
			Z2=W2.dot(A1)
			#print Z2
			#print "This was Z2"
			Z2_scoring = np.squeeze(np.asarray(Z2[0][0]))
			#print Z2_scoring
			#print "This was Z2_scoring"
			#print Z2_scoring, "\tfor \"",current_window[center],"\"\n"

			"""
			Then calculate the score for the other words
			"""
		
			for word_tuple in words:
				#print current_window
				#print word_tuple[0], vector_update[word_tuple[0]]
				if(word_tuple[0]!=current_window[center]):
					#print "Wrong word evaluation"
					#print word_tuple[0]
					wrong_window = copy.deepcopy(current_window)
					wrong_window[center] = word_tuple[0]
					#print wrong_window
					#print "This is what I am working on currently"
					#wrong_window[center] = 
					if(args.verbosity==True):
						print "\n",word_tuple[0],"\t: ",
						print vector_update[word_tuple[0]]
					abc_window_temp = abc
					
					if(args.verbosity):
						print "The value of center is", center
					
					for i in xrange ((center)*args.word_size,  (center+1)*args.word_size):
						abc_window_temp[i] = vector_update[word_tuple[0]][i-(center)*args.word_size]

					X = abc_window_temp
					X = np.asarray(X)
					input_size_X = args.word_size*args.window_size
					X = X.reshape(input_size_X, 1)
					add_row = np.array([1])
					add_row = add_row.reshape(1, 1)
					X = np.vstack((add_row,X))
					#print X
					#print "This was X"

					"""
					print "W1's class is", W1.__class__
					print W1.shape
					print W1
					print W1[0]
					print "Yeah!"
					W1 = np.linalg.norm(W1[0])
					print "W1's new class is", W1.__class__
					print W1
					#print W1.shape
					#W2 = np.linalg.norm(W2)
					model["W1"]=np.random.randn(args.deep_neural,input_size+1)*.001
					model["W2"]=np.random.randn(1,args.deep_neural+1)*.001
					print W1
					"""
					Z1=W1.dot(X)
					#print Z1
					#print "This was Z1"
					A1=np.zeros_like(Z1)
					A1=sigmoid(Z1)
					#print A1
					#print "This was A1"
					A1=np.concatenate((np.array([[1]]),A1),0)
					Z2=W2.dot(A1)

					"""
					Z2-=np.max(Z2)
					print "updated Z2 after subtraction is", Z2
					Z2=np.exp(Z2)
					Z2/=np.sum(Z2)
					"""

					#print Z2, "This was THE SHIT, the Z2"
					Z2_scoring_wrong = np.squeeze(np.asarray(Z2[0][0]))
					#print Z2_scoring_wrong, "This was Z2_scoring_wrong"
					#print Z2_scoring, "This was Z2_scoring"

					"""
					abs_error = abs(Z2_scoring_wrong - Z2_scoring)
					if abs_error<1:
						update_backpropagate = 
					"""

					"""
					print Z2_scoring - Z2_scoring_wrong - 1, "\tfor \"",word_tuple[0],"\""
					update_backpropagate = Z2_scoring - Z2_scoring_wrong - 1
					error_value = max(0, update_backpropagate)
					print error_value, " is here", "Bitch\n"
					"""
					
					#print 1 + Z2_scoring - Z2_scoring_wrong, "\tfor \"",word_tuple[0],"\""
					if Z2_scoring > Z2_scoring_wrong + 1:
						error_value = 0
					else:
						#print "abs(Z2_scoring - Z2_scoring_wrong) + 1 =",
						#print Z2_scoring, "-", Z2_scoring_wrong, "+ 1"
						error_value = abs(Z2_scoring - Z2_scoring_wrong) + 1

					"""
					print "The error_value is", error_value
					if(abs(error_value)>1000):
						#error_value = scale_magnitude(error_value)
						if(error_value<0):
							error_value = -1000
						else:
							error_value = 1000
					"""
					"""
					update_backpropagate = 1 + Z2_scoring - Z2_scoring_wrong
					error_value = max(0, update_backpropagate)
					"""
					#print error_value, " is the error\n"

					if error_value<=1 :
						#print "No error found"
						abcd_dummy = 1
					elif error_value>1 :
						margin=1

						grad={}
						dZ2=np.zeros_like(Z2)
						dZ2[0][0]=error_value
						dW2=dZ2.dot(A1.T)
						dA1=np.dot(W2.T,dZ2)
						#removing bias activation
						dA1=dA1[1:]
						dZ1=dsigmoid(Z1)*dA1
						dW1=dZ1.dot(X.T)
						dx=np.dot(W1.T,dZ1)
						dx=dx[1:]

						#reg = 0.001
						reg = 1

						#ADDING REGULARIZATION TO WEIGHTS
						dW1[:,1:]+=reg*W1[:,1:]
						dW2[:,1:]+=reg*W2[:,1:]

						abs_dx_max = abs(dx.max())
						abs_dx_min = abs(dx.min())
						#print abs_dx_max, "is the maximum element in dx"
						#print abs_dx_min, "is the minimum element in dx"
						abs_dx_update = max(abs_dx_min, abs_dx_max)
						if abs_dx_update>1000:
							dx = dx*10/abs_dx_update

						abs_dW1_max = abs(W1.max())
						abs_dW1_min = abs(W1.min())
						#print abs_dW1_max, "is the maximum element in dW1"
						#print abs_dW1_min, "is the minimum element in dW1"
						abs_dW1_update = max(abs_dW1_min, abs_dW1_max)
						if abs_dW1_update>1000:
							dW1 = dW1*10/abs_dW1_update

						abs_dW2_max = abs(dW2.max())
						abs_dW2_min = abs(dW2.min())
						#print abs_dW2_max, "is the maximum element in dW2"
						#print abs_dW2_min, "is the minimum element in dW2"
						abs_dW2_update = max(abs_dW2_min, abs_dW2_max)
						if abs_dW2_update>1000:
							dW2 = dW2*10/abs_dW2_update

						grad["W1"]=dW1
						grad["W2"]=dW2
						grad["X"]=dx
						W1 = W1 + dW1
						W2 = W2 + dW2
						#X = X + dx
						for i in xrange(0,3):
							#print word_tuple[0]

							#print current_window[i], vector_update[current_window[i]]
							#print "The update vector is "
							#print dx

							for j in xrange(0,args.word_size):
								#print vector_update[current_window[i]][j], "\t",
								d_X_value = np.squeeze(np.asarray(dx[i*args.word_size + j]))
								#vector_update[current_window[i]][j] = vector_update[current_window[i]][j] + args.epsilon*d_X_value
								vector_update[current_window[i]][j] = vector_update[current_window[i]][j] + 1*d_X_value
								#print vector_update[current_window[i]][j]
							#print "New vector value is \t",vector_update[current_window[i]]
				"""
				else:
					continue;
				"""
			file_write = open("logs/"+"backpropagation.txt", "a")
			file_write.write("Total errors for the window is " + str(cnt) + " corresponding to the repetition " + str(repetition-1) + "\n")
		#print "Now the cnt is", cnt, " and the repetition", repetition-1
		file_write.write("\n\n")
	return

def word_vocab_build():
	global words
	global vector_update
	
	lines = [line.strip() for line in open(args.input_file)]
	lines = [line.split() for line in lines]
	words = [item for sublist in lines for item in sublist]
	
	if(not args.uppercase):
		words = [x.lower() for x in words]
	
	words.append("DGDD")
	words = Counter(words).most_common(len(words))
	statinfo = (os.stat(args.input_file).st_size)/(1024**2)
	#print "The input file is of size :",statinfo
	
	if(statinfo<10):
		#print words
		i_dont_care = 1
	
	file_output = open("logs/"+args.vocabulary, 'w')
	file_output_count = open("logs/"+args.vocabulary_count, 'w')
	
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
	#print vec_init
	#file_vec_output = open(args.vector_output, 'w')
	file_vec_output_word = open("logs/"+"word_vector_output_verbose_init.txt", 'w')
	global vector_update
	vector_update = {}
	vec_dummy = [1]*args.word_size
	vector_update["DGDD"] = vec_dummy
	#print "DGDD and : ",
	#print vector_update["DGDD"]
	
	for word_tuple in words:
		if(word_tuple[1]<args.word_limit):
			break
		else:
			#vec_init = [random.uniform(1, 10) for _ in range(0, args.word_size)]
			vec_init = [random.uniform(0, 2) for _ in range(0, args.word_size)]
			#model["W1"]=np.random.randn(args.deep_neural,input_size+1)*.001
			vec_init_str=' '.join(str(e) for e in vec_init)
			vector_update[word_tuple[0]] = vec_init
			#file_vec_output.write(vec_init_str+os.linesep)
			file_vec_output_word.write(word_tuple[0]+"\t"+vec_init_str+os.linesep)
	
	#print
	#print

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
		
		#print "We are sending : "
		#print words_matrix
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
	
	#print words_matrix
	#print "The sliding window"
	slidingWindow(words_matrix, args.window_size)

def weight_init():
	bias_print = 10
	
	if(bias_print==1):
		print "Init"
	
	global W1
	global W2
	global model
	model = {}
	input_size = args.word_size*args.window_size
	model["W1"]=np.random.randn(args.deep_neural,input_size+1)*.001
	model["W2"]=np.random.randn(1,args.deep_neural+1)*.001
	W1 = model["W1"]
	W2 = model["W2"]
	global X
	
	if(bias_print==1):
		print "The hidden neural is :",args.deep_neural
		print "The word vector size is :",args.word_size
		print "The window size is :",args.window_size
		print "W = numpy.random.uniform(low=0, high=10, size=(args.deep_neural,args.word_size*args.window_size))"
		print "The weight matrix is :"
		print W.shape
		print "W is"
		print W

def word_vec_print():
	global vector_update
	global words
	#print words
	#print "Printing the vectors"
	file_vec_output_word_final = open("logs/"+"word_vector_output_verbose_final.txt", 'w')
	for word_tuple in words:
		if(word_tuple[1]<args.word_limit):
			break
		else:
			print_word_vector_str=' '.join(str(e) for e in vector_update[word_tuple[0]])
			file_vec_output_word_final.write(word_tuple[0]+"\t"+print_word_vector_str+os.linesep)
	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
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
	word_vocab_build()
	word_vec_init()
	word_vec_process()
	word_vec_print()
	"""
	print "The dimensions of the variable W1 is",
	print W1.__class__
	print W1.shape
	print "The dimensions of the variable W2 is",
	print W2.__class__
	print W2.shape
	"""
	numpy.savetxt("logs/"+"W1.csv", W1, delimiter=",")
	numpy.savetxt("logs/"+"W2.csv", W2, delimiter=",")