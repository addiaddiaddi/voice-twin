# Final linalg project (senior yr hs)
Classifying voices with Fourier Transform and KNN.  
Do not worry about code quality. This was a math class 🙂. 

# record.py
Upon running the program, it will prompt you for a name and number of samples to take. You can then press enter, and speak quickly. It will record your voice and perform the fourier transform on it. The transform will display, and the program will prompt you if you would like to keep that sample, or discard it.

# knn.py
Upon running this, it will create a knn model with the provided data, segmenting out 30% for test cases. It will then create an adjacency matrix with the euclidean distance between all of the data points and then print it, with the red concentration of the value correlating with distance. It will then produce a 2D and 3D chart of the adjacency matrix using a spring algorithm (edges become "springs" with a spring constant inversely proportional to the distance), creating a mesh network of data points (nodes/voices) allowing humans to visualize the similarities between data points. 
