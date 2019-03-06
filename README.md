# This are basics implementations of some machine learning algorithms.
## Reads and visualize data file
__readFile(datafile, columns)__  
This function reads the data file initialy with no header.  
The __columns__ : is an array of character that correspond to each column of the data set.  
The __dataFile__ : The path to the data set  
The function used for visualisation is : __visualizeData(dataFile, columns)__  
For the case of the __iris__ dataset, the columns argument is set as : ['x1', 'x2', 'x3', 'x4', 'class']
## Minkwoski distance 
def minkowski(a, b, r) 
where a and b are vectors and r a parameter
## Distance between elements of the test file and elements of the train file
__DistTrainDataTestData(trainFile, testFile, columns, dataColumns)__  
__testFile__: is the path to the test file  
__dataColumns__ : is a list of columns that holds the data, without the class column
## Split dataset 
splitDataFrame(dataFrame, part=1, partition=2)  
This function will split you dataFrame into __2 partitions__ and returns the 
__1 (first) part__