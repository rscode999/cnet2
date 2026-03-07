MAIN='main.cpp' #Put your path to a file containing `main` here
OUTPUT_EXECUTABLE_NAME='a' #put your desired executable name here
EIGEN_DIRECTORY_PATH='eigenlite/' #Put your path to the Eigen 3 folder here

c: # Standard compilation, with Optimization Level 2. Should be used for code verification.
	g++ ${MAIN}  -o ${OUTPUT_EXECUTABLE_NAME}  -std=c++14  -I ${EIGEN_DIRECTORY_PATH} -O2 -Wall -Werror -Wpedantic

eigen_lite:
	g++ ${MAIN}  -o ${OUTPUT_EXECUTABLE_NAME}  -std=c++14  -I 'eigen_lite/' -O2 -Wall -Werror -Wpedantic

debug: # Optimization Level 0, for use with debuggers
	g++ ${MAIN}  -o ${OUTPUT_EXECUTABLE_NAME}  -std=c++14  -I ${EIGEN_DIRECTORY_PATH} -O0 -Wall -Werror -Wpedantic

optimized: #Optimization Level 3, for maximum speed
	g++ ${MAIN}  -o ${OUTPUT_EXECUTABLE_NAME}  -std=c++14  -I ${EIGEN_DIRECTORY_PATH} -O3 -Wall -Werror -Wpedantic