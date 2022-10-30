==FUNCTION==
RecogniseFace([image], [featureType], [classifierType])

RecogniseFace function accepts three arguments:

image - The image you want to input, reassign the variable fileName to the desired image file
	Make sure that the images are relative to the Script file.

featureType - The desired feature type, replace with:
		"surf" for SURF Feature Extractor
		"hog" for HOG Feature Extractor

classifierType - The desired classifier type, replace with:
		"svm" for SVM classifier
		"cnn" for CNN classifier

==IMPORTANT==
Arguments/inputs are case sensitive - use lower cases.
Ensure any files are relative to the Script file and not in any sub-folders or other directory.
Feature and Classifier Combinations not implemented in the function will not work and simply output a statement.
Do not use Square Brackets.