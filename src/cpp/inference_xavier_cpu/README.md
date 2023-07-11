This code was tested on an NVIDIA Xavier AGX under Linux Tegra 4.9.10. 
The used TensorFlow version is : TensorFlow 2.14.0. 


To run inferences using the Xavier CPUs use the following steps : 
1. Compile using the makefile. Note that the environment variable TF_DIR must be set to the TensorFlow directory path. 
2. Launch test : ./output <model.tflite> <X_test.csv> <y_test.csv> <NB_INFERENCES> <NB_FEATURES> <NB_CLASSES>
