#include <tensorflow/lite/c/c_api.h>
#include <iostream>
#include "../utils.h"


int main(int argc, char **argv) {

  const char* model_path = argv[1];
  const char* xtest_path = argv[2]; 
  const char* ytest_path = argv[3]; 
  int NB_INFER = atoi(argv[4]); 
  int NB_FEATURES =atoi(argv[5]); 
  int NB_CLASSES = atoi(argv[6]);
 
  // Load the TensorFlow Lite model
  TfLiteModel* model = TfLiteModelCreateFromFile(model_path);
  if (!model) {
    std::cerr << "Failed to load model: " << model_path << std::endl;
    return 1;
  }

  // Create the interpreter
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  TfLiteInterpreterOptionsDelete(options);
  if (!interpreter) {
    std::cerr << "Failed to create interpreter." << std::endl;
    TfLiteModelDelete(model);
    return 1;
  }

  // get Data 
   float ** X_test; 
    X_test = new float*[NB_INFER];
    for (int i=0; i<NB_INFER; i++)
	X_test[i] = new float[NB_FEATURES]; 
	
    get_x_test(X_test, xtest_path, NB_INFER, NB_FEATURES); 

    std::cout<<"Read y_test" <<std::endl; 
    float y_test[NB_INFER]; 
    get_y_test(y_test, ytest_path, NB_INFER); 

  // Allocate tensors
  if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
    TfLiteInterpreterDelete(interpreter);
    TfLiteModelDelete(model);
    return 1;
  }

  // Get the input tensor
  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  if (!input_tensor) {
    std::cerr << "Failed to get input tensor." << std::endl;
    TfLiteInterpreterDelete(interpreter);
    TfLiteModelDelete(model);
    return 1;
  }
   float * y_obtained; 
   int match =0; 
   y_obtained = new float[NB_CLASSES];
    
  // Perform inference
  for (int i=0; i < NB_INFER; i++){
	 TfLiteTensorCopyFromBuffer(input_tensor, X_test[i], NB_FEATURES*sizeof(float)); 
	 TfLiteInterpreterInvoke(interpreter); 
	 
        // Get the output tensor
 	 const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
 	 if (!output_tensor) {
 		   std::cerr << "Failed to get output tensor." << std::endl;
  		   TfLiteInterpreterDelete(interpreter);
    		   TfLiteModelDelete(model);
    			return 1;
 	 }
	TfLiteTensorCopyToBuffer(output_tensor, y_obtained, NB_CLASSES*sizeof(float)); 
       	float max_score=y_obtained[0];
	int class_predicted=0;
	for (int k=1; k<NB_CLASSES; k++){
		if (y_obtained[k] > max_score) {
			max_score=y_obtained[k];
			class_predicted=k;
		}
	}
        if (class_predicted == y_test[i]){
		match++;	
	}
   	
 
 }
 
 std::cout << "Accuracy : " << (float) match/NB_INFER << std::endl;
 
 
  // Cleanup
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  delete(X_test); 
  delete(y_obtained); 
  return 0;
}

