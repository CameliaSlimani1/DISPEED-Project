// Inspired by SampleOnnxMNIST.cpp 
// Created by Camélia on 15/03/2022 

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <chrono>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include "NvInferRuntime.h"
#include "NvCaffeParser.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"

#include "logger.h"
#include "sampleUtils.h"
#include "sampleOptions.h"
#include "sampleEngines.h"
#include "../utils.h"
const std::string gSampleName;// = "TensorRT.dnn_dsd";
char * x_test_path; 
char * y_test_path; 
int  NB_INFER;// = 80000; 
int  NB_FEATURES;// = 25; 
int N_CLASSES ;//= 10; 
int BATCH_SIZE; 
/************ Added by Camélia *********************/



class DNNBalanced
{
	 template <typename T> using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
public : 
	  DNNBalanced(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }
	
    bool build();

    bool infer();

    //ICudaEngine* loadEngine();

private:
    samplesCommon::OnnxSampleParams mParams;


    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify


     std::shared_ptr<nvinfer1::ICudaEngine> mEngine; 
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    bool processInput(const samplesCommon::BufferManager& buffers);

    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};




bool DNNBalanced::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);     
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }
    std::cout<<"Trace 1 " << std::endl;
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
    std::cout<<"Trace 2 " << std::endl;
    
    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }
   
    
    auto constructed = constructNetwork(builder, network, config, parser);
    
    if (!constructed)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }
     
     
    mInputDims = network->getInput(0)->getDimensions();
    
    mOutputDims = network->getOutput(0)->getDimensions();
     std::cout<<"Input " <<  mInputDims << "   " << mOutputDims<<std::endl;
    return true;
}


bool DNNBalanced::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
      std::cout<<locateFile(mParams.onnxFileName, mParams.dataDirs).c_str() << std::endl; 
 
    auto parsed = parser->parseFromFile(
        locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }
    
    config->setMaxWorkspaceSize(1_GiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}







bool DNNBalanced::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    time_t start_dataload, end_dataload; // start_infer, end_infer; 
    
    time(&start_dataload); 
    
    std::cout<<"Read X_test" <<std::endl; 
	
    float ** X_test; 
    X_test = new float*[NB_INFER];
    for (int i=0; i<NB_INFER; i++)
	X_test[i] = new float[NB_FEATURES]; 
	
    get_x_test(X_test, x_test_path, NB_INFER, NB_FEATURES); 

    std::cout<<"Read y_test" <<std::endl; 
    float y_test[NB_INFER]; 
    get_y_test(y_test, y_test_path, NB_INFER); 
    time(&end_dataload); 

    sample::gLogInfo << "Begin Inference  "<< BATCH_SIZE << "  " << NB_FEATURES << " " << NB_INFER << std::endl;
    context->setBindingDimensions(0, nvinfer1::Dims3(BATCH_SIZE, NB_FEATURES,1));
    //context->setBindingDimensions(0, nvinfer1::Dims2(BATCH_SIZE, NB_FEATURES));
    
    int i = 0, j=0, class_predicted, match=0;
    float max_score;
    
    auto start_infer = std::chrono::high_resolution_clock::now();
    for (i=0; i < NB_INFER ; i+=BATCH_SIZE){
 	
	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
	for (int k=0; k<BATCH_SIZE;k++){
		for (j=0; j< NB_FEATURES; j++){
			hostDataBuffer[k*NB_FEATURES+j] = X_test[i+k][j]; 
		}
 	}
	buffers.copyInputToDevice();
	bool status = context->executeV2(buffers.getDeviceBindings().data());
    	if (!status)
    	{
        	return false;
    	}
	buffers.copyOutputToHost();
	
	float* output = 	static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
        
	for (int k=0; k<BATCH_SIZE; k++){	
	max_score=output[N_CLASSES*k];
	class_predicted=0;
	for (int l=N_CLASSES*k+1; l<N_CLASSES*k+N_CLASSES; l++){
		if (output[l] > max_score) {
			max_score=output[l];
			class_predicted=l%N_CLASSES;
			 
		}
	}
		if (class_predicted == y_test[i+k]){
			match++;
		}
	}
  }
    auto end_infer = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_infer-start_infer); 
    sample::gLogInfo << "Match: " << match << std::endl;
    sample::gLogInfo  << "Accuracy on inference platform: " << float(match)/NB_INFER *100 << " % " << std::endl;

    sample::gLogInfo << "infer time: " <<elapsed.count() << "ms " << std::endl;
    char cmd[100]; 
    sprintf(cmd, "grep ^VmPeak /proc/%d/status", getpid()); 
    system(cmd); 
    return true;
}


samplesCommon::OnnxSampleParams initializeSampleParams( char ** args, const int argc)
{
    samplesCommon::OnnxSampleParams params;
 

    for (int i = 1; i < argc; ++i)
    {
        if (!strncmp(args[i], "--model=", 8))
        {
	   
	   params.onnxFileName = (args[i] + 8);
	   params.dataDirs.push_back("./");
           params.dataDirs.push_back("./");
        }
	else if (!strncmp(args[i], "--inputTensorName=", 18))
        {
	    params.inputTensorNames.push_back((args[i] + 18));
	
        }
	else if (!strncmp(args[i], "--outputTensorName=", 19))
        {
	   params.outputTensorNames.push_back((args[i] + 19));
	
        }
        else if (!strncmp(args[i], "--int8", 6))
        {
            params.int8 = true;
        }
        else if (!strncmp(args[i], "--fp16", 6))
        {
            params.fp16 = true;
        }
        else if (!strncmp(args[i], "--useDLACore=", 13))
        {
            params.dlaCore = std::stoi(args[i] + 13);
        }
	else if (!strncmp(args[i], "--xtestPath=", 12))
	{
	    x_test_path= args[i]+12;
	} 
	else if (!strncmp(args[i], "--ytestPath=", 12))
	{
	    y_test_path= args[i]+12;
	} 
	else if (!strncmp(args[i], "--nbInfer=", 10))
	{
	    NB_INFER= std::stoi(args[i]+10);
	} 
	else if (!strncmp(args[i], "--nbFeatures=", 13))
	{
	    NB_FEATURES= std::stoi(args[i]+13);
	} 
	else if (!strncmp(args[i], "--nbClasses=", 12))
	{
	    N_CLASSES= std::stoi(args[i]+12);
	  
	} 
	else if (!strncmp(args[i], "--batchSize=", 12))
	{
	    BATCH_SIZE= std::stoi(args[i]+12);
	    sample::gLogInfo << BATCH_SIZE << std::endl;
	} 
        else
        {
            sample::gLogError << "Invalid Argument: " << args[i] << std::endl;
        }
    }
 
    return params;
}


int main(int argc, char** argv)
{


    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    DNNBalanced sample(initializeSampleParams(argv, argc));

    sample::gLogInfo << "Building and running a GPU inference engine for Simple DNN " << std::endl;
    
    	

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
