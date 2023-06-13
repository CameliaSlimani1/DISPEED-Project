# Security, Performance, Energy Trade-off for Intrusion Detection Systems - DISPEED Project 

## Project Description 
Drones, when working in swarms, are supposed to gain more autonomy and efficiency during their mission.
Yet, security threats and low energy levels can disrupt the progress
of the mission. The presented work in this paper is part
of a project that aims to propose strategies for deploying
IDSs on swarms of drones. The proposed strategies would
make it possible to achieve, at runtime, a trade-off between
correctly (accuracy) and rapidly (latency) detecting intrusions
according to mission criticality level and traffic load and the
energy and resource (computation and memory) usage of the
ran IDS. For this sake, the project was subdivided into 3
steps:
 1. Forming an exploration space of IDS models ; 
 2. Implementing, optimizing and characterizing them on multiple platforms ;
 3. Proposing online strategies of IDS mapping on
heterogeneous computing elements and memory capabilities
of a single drone.

![alt text](https://github.com/CameliaSlimani1/DISPEED_Project_demo/blob/main/docs/img/overview.png)

## Repository Structure
This repository contains the following elements : 
1. [src]( https://github.com/CameliaSlimani1/DISPEED_Project_demo/tree/main/src "src") : it contains the code to generate IDSs, describe Platforms and Implementations, and generate reports. The entities and utility codes are described hereafter : 
   * [entities](https://github.com/CameliaSlimani1/DISPEED_Project_demo/tree/main/src/entities) : it containes the code for abstract entities related to the project (IDSmodel, platform, implementation, dataset, etc.). 
   * [utils](https://github.com/CameliaSlimani1/DISPEED_Project_demo/tree/main/src/utils) : this directory contanins utility codes to generate IDS models (RF, CNN, DNN) and performance reports. 
3. [output](https://github.com/CameliaSlimani1/DISPEED_Project_demo/tree/main/output "output") : it constains the output data generated from the project : 
   * [Platforms](https://github.com/CameliaSlimani1/DISPEED_Project_demo/tree/main/output/Platforms) : this directory contains JSON files of the platforms used in the characterization process ; 
   * [Implementations](https://github.com/CameliaSlimani1/DISPEED_Project_demo/tree/main/output/Implementations) : it containes JSON files of the charaterization of the IDS models on platforms. The JSON files include : the model characterized, the used platform, the volume of data on which the charecterization was run, model's accuracy, model's F1-score, inference time in milliseconds, memory peak in Megabytes, energy consumed in Joules, and a short textual description of the model. 
   * [models](https://github.com/CameliaSlimani1/DISPEED_Project_demo/tree/main/output/models) : it containes the preprocess/IDS models that were trained for this project.    



