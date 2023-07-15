#include <cstdlib>
#include <fstream>
#include <sstream>

void get_y_test (float * y_test, const char * path, int NB_INFER){
    
    std::ifstream src; 
    
    src.open(path);
    if (!src)
    {
    	std::cerr << "\a error reading y test data\n"; 
		exit(EXIT_FAILURE); 
    }   
    	
    std::string buffer; 
    int j; 	
    for (j=0; j < NB_INFER ; j++){
		std::getline(src, buffer);
		y_test[j] = std::stof(buffer); 
		
		
    }		
    src.close();	
}


void  get_x_test (float ** X_test, const char * path,  int NB_INFER, int NB_FEATURES){
    std::ifstream src; 
    std::string buffer; 
    unsigned int strpos=0, endpos=0; 
    char sep =',';
   
    src.open(path); 
    if (!src)
    {
    	std::cerr << "\a error reading x test data\n"; 
		exit(EXIT_FAILURE); 
    }   
	
  int i; 
  for (int j=0; j < NB_INFER ; j++){
	std::getline(src, buffer);
	endpos = buffer.find(sep);
	for (i=0; i<NB_FEATURES; i++){
		if (i==NB_FEATURES-1) 
			endpos--;
		X_test[j][i] = std::stof(buffer.substr(strpos,endpos - strpos));
		strpos = endpos + 1;
		endpos = buffer.find(sep, strpos);
	}
			
	strpos = endpos+1;
	
	}
  src.close();	 
}


