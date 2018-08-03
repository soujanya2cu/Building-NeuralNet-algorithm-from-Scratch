
/*macro to put variables into a macro variable */
%macro GENERATEVARLIST(DSN=, EXCLUDE=);
	%global VARNAME;
 /*SELECTING ALL THE VARIABLES TO BE IMPUTED*/

	%IF %INDEX(&DSN, .) %THEN
		%DO;
			%LET LIB=%UPCASE(%SCAN(&DSN, 1, .));
			%LET DATA=%UPCASE(%SCAN(&DSN, 2, .));
		%END;
	%ELSE
		%DO;
			%LET LIB=WORK;
			%LET DATA=%UPCASE(&DSN);
		%END;
	%LET NEXC=%SYSFUNC(COUNTW(&EXCLUDE, %STR( )));

	PROC SQL NOPRINT;
		SELECT NAME INTO: VARNAME SEPARATED BY ' ' FROM DICTIONARY.COLUMNS WHERE 
			UPCASE(LIBNAME)="&LIB" AND UPCASE(MEMNAME)="&DATA" AND NAME NOT 
			IN("%SCAN(&EXCLUDE,1,%STR( ))" %DO A=2 %TO &NEXC;
			, "%SCAN(&EXCLUDE,&A,%STR( ))" %END;
		);
	QUIT;

%mend;

/*macro to calculate the number of observations in the Train dataset*/
%macro TOTALTRAINOBS(DSN=);
%global TRAINNUMOBS;
proc sql noprint;
select count(*)
into :TRAINNUMOBS
from &DSN;
quit;
%mend;

/*macro to calculate the number of observations in the Valid dataset*/
%macro TOTALVALIDOBS(DSN=);
%global VALIDNUMOBS;
proc sql noprint;
select count(*)
into :VALIDNUMOBS
from &DSN;
quit;
%mend;

/*output the number of observations in the dataset*/
%TOTALTRAINOBS(DSN=SOU_FOR.sample_trainfornntest)
%put &TRAINNUMOBS; 

/*output the number of observations in the dataset*/
%TOTALVALIDOBS(DSN=SOU_FOR.sample_validfornntest)
%put &VALIDNUMOBS; 

 /*Standardzing the training dataset*/
proc stdize data= SOU_FOR.sample_trainfornntest out= curnnetvar_stdtrain method=std;
var AMS3013 AMS3022 AMS3023 AMS3856 ;
run;
proc stdize data= SOU_FOR.sample_validfornntest out= SOU_FOR.curnnetvar_stdvalid method=std;
var AMS3013 AMS3022 AMS3023 AMS3856 ;
run;


/* Shuffling the dataset*/
data SOU_FOR.curnnetvar_stdtrain;
 set curnnetvar_stdtrain;
 seed=100;
 shuffling=ranuni(seed);
 run;
proc sort data=SOU_FOR.curnnetvar_stdtrain;
 by shuffling;
  run;
data SOU_FOR.curnnetvar_stdtrain(drop= seed shuffling);
 set SOU_FOR.curnnetvar_stdtrain;
  run;


DATA curnnetvar1(DROP=RESP_DV) ;
SET SOU_FOR.curnnetvar_stdtrain(obs=1);
RUN;

/*************************************************************************************
* Macro #1 :NeuralTest
* Description: Created main macro created macro with the below paramters


* Following are the Parameters defined for the Macro:
* dsn                      Dataset name
* hiddenextension          Extension for hidden layer weights & bias
* outextension             Extension for output layer weights & bias
* outputnodes              Number of output nodes in neural network
* LR                       Learning rate
* StartIteration           Iteration number to start with
* EndIteration             Iteration number to end with
*
**************************************************************************************/
%macro NeuralTest(dsn=, hiddenextension=, hiddennodes=, outextension=, outputnodes=, LR=, StartIteration=, EndIteration=);
/* Open the data set */ 
  %let dsid=%sysfunc(open(&dsn));
/* The variable nvar will contain the number of variables that are in the dataset*/
  %let nvar=%sysfunc(attrn(&dsid, nvars));

/* Using the GENERATEVARLIST macro to gnerate variables names in the datasets*/
%GENERATEVARLIST(DSN=SOU_FOR.curnnetvar_stdtrain, exclude=RESP_DV MK);
%put &varname;
%let inputvarname = &varname;

/***************** Preparing the data for Front Propagation*******************************************************/
 
 /* Generating variables for Hidden layer weights and Bias by adding Extension and then 
 initialize weights and biases with random values*/
 data hidden_weights (drop=&inputvarname);
 set &dsn;
    %do p=1 %to &nvar;
    %let var=%sysfunc(varname(&dsid, &p));
         %do q=1 %to &hiddennodes;
		 &var=&var&hiddenextension&q;
		 &var&hiddenextension&q=rand('uniform');
		 %end;
	%end;
/* Close the data set */ 
	%let rc=%sysfunc(close(&dsid));
	run;
 
%GENERATEVARLIST(DSN=hidden_weights, exclude=RESP_DV MK);
%put &varname;
%let hidden_weights_name = &varname;
%put &hidden_weights_name; 
   
 data hidden_intercept;
	%do r=1 %to &hiddennodes;
	BIAS_H&r=rand('uniform');
	 %end;
	  run;
	 
%GENERATEVARLIST(DSN=hidden_intercept, exclude=RESP_DV MK);
%put &varname;
%let hidden_interceptname= &varname;
%put &hidden_interceptname;
     
 data hidden_node;                                              
   %do s=1 %to &hiddennodes;
    H&s=0;
    %end;
    run;
     
%GENERATEVARLIST(DSN=hidden_node, exclude=RESP_DV MK);
%put &varname;
%let hidden_nodename= &varname;
%put &hidden_nodename;

 /* Randomly generating the weights for output node and bias*/
%let dsn2 = hidden_node;
%put &dsn2;
%let dsid2=%sysfunc(open(&dsn2));
%put &dsid2;
%let n=%sysfunc(attrn(&dsid2, nvars));

  data output_weights (drop=&hidden_nodename);
  set hidden_node;
    %do t=1 %to &hiddennodes;
	%let var=%sysfunc(varname(&dsid2, &t));
         %do u=1 %to &outputnodes;
		 &var=&var&outextension;
		 &var&outextension=rand('uniform');
		 %end;
	%end;
   %let rc=%sysfunc(close(&dsid2));
	    run;
 
%GENERATEVARLIST(DSN=output_weights, exclude=RESP_DV MK);
%put &varname;
%let output_weights_name = &varname;
%put &output_weights_name;

  data out_intercept;
  BIAS&outextension=rand('uniform');
   run;

%GENERATEVARLIST(DSN=out_intercept, exclude=RESP_DV MK);
%put &varname;
%let out_interceptname= &varname;
%put &out_interceptname;
	
  data output_node;
			OutputZ=0;
			 run;
			
%GENERATEVARLIST(DSN=output_node, exclude=RESP_DV MK);
%put &varname;
%let output_nodename= &varname;
%put &output_nodename;

  data error;
   NNetError=0;
 NNetErrorSQ = 0;
 NNetPred1 = 0;
 NNetPred0=0;NNPrediction=0; NNCorrect=0;
  run;

%GENERATEVARLIST(DSN=error, exclude=RESP_DV MK);
%put &varname;
%let error_nodename = &varname;
%put &error_nodename; 

/***************** Preparing the data for Back Propagation*******************************************************/
/* Generating variables for Hidden layer weights and Bias by adding Extension and then 
 initialize weights and biases with random values*/

 data deltahidden_node;
	%do dhn=1 %to &hiddennodes;
		DELTAH&dhn=0;
	%end;
	run;
			 
%let dsn3 = output_weights;
%put &dsn3;
%let dsid3=%sysfunc(open(&dsn3));
%put &dsid3;			 
			
 data deltaoutput_weights(drop=&output_weights_name);
 set output_weights;
   %do dow=1 %to &hiddennodes;
	%let var=%sysfunc(varname(&dsid3, &dow));
	%put &var;
	&var=delta&var;
	delta&var=0;
	%put &var;
   %end; 
%let rc=%sysfunc(close(&dsid3));
run;

%let dsn4 = hidden_intercept;
%put &dsn4;
%let dsid4=%sysfunc(open(&dsn4));
%put &dsid4;		

data deltahidden_intercept(drop=&hidden_interceptname);
	set hidden_intercept;
	%do dhi=1 %to &hiddennodes;
	%let var=%sysfunc(varname(&dsid4, &dhi ));
	%put &var;
	&var=delta&var;
	delta&var=0;
	%put &var;
	%end; 
%let rc=%sysfunc(close(&dsid4));			 
			 

%let dsn5 = hidden_weights;
%put &dsn5;
%let dsid5=%sysfunc(open(&dsn5));
%put &dsid5;		

%let dhiddenvar=%sysevalf(&nvar*&hiddennodes);
%put &dhiddenvar;
data deltahidden_weights(drop=&hidden_weights_name);
	set hidden_weights;
	
	%do dhw=1 %to &dhiddenvar;
	%let var=%sysfunc(varname(&dsid5, &dhw ));
	%put &var;
	&var=delta&var;
	delta&var=0;
	%put &var;
	%end; 
%let rc=%sysfunc(close(&dsid5));
run;	
	
	
			 
%GENERATEVARLIST(DSN=deltahidden_node, exclude=RESP_DV MK);
%put &varname;
%let deltahidden_nodename= &varname;
%put &deltahidden_nodename;

%GENERATEVARLIST(DSN=deltaoutput_weights, exclude=RESP_DV MK);
%put &varname;
%let deltaoutput_weightname= &varname;
%put &deltaoutput_weightname;

%GENERATEVARLIST(DSN=deltahidden_intercept, exclude=RESP_DV MK);
%put &varname;
%let deltahidden_interceptname= &varname;
%put &deltahidden_interceptname;

%GENERATEVARLIST(DSN=deltahidden_weights, exclude=RESP_DV MK);
%put &varname;
%let deltahidden_weightname= &varname;
%put &deltahidden_weightname;

 data neuraltestdata;
 merge hidden_weights hidden_intercept hidden_node output_weights out_intercept output_node error ;
  run;

/* Passing initial weights to test the algorithm with pythom inbuilt algorithm*/
data neuraltestdata;
set neuraltestdata;	
AMS3008_h1=	0.36737435;
AMS3013_h1=	-0.20555103;
AMS3022_h1=	0.32545713;
AMS3023_h1=	-0.404579;
AMS3856_h1=	0.21787196;
AMS3008_h2=	-0.4128615;
AMS3013_h2=	-0.06723996;
AMS3022_h2=	0.18460921;
AMS3023_h2=	0.2435651;
AMS3856_h2=	-0.36470094;
AMS3008_h3=	0.40989003;
AMS3013_h3=	0.09232526;
AMS3022_h3=	0.30514421;
AMS3023_h3=	0.50198019;
AMS3856_h3=	-0.14147857;
BIAS_H1=0.522360765;
BIAS_H2=0.545206041;
BIAS_H3=0.047350888;
H1_RESP_DV=0.47802999;
H2_RESP_DV=	0.14920203;
H3_RESP_DV=	-0.28025479;
BIAS_RESP_DV=0.02469232;
 run;
   



data neuraltestdata_delta ;
merge neuraltestdata deltahidden_weights deltahidden_intercept deltahidden_node deltaoutput_weights ;
run;
/*End of Initializing delta variables for nodes and weights   */

/***************************** Macro for FRONT PROPAGATION ***********************/
%MACRO FRONTPROPAGATION (fpdsn=, fpdsnout=);

  data &fpdsnout(drop=k l a b);
  set &fpdsn;
  array inputdata(*) &inputvarname;
  array hiddenweights(&nvar, &hiddennodes) &hidden_weights_name;
  array hiddenintercepts(*) &hidden_interceptname;
  array hiddennodes(*) &hidden_nodename;
  array outweights (&hiddennodes, &outputnodes) &output_weights_name;
  array outintercepts(*) &out_interceptname;
  array outputnodes(*) &output_nodename;
	

/*To calculate the hidden node values by multiplying each input variable
with their correspnding weights and summing up across all variables.
This process is repeated for all nodes and the corresponding values are stored in hidden node variables*/
	do k=1 to &hiddennodes;
	    put k= hiddennodes(k)= hiddenintercepts(k)=;
	     hiddennodes(k) =  hiddennodes(k)+ hiddenintercepts(k);
       do l=1 to &nvar;
          put k=l=hiddennodes(k)=;   
		 hiddennodes(k)=hiddennodes(k)+(inputdata(l) * hiddenweights(l, k));
	
/*This step is to check if the multiplication is happening between variables and their 
corresponding weights for each node in the log*/
		 put k=l=inputdata(l)=hiddenweights(l, k)=;
	  end;
        
        put k=l=hiddennodes(k)=;
        hiddennodes(k)= 1/(1+exp(-hiddennodes(k)));
        put k=l=hiddennodes(k)=;
    end;
          
    do a = 1 to &outputnodes;
        put a=   outputnodes(a)=   outintercepts(a)=;
        outputnodes(a) = outintercepts(a);
        put a=   outputnodes(a)=   outintercepts(a)=;
      do b = 1 to &hiddennodes;
        put a=b=outputnodes(a)=;
        outputnodes(a)=outputnodes(a)+(hiddennodes(b) * outweights(b,a));
        put a=b=hiddennodes(b)= outweights(b,a)=;
	  end;
       put a=b=outputnodes(a)=;
        outputnodes(a)= 1/(1+exp(-outputnodes(a)));
        put a=b=outputnodes(a)=;
    end;
        
  NNetError=RESP_DV-OutputZ; NNetErrorSQ = NNetError*NNetError;
  NNetPred1 = OutputZ;
  NNetPred0=1-NNetPred1;
  if NNetPred1 > NNetPred0 then NNPrediction= 1;
  else NNPrediction= 0;
  NNCorrect = (NNPrediction=RESP_DV);
  run;
  %MEND ;


/* **************BACKPROPAGATION Macro  ******************************************* */
%MACRO BACKPROPAGATION (bpdsn=, bpdsnout= );

 data &bpdsnout(drop=k l);
set &bpdsn;
array inputdata(*) &inputvarname;
array hiddennodes(*) &hidden_nodename;
Array deltahiddennodes(*) &deltahidden_nodename;
Array deltahiddenoutweight(*) &deltaoutput_weightname;
Array deltahiddenintercept(*) &deltahidden_interceptname;
  array hiddenintercepts(*) &hidden_interceptname;
 array outweightsbp (*) &output_weights_name;
array hiddenweights(&nvar, &hiddennodes) &hidden_weights_name;
array deltahiddenweights(&nvar, &hiddennodes) &deltahidden_weightname;
/*Error responsibility for Node Z, an output node*/
/*DELTAZ=outputZ (1−outputZ )(actualZ − outputZ )*/
DELTAZ= OutputZ*(1-OutputZ)*(Resp_DV-OutputZ);
put DELTAZ=;
/*Now adjust “constant” weight w0Z using rules*/
LR=&LR;
put LR=;
deltaBIAS_RESP_DV = LR*DELTAZ*1; 
put deltaBIAS_RESP_DV= ;
put BIAS_RESP_DV=;
BIAS_RESP_DV=(BIAS_RESP_DV) + (deltaBIAS_RESP_DV);
put deltaBIAS_RESP_DV=;
put BIAS_RESP_DV=;
/*Move upstream to Node H1 H2 & H3, hidden layer nodes*/
/*Only node downstream from Node H1, H2, H3 is Node Z*/
do k=1 to &hiddennodes;
/*deltaH1 = H1*(1-H1)*H1_RESP_DV*deltaZ*/
deltahiddennodes(k)= hiddennodes(k)*(1-hiddennodes(k))*outweightsbp (k)*DELTAZ;
/*Adjust weight wAZ using back-propagation rules*/
/*Δh1_RESP_DV= η*δZ*OUTPUT H1 */
deltahiddenoutweight(k)=LR*DELTAZ*hiddennodes(k);
put k=outweightsbp(k)=deltahiddenoutweight(k)=;
 /*h1_RESP_DV = h1_RESP_DV + Δh1_RESP_DV*/
outweightsbp(k) =outweightsbp(k) + deltahiddenoutweight(k);
put k=outweightsbp(k)=;
put k=deltahiddenintercept(k)=deltahiddennodes(k)=;
/*ΔBIAS_h1 = η*deltaH1 *1*/
deltahiddenintercept(k)=LR*deltahiddennodes(k)*1;
put k=deltahiddenintercept(k)=deltahiddennodes(k)=;
/*BIAS_h1 =BIAS_h1 + ΔBIAS_h1*/
hiddenintercepts(k)=hiddenintercepts(k)+deltahiddenintercept(k);
put k=hiddenintercepts(k)=;

end;
       do k=1 to &hiddennodes;
	   
       do l=1 to &nvar;
          put k=l=deltahiddenweights(l,k)= LR= deltahiddennodes(k)= inputdata(l)=;
         /* ΔAMS3008_H1 =  η*deltaH1 *AMS3008 */
		 deltahiddenweights(l,k)= LR* deltahiddennodes(k)*inputdata(l);
	     put k=l=deltahiddenweights(l,k)=;
		 end;
         end;
    
      do k=1 to &hiddennodes;
	   
       do l=1 to &nvar;
          put k=l=hiddenweights(l,k)= deltahiddenweights(l,k)=;
          /*AMS3008_H1 = AMS3008_H1 + ΔAMS3008_H1*/
		 hiddenweights(l,k)= hiddenweights(l,k)+deltahiddenweights(l,k);
	     put k=l=hiddenweights(l,k)=;
		 end;
         end;
         run;

%MEND;

/* created a dummy dataset so that SAS does not throw any error when started the first iteration*/
 data SOU_FOR.neuraltestdataFPTRAIN_ITR0;
 set neuraltestdata;
 run;
 
 
/* If Iteration is starting from 1 we will genearate the dataset with random weights for Front Propgation
Else will take the wieghts from the previous batch*/

%IF &StartIteration=1 %THEN
%do;

data neuraltestdataFP;
set neuraltestdata_delta;
run;
 %end;

%ELSE;

%do;
%let X= %SYSEVALF(&StartIteration -1);
%put &X;
data neuraltestdataFP;
set SOU_FOR.neuraltestdataFPTRAIN_ITR&X;
run;
 %end;

/******************** Logic ended here **************************************************************************/

%MACRO NEURALITERATION;

%do ITER= &StartIteration %to &EndIteration;
        %do loop=1 %to &TRAINNUMOBS; /* This loop runs until the end of the dataset */

data curnnetvar_s;
set SOU_FOR.curnnetvar_stdtrain (firstobs=&loop obs=&loop);
run;

data neuraltestdataFP;
   merge curnnetvar_s neuraltestdataFP;
   run;
   
%FRONTPROPAGATION(fpdsn=neuraltestdataFP, fpdsnout=neuraltestdataFP); 

%BACKPROPAGATION(bpdsn=neuraltestdataFP, bpdsnout=neuraltestdataBP);

data neuraltestdataFP;
set neuraltestdataBP(drop= &inputvarname resp_dv);
run;

        %end;

/* Saving each iteration weights to use in subsequent batches*/
 data SOU_FOR.neuraltestdataFPTRAIN_ITR&ITER;
 set neuraltestdataBP(drop= &inputvarname RESP_DV);
 run;
 
 /*Dropping input variables and response variables*/
 data neuraltestdataFPTRAIN_ITR1;
 set neuraltestdataBP(drop= &inputvarname RESP_DV);
 run;
 
 /* Saving the weights in the separate dataset after each iteration with name of dataset as 
 neuralWeightsTrain_ITR1,neuralWeightsTrain_ITR2 etc*/
 
  data SOU_FOR.neuralWeightsTrain_ITR&ITER;
 set neuraltestdataFPTRAIN_ITR1(keep= &hidden_weights_name &hidden_interceptname &output_weights_name &out_interceptname);
 run;
 
 /* concatenating  all the datasets with weights and saved the dataset with 
 name NEURALWeightsAllIterations_TRAIN*/
 
 data SOU_FOR.NEURALWeightsAllIterations_TRAIN;
 set SOU_FOR.neuralWeightsTRAIN_ITR1-SOU_FOR.neuralWeightsTRAIN_ITR&ITER;
 run;
 
 /* Front propogating the last observation weight on whole training dataset*/
 data neuraltestdataFPTRAIN_ITR1(drop=i);
 set neuraltestdataFPTRAIN_ITR1;
   do i = 1 to &TRAINNUMOBS;
      output;
      end;
      run;
      data neuraltestdataFPTRAIN_ITR1;
     /* set neuraltestdataFPTRAIN_ITR1;*/
      merge SOU_FOR.curnnetvar_stdtrain  neuraltestdataFPTRAIN_ITR1;
      run;
       %FRONTPROPAGATION(fpdsn=neuraltestdataFPTRAIN_ITR1, fpdsnout=neuraltestdataFPTRAIN_FinaLITR&ITER); 

/* calculating the Fit statisics on training dataset*/
proc sql;
create table SOU_FOR.NEURALTRAINITER&ITER as
select mean(nneterrorsq) as NNASE, sum(NNCorrect )/count(*) as NNAccuracy, 
 (1-(sum(NNCorrect )/count(*))) as NNMCR from neuraltestdataFPTRAIN_FinaLITR&ITER;		
quit;

/* Appending all Iterations Fit Statistics into one datset*/ 
data SOU_FOR.NEURALTRAINSUMMARY;
 set SOU_FOR.NEURALTRAINITER1-SOU_FOR.NEURALTRAINITER&ITER;
 run;
 
 /* Merging the Weights and Fit Statistics into dataset for all Iteration*/
  data SOU_FOR.NNIterationsWeights_TRAINSummary;
 merge SOU_FOR.NEURALWeightsAllIterations_TRAIN SOU_FOR.NEURALTRAINSUMMARY;
 run;
 
 /*Dropping input variables and response variables*/
 data neuraltestdataFPVALID_ITR1;
 set neuraltestdataBP(drop= &inputvarname RESP_DV);
 run;
 
  /* Saving the weights in the separate dataset after each iteration with name of dataset as 
 neuralWeightsValid_ITR1,neuralWeightsValid_ITR2 etc*/
 data SOU_FOR.neuralWeightsValid_ITR&ITER;
 set neuraltestdataFPVALID_ITR1(keep= &hidden_weights_name &hidden_interceptname &output_weights_name &out_interceptname);
 run;
  
  /* concatenating  all the datasets with weights and saved the dataset with 
 name NEURALWeightsAllIterations_Valid*/
 data SOU_FOR.NEURALWeightsAllIterations_Valid;
 set SOU_FOR.neuralWeightsValid_ITR1-SOU_FOR.neuralWeightsValid_ITR&ITER;
 run; 

 /* Front propogating the last observation weight on whole validation dataset*/
 data neuraltestdataFPVALID_ITR1(drop=i);
 set neuraltestdataFPVALID_ITR1;
   do i = 1 to &VALIDNUMOBS;
      output;
      end;
      run;
     
  data neuraltestdataFPVALID_ITR1;
  merge SOU_FOR.curnnetvar_stdvalid neuraltestdataFPVALID_ITR1;
  run;
       
 %FRONTPROPAGATION(fpdsn=neuraltestdataFPVALID_ITR1, fpdsnout= neuraltestdataFPVALID_FinaLITR&ITER);
  
/* calculating the Fit statisics on validation dataset*/
proc sql;
create table SOU_FOR.NEURALVALIDITER&ITER as
select mean(nneterrorsq) as NNASE, sum(NNCorrect )/count(*) as NNAccuracy, 
 (1-(sum(NNCorrect )/count(*))) as NNMCR from neuraltestdataFPVALID_FinaLITR&ITER;		
quit;

/* concatening all valid iterations accuracy and MCR into one dataset*/
data SOU_FOR.NEURALVALIDSUMMARY;
 set SOU_FOR.NEURALVALIDITER1-SOU_FOR.NEURALVALIDITER&ITER;
 run;
  data SOU_FOR.NNIterationsWeights_ValidSummary;
 merge SOU_FOR.NEURALWeightsAllIterations_Valid SOU_FOR.NEURALVALIDSUMMARY;
 run;
%end;  
 /* Ending the macro NEURALITERATION */
%MEND;
%NEURALITERATION; 
/*Ending the main macro neuraltest */
%MEND;
/*Running the main macro by passing all parameters*/
%NeuralTest(dsn=curnnetvar1, hiddenextension=_H, hiddennodes=3, outextension=_RESP_DV, outputnodes=1, LR=0.1, StartIteration=4, EndIteration=6);


/* Printing the Fit statistics summary for all iterations in Training dataset*/
Title 'Fit statistics summary for all iterations in Training dataset';
proc print data=SOU_FOR.NEURALTRAINSUMMARY;
run;

/* Printing the Fit statistics summary for all iterations in Validation dataset*/
Title 'Fit statistics summary for all iterations in Validation dataset';
proc print data=SOU_FOR.NEURALVALIDSUMMARY;
run;


/* Printing the Weights Fit statistics summary for all iterations in Training dataset*/
Title ' Weights & Fit statistics summary for all iterations in Training  dataset';
proc print data=SOU_FOR.NNITERATIONSWEIGHTS_TRAINSUMMARY;
run;








 
