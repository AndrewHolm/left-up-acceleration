//File: accel_swipe_left_up.ino
//Author: Andy Holm
//Description: Used to run inference on a pretrained TensorFlow model
//             Determines if the board is being swiped to the left or up
//


// Dependencies
//-----------------------------------------------------------------------------

#include <TensorFlowLite.h>
#include <Arduino.h>
#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "accel_swipe_model.h" // This is the header file containing the model's data (weights and biases)
#include <vector>
#include <cmath>


// prototypes
//-----------------------------------------------------------------------------
void collectSampleData(std::vector<float>& samples);
void normalizeData(std::vector<float>& data);


//variable required by TFLu
//-----------------------------------------------------------------------------

// Model parsed by the TFLu parser
const tflite::Model* model = nullptr;

// The pointer to the interpreter
tflite::MicroInterpreter* interpreter = nullptr; 
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// Input and output tensors
TfLiteTensor* tflu_i_tensor = nullptr;
TfLiteTensor* tflu_o_tensor = nullptr;

// 50hz sample rate (x, y, and directions)
const int SAMPLE_RATE = 50;
const int NUM_AXIS = 3;
const int TOTAL_SAMPLES = NUM_AXIS * SAMPLE_RATE;

// Tensor arena size, memory required by interpreter TFLu does not use dynamic
//  allocation. Arena size is determined by model size through experiments.
const int kTensorArenaSize = 15* 1024;

// allocation of memory 
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Create a resolver to load the model's operators
static tflite::AllOpsResolver resolver;

// Pin number for the onboard LED
const int ledPin = LED_BUILTIN;  


// Setup
//-----------------------------------------------------------------------------

void setup() {

  // Initialize serial communication
  Serial.begin(9600);
  while (!Serial);

  // Initialize accelerometer
  if(!IMU.begin()) {
    Serial.print("Failes to initialize IMU");
    while(1);
  }

  // Set the LED pin as an output
  pinMode(ledPin, OUTPUT);  

  //load the TFLite model from the C-byte array
  model = tflite::GetModel(model_tflite);

  // make sure model schema version is compatible (from tflite website)
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
    "Model provided is schema version %d not equal not equal to supported version "
    "  %d. \n", model->version(), TFLITE_SCHEMA_VERSION);  
  }

  // Initialize TensorFlow Lite
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize); //, error_reporter
  interpreter = &static_interpreter;


  // Allocate memory for the model's input and output tensors
  interpreter->AllocateTensors();

  // Get pointers to the model's input and output tensors
  tflu_i_tensor = interpreter->input(0);
  tflu_o_tensor = interpreter->output(0);

}// end setup


void loop() {

  // Initialize the vector for sample data
  std::vector<float> samples;

  Serial.println("Starting collection in 10 seconds");
  delay(10000); // wait 10 seconds

  // Call the function to collect sample data
  collectSampleData(samples);

  //Print the collected sample data
  Serial.println("Samples: ");
  for (const auto& sample : samples) {
    Serial.print(sample);
    Serial.print(", ");
  } 

  Serial.println(" ");

  // Normalize the data
   normalizeData(samples);

   //Print the normalized data
  Serial.println("Normalized samples: ");
  for (const auto& sample : samples) {
    Serial.print(sample);
    Serial.print(", ");
  } 

  // Copy the normalized data to the input tensor
  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    tflu_i_tensor->data.f[i] = samples[i];
  }

  // run the inference
  interpreter->Invoke();

  //get prediction
  float prob = tflu_o_tensor->data.f[0];
    
  Serial.println(" ");
  Serial.print("Probability ");
  Serial.println(": ");
  Serial.println(prob);

  if(prob < 0.5){
    Serial.print("Left");
  }

  else{
    Serial.print("Up");
  }

  Serial.println(" ");

  // clear vector for next test
  samples.clear();
}// end loop


// Function to normalize the data (zero mean and unit variance)
//-----------------------------------------------------------------------------

  void normalizeData(std::vector<float>& data) {

    // Calculate mean
    float sum = 0.0;
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
      sum += data[i];
    }
    float mean = sum / TOTAL_SAMPLES;

    // Calculate variance
    float variance = 0.0;
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
      variance += std::pow(data[i] - mean, 2);
    }
    variance /= TOTAL_SAMPLES;

    // Calculate standard deviation
    float stdDev = std::sqrt(variance);

    // Normalize the data
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
      float normalizedValue = (data[i] - mean) / stdDev;
      data[i] = normalizedValue;
    }
  }// end normalizeData


//Function to collect the sample data
//-----------------------------------------------------------------------------

  void collectSampleData(std::vector<float>& samples) {
    // Collect sample data (1 second at 50hz)
    int count = 0;
  
    digitalWrite(ledPin, HIGH);   // Turn on the LED

    while (count < SAMPLE_RATE) { 
      
      float x, y, z;
      IMU.readAcceleration(x, y, z);
      samples.push_back(x);
      samples.push_back(y);
      samples.push_back(z);
      count++;

      delay(20); // 50 samples with 20ms delay = ~1 second of data
    }
    
    digitalWrite(ledPin, LOW);   // Turn off the LED

  }// end collectSampleData


  

