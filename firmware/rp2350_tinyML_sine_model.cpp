#include <stdio.h>
#include <math.h>
#include "pico/stdlib.h"
#include "hardware/pwm.h"
#include "model.h"

// Includes do TFLite Micro
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Configurações
#define TENSOR_ARENA_SIZE (8 * 1024)

#define LED_PIN 25

// Global variables for TFLite Micro
namespace
{
    tflite::ErrorReporter *error_reporter = nullptr;
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;
    alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];
}

/**
 * @brief Configures the GPIO pin connected to the LED to use PWM.
 *
 * This function sets up the PWM slice and channel for the specified LED_PIN,
 * configures the wrap value (period) and initial level, and then enables the PWM.
 */
void setup_pwm_led()
{
    // Configura o LED para PWM
    gpio_set_function(LED_PIN, GPIO_FUNC_PWM);
    uint slice_num = pwm_gpio_to_slice_num(LED_PIN);
    pwm_set_wrap(slice_num, 255);                 // Configura o período do PWM
    pwm_set_chan_level(slice_num, PWM_CHAN_A, 0); // Inicializa com LED apagado
    pwm_set_enabled(slice_num, true);             // Habilita o PWM
}

/**
 * @brief Sets the brightness of the LED.
 *
 * @param brightness The brightness value to set, from 0 (off) to 255 (full brightness).
 */
void set_led_brightness(int brightness)
{
    pwm_set_gpio_level(LED_PIN, brightness);
}

/**
 * @brief Main entry point of the application.
 *
 * Initializes the system, sets up the TFLite Micro interpreter with a sine wave model,
 * and enters an infinite loop to run inference. In each iteration, it calculates an
 * input value based on time, runs the model to get a sine wave output, and uses this
 * output to control the brightness of an LED.
 *
 * @return int Returns 0 on successful execution, -1 on failure.
 */
int main()
{
    stdio_init_all();

    //  Initialize the TFLite Micro runtime
    tflite::InitializeTarget();

    // Load the model
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Model provided is schema version %d not equal "
                             "to supported version %d.",
                             model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    // Resolve operations
    static tflite::MicroMutableOpResolver<6> resolver;
    if (resolver.AddFullyConnected() != kTfLiteOk)
    {
        return -1;
    }
    if (resolver.AddRelu() != kTfLiteOk)
    {
        return -1;
    }
    if (resolver.AddQuantize() != kTfLiteOk)
    {
        return -1;
    }
    if (resolver.AddDequantize() != kTfLiteOk)
    {
        return -1;
    }

    // Build the interpreter
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        printf("Failed to allocate tensors\n");
        return -1;
    }

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);

    // LED PWM setup
    setup_pwm_led();

    while (true)
    {
        // Input: simulates a sine wave
        // Get time in milliseconds and convert to seconds
        float position = to_ms_since_boot(get_absolute_time()) / 1000.0f;

        // Multiply to fit the sine wave into the range of the model's input (0 to 2pi)
        // fmod to wrap around the value and prevent overflow. This keeps the input within the expected range for the model.
        float x_val = position * 1.57f;
        x_val = fmod(x_val, 2.0f * 3.14159265359f);

        // Fill input tensor
        // Check if the model expects Int8 (quantized) or Float
        if (input->type == kTfLiteInt8)
        {
            // Convert float to int8 using the model's quantization parameters
            input->data.int8[0] = static_cast<int8_t>(x_val / input->params.scale + input->params.zero_point);
        }
        else
        {
            input->data.f[0] = x_val;
        }

        // Run inference
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk)
        {
            printf("Failed to invoke tflite!\n");
            return -1;
        }

        // Output: read the output tensor
        // Outputs a value between -1 and 1 (sine wave)
        float y_pred = 0.0f;

        // Reads the output tensor, converting from quantized int8 to float if necessary
        if (output->type == kTfLiteInt8)
        {
            y_pred = (output->data.int8[0] - output->params.zero_point) * output->params.scale;
        }
        else
        {
            y_pred = output->data.f[0];
        }

        // Calculate the mathematical ground truth for comparison
        float y_true = sin(x_val);

        // Print data for Serial Plotter
        printf("Pred:%.2f,True:%.2f\n", y_pred, y_true);

        // Calculate brightness for LED visualization
        // Convert range [-1, 1] to [0, 255] for PWM
        int brightness = (int)((y_pred + 1) * 127.5f);

        // Clamp values to valid PWM range
        if (brightness < 0)
            brightness = 0;
        if (brightness > 255)
            brightness = 255;

        // Set LED brightness
        set_led_brightness(brightness);
        
        sleep_ms(20);
    }
}
