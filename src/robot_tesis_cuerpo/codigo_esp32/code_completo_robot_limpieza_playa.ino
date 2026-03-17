// Codigo del control del Robot completo 
/*Nombre: Guillermo Cid Ampuero - Matias Toribio Clarck
Escuela de Ingenieria Electrica PUCV, EIE
2025*/
/*Este codigo considera un factor lineal y angular que se aplicaban al robot cuando 
este presentaba problema de motores, donde esos problemas  ya no existen, se pueden
establecer con un valor de 1 para ambos*/ 
/* Este codigo usa microros, por lo que tambien debes instalar micro ros en tu rpi4
Ademas, debes revisar las variables del encoder, el tamaño de las ruedas entre otras*/

// Librerias
#include <micro_ros_arduino.h>
#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <geometry_msgs/msg/twist.h> //Para los datos de cmd_vel
#include <std_msgs/msg/float32.h> // Para los datos enviados a la odometria

// Validación de Placa

#if !defined(ESP32) && !defined(TARGET_PORTENTA_H7_M7) && !defined(ARDUINO_NANO_RP2040_CONNECT) && !defined(ARDUINO_WIO_TERMINAL)
#error This example is only available for Arduino Portenta, Arduino Nano RP2040 Connect, ESP32 Dev module and Wio Terminal
#endif

///////Tiempos de Muestreo///////////
unsigned long lastTime = 0;  // Tiempo anterior
unsigned long sampleTime = 30;  // Tiempo de muestreo
long currentMillis = 0;

//// DEFINICION DE PINES BTS7960 ////
// --- Motor Derecho ---
int R1_PWM = 32;
int L1_PWM = 26;

// --- Motor Izquierdo ---
int R2_PWM = 4;
int L2_PWM = 5;

// --- Motor Derecho ---
// Para el BTS7960 se usan dos pines: uno para "adelante" y otro para "atrás".
int R3_PWM = 25;  // Canal PWM para avanzar (adelante) en el motor derecho
int L3_PWM = 23;  // Canal PWM para retroceder (atrás) en el motor derecho

// --- Motor Izquierdo ---
// Se usan dos pines: uno para avanzar y otro para retroceder.
int R4_PWM = 12;  // Canal PWM para retroceder (atrás) en el motor izquierdo
int L4_PWM = 13;  // Canal PWM para avanzar (adelante) en el motor izquierdo

// Motor derecho:
int encoder_derecho_amarillo_A = 19;  // Canal A (usado para interrupción)
int encoder_derecho_verde_B    = 14;  // Canal B (se lee para determinar dirección)

// Motor izquierdo:
int encoder_izquierdo_amarillo_A = 18;  // Canal A (interrupción)
int encoder_izquierdo_verde_B    = 27;  // Canal B

// Factores para compensar diferencia de velocidad de los motores
double factor_lineal=6.0; // Factor cuando el robot va hacia delante y atras
double factor_angular=3.3; // Factor cuando el robot gira


///// Variable para recepcion de datos desde ROS ////
float w_rueda_derecha_entrada=0.0; //VelAngularDerecha que se genera a partir de los datos enviados por ros
float w_rueda_izquierda_entrada=0.0; //VelAngularIzquierda que se genera a partir de los datos enviados por ros
float linear=0.0; // Variable para recibir cmd_vel.linear.x desde ros
float angular=0.0; // Variable para recibir cmd_vel.angular.z desde ros

///// VARIABLES PARA ODOMETRIA ////
float x =0.0;
float y=0.0;
float xp=0.0;
float yp=0.0;
float phi=0.0;

float uMedido_total=0.0; // Velocidad Lineal Total
float wMedido_total=0.0; // Velocidad Angular Total

float wMedido_derecho=0.0; // Velocidad Angular Rueda Derecha
float wMedido_izquierdo=0.0; // Velocidad Angular Rueda Izquierda

volatile long pulsos_encoder_derecho=0; // Pulsos para lectura del encoder derecho
volatile long pulsos_encoder_izquierdo=0; // Pulsos para lectura del encoder izquiedo

float distancia_ruedas=0.975; // Distancia entre ruedas En Ros2: En eje x =0.665 Eje y = 0.975
float radio_ruedas=0.21; // Radio de las ruedas

float constValueD=0.9888551002800;// (1000*2*pi)/(Resolucion) ---> Resolucion = 6354 Resolucion encoder cuadruple
float constValueI=0.9888551002800;// (1000*2*pi)/(Resolucion) ---> Resolucion = 6354 Resolucion encoder cuadruple

// Variables para obtener la relacion de las velocidades a PWM
uint16_t left_pwm=0;
uint16_t right_pwm=0;

float left_speed=0.0;
float right_speed=0.0;

/// Suscriptores y Publicadores de MicroROS
rcl_subscription_t subscriber; //Suscriptor a cmd_vel


rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rclc_executor_t executor;

// Publicadores de datos en ROS mediante MicroRos

rcl_publisher_t publisher_x;
rcl_publisher_t publisher_y;
rcl_publisher_t publisher_phi;
rcl_publisher_t publisher_uMeas;
rcl_publisher_t publisher_wMeas;
rcl_publisher_t publisher_wMeas_derecho;
rcl_publisher_t publisher_wMeas_izquierdo;

// Mensaje de MicroRos
geometry_msgs__msg__Twist cmd_vel_msg;
std_msgs__msg__Float32 x_msg;
std_msgs__msg__Float32 y_msg;
std_msgs__msg__Float32 phi_msg;
std_msgs__msg__Float32 uMeas_msg;
std_msgs__msg__Float32 wMeas_msg;
std_msgs__msg__Float32 wMeas_msg_derecho;
std_msgs__msg__Float32 wMeas_msg_izquierdo;

// Validacion de inicio
#define LED_PIN 13

#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){} }

void error_loop() {
  while (1) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    delay(100);
  }
}

// Function to map a value from one range to another
float fmap(float x, float in_min, float in_max, float out_min, float out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

// Callback to handle incoming cmd_vel messages
void cmd_vel_callback(const void * msg_in) {
  const geometry_msgs__msg__Twist * msg = (const geometry_msgs__msg__Twist *)msg_in;

  linear = msg->linear.x;
  angular = msg->angular.z;

  // https://wiki.ros.org/diff_drive_controller

  // Convert the linear and angular speeds to left and right speeds
  left_speed = (linear - angular) / 2.0; // left_speed = linear - angular / 2 (Diff Drive Robot)
  right_speed = (linear + angular) / 2.0; // right_speed = linear + angular / 2 (Diff Drive Robot)

  left_pwm = (uint16_t) fmap((left_speed), -1.0, 1.0, 180, 255);
  right_pwm = (uint16_t) fmap((right_speed), -1.0, 1.0, 180, 255);

}

// Funcion para leer el encoder derecho
void IRAM_ATTR encoder_derecho_isr() {
    int state = digitalRead(encoder_derecho_verde_B);
    if (state== HIGH){
      pulsos_encoder_derecho --;   
    }else{
      pulsos_encoder_derecho ++;
  }
}

// Funcion para leer el encoder izquierdo
void IRAM_ATTR encoder_izquierdo_isr() {
    int state = digitalRead(encoder_izquierdo_verde_B);
    if (state== HIGH){
      pulsos_encoder_izquierdo ++;   
    }else{
      pulsos_encoder_izquierdo --;
  }
}
// Funcion para calcular la Velocidad angular derecha
void medicion_vel_angular_derecho(){
  wMedido_derecho = constValueD*pulsos_encoder_derecho/(millis()-lastTime);
  pulsos_encoder_derecho=0;
}

// Funcion para calcular la Velocidad angular izquierda
void medicion_vel_angular_izquierdo(){
  wMedido_izquierdo = constValueI*pulsos_encoder_izquierdo/(millis()-lastTime);
  pulsos_encoder_izquierdo=0;
}

//https://dl.espressif.com/dl/package_esp32_index.json
void setup() {
  // Initialize WiFi transport
 set_microros_wifi_transports("ferp24g", "24gfiseieid23i10249", "192.168.1.103", 8888); // Credenciales del router
  /* Como dato: si deseas agregar otra ESP32 con este protocolo, basta con que cambies el puertoo 8888 a otro puerto
  y ya puedes tener publicacion de datos en ROS2 desde otra esp32*/
  //set_microros_wifi_transports("Lab_robotica", "roboticaeie", "192.168.0.143", 8888);

  //set_microros_wifi_transports("VTR-1865041", "bmZbkqvH4fwm", "192.168.0.19", 8888);
  // Estado de los pines
  pinMode(LED_PIN, OUTPUT);
  pinMode(R1_PWM,OUTPUT);
  pinMode(L1_PWM,OUTPUT);
  pinMode(R2_PWM,OUTPUT);
  pinMode(L2_PWM,OUTPUT);
  pinMode(R3_PWM,OUTPUT);
  pinMode(L3_PWM,OUTPUT);
  pinMode(R4_PWM,OUTPUT);
  pinMode(L4_PWM,OUTPUT);

  pinMode(encoder_derecho_amarillo_A, INPUT_PULLUP);
  pinMode(encoder_derecho_verde_B, INPUT_PULLUP);
  pinMode(encoder_izquierdo_amarillo_A, INPUT_PULLUP);
  pinMode(encoder_izquierdo_verde_B, INPUT_PULLUP);

  // Canales para trababajar con PWM
  ledcSetup(0, 1000, 8); // Canal 0 para R1_PWM
  ledcAttachPin(R1_PWM, 0);

  ledcSetup(1, 1000, 8); // Canal 1 para L1_PWM
  ledcAttachPin(L1_PWM, 1);

  ledcSetup(2, 1000, 8); // Canal 2 para R2_PWM
  ledcAttachPin(R2_PWM, 2);

  ledcSetup(3, 1000, 8); // Canal 3 para L2_PWM
  ledcAttachPin(L2_PWM, 3);

  ledcSetup(4, 1000, 8); // Canal 4 para R3_PWM
  ledcAttachPin(R3_PWM, 4);

  ledcSetup(5, 1000, 8); // Canal 5 para L3_PWM
  ledcAttachPin(L3_PWM, 5);

  ledcSetup(6, 1000, 8); // Canal 6 para R4_PWM
  ledcAttachPin(R4_PWM, 6);

  ledcSetup(7, 1000, 8); // Canal 7 para L4_PWM
  ledcAttachPin(L4_PWM, 7);

  // Interrupciones para los encoder
  attachInterrupt(digitalPinToInterrupt(encoder_derecho_amarillo_A), encoder_derecho_isr, RISING); // Configurar la interrupción
  attachInterrupt(digitalPinToInterrupt(encoder_izquierdo_amarillo_A), encoder_izquierdo_isr, RISING); // Configurar la interrupción

  
  lastTime = millis();
  digitalWrite(LED_PIN, HIGH);

  Serial.begin(115200);
  delay(2000);

  allocator = rcl_get_default_allocator();

  // Create init_options
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));

  // Create node
  RCCHECK(rclc_node_init_default(&node, "micro_ros_arduino_wifi_node", "", &support));

  // Create subscriber
  RCCHECK(rclc_subscription_init_default(
    &subscriber,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist),
    "/cmd_vel"));

      // Create executor
  RCCHECK(rclc_executor_init(&executor, &support.context, 1, &allocator));
  RCCHECK(rclc_executor_add_subscription(&executor, &subscriber, &cmd_vel_msg, &cmd_vel_callback, ON_NEW_DATA));
  
  RCCHECK(rclc_publisher_init_default(&publisher_x, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "x"));
  RCCHECK(rclc_publisher_init_default(&publisher_y, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "y"));
  RCCHECK(rclc_publisher_init_default(&publisher_phi, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "phi"));
  RCCHECK(rclc_publisher_init_default(&publisher_uMeas, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "uMeas"));
  RCCHECK(rclc_publisher_init_default(&publisher_wMeas, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "wMeas"));
  RCCHECK(rclc_publisher_init_default(&publisher_wMeas_derecho, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "wMeas_derecho"));
  RCCHECK(rclc_publisher_init_default(&publisher_wMeas_izquierdo, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "wMeas_izquierdo"));

  
}

void loop() {
  // Spin the executor to process callbacks

  if(millis()-lastTime >= sampleTime){
    rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100));

  // Logica para el control del motor
  if (right_speed > 0 && left_speed > 0){
    giroHorario(0,1,right_pwm/factor_lineal,0);
    giroHorario(4,5,right_pwm,0);
    giroAntihorario(2,3,0,left_pwm/factor_lineal);
    giroAntihorario(6,7,0,left_pwm); 

  } else if (right_speed < 0 && left_speed < 0){
    giroAntihorario(0,1,0,right_pwm/factor_lineal);
    giroAntihorario(4,5,0,right_pwm);
    giroHorario(2,3,left_pwm/factor_lineal,0);
    giroHorario(6,7,left_pwm,0); 

  } else if (right_speed > 0 && left_speed < 0){
    giroHorario(0,1,right_pwm/factor_angular,0);
    giroHorario(4,5,right_pwm,0);
    giroHorario(2,3,left_pwm/factor_angular,0);
    giroHorario(6,7,left_pwm,0);

  } else if (right_speed < 0 && left_speed > 0){
    giroAntihorario(0,1,0,right_pwm/factor_angular);
    giroAntihorario(4,5,0,right_pwm);
    giroAntihorario(2,3,0,left_pwm/factor_angular);
    giroAntihorario(6,7,0,left_pwm); 

  }else {
    parar(0,1,0);
    parar(4,5,0);
    parar(2,3,0);
    parar(6,7,0);
  }

    
    medicion_vel_angular_derecho();    
    medicion_vel_angular_izquierdo();
    lastTime = millis();
    
    uMedido_total = (radio_ruedas * (wMedido_derecho + wMedido_izquierdo)) / 2;
    wMedido_total = (radio_ruedas * (wMedido_derecho - wMedido_izquierdo)) / distancia_ruedas;

    phi = phi+0.1*wMedido_total;
    xp = uMedido_total*cos(phi);
    yp = uMedido_total*sin(phi);

    Serial.print("Left Speed: ");
    Serial.print(left_speed);
    Serial.print(" Right Speed: ");
    Serial.print(right_speed);
    Serial.print("    Left PWM:");
    Serial.print(left_pwm);
    Serial.print(" Right PWM: ");
    Serial.println(right_pwm);
    /*Serial.print("X: ");
    Serial.print(x);
    Serial.print(" Y:");
    Serial.print(y);
    Serial.print(" PHI:");
    Serial.println(phi);*/
      
    x = x + 0.1*xp;
    y = y + 0.1*yp; 
  // Publicar datos 
    x_msg.data =x; //  X
    y_msg.data =y;  //  Y
    phi_msg.data =phi; // Incremento ficticio en orientación
    uMeas_msg.data =uMedido_total; // Velocidad lineal medida
    wMeas_msg.data =wMedido_total; // Velocidad lineal medida
    wMeas_msg_derecho.data =wMedido_derecho; // Velocidad angular medida rueda derecha 
    wMeas_msg_izquierdo.data =wMedido_izquierdo; // Velocidad angular medida rueda izquierda
    
    // Publico los valores
    RCSOFTCHECK(rcl_publish(&publisher_x, &x_msg, NULL));
    RCSOFTCHECK(rcl_publish(&publisher_y, &y_msg, NULL));
    RCSOFTCHECK(rcl_publish(&publisher_phi, &phi_msg, NULL));
    RCSOFTCHECK(rcl_publish(&publisher_uMeas, &uMeas_msg, NULL));
    RCSOFTCHECK(rcl_publish(&publisher_wMeas, &wMeas_msg, NULL));
    RCSOFTCHECK(rcl_publish(&publisher_wMeas_derecho, &wMeas_msg_derecho, NULL));
    RCSOFTCHECK(rcl_publish(&publisher_wMeas_izquierdo, &wMeas_msg_izquierdo, NULL));
}
  
}

// Funciones para el control de los motores
void giroAntihorario(int canal_1,int canal_2, int cv_1,int cv_2)
{
  ledcWrite(canal_1,cv_1);
  ledcWrite(canal_2,cv_2);

}

void giroHorario(int canal_1,int canal_2, int cv_1,int cv_2)
{
  ledcWrite(canal_1,cv_1);
  ledcWrite(canal_2,cv_2);

}

void parar(int canal_1,int canal_2, int cv)
{
  ledcWrite(canal_1,cv);
  ledcWrite(canal_2,cv);

}